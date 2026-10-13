"""Image generation backends. All return PNG/JPEG bytes for a prompt.

    PINCH_IMAGE_BACKEND = local        SD-Turbo on your GPU (default; free, ~1 s/panel on 4 GB)
                        | sdxl         SDXL-Turbo, 4-bit, on your GPU (free, ~8.5 s/panel, better art)
                        | huggingface  FLUX.1-schnell via HF Inference (needs HF_TOKEN, free tier)
                        | pollinations hosted (needs POLLINATIONS_TOKEN since anonymous use returns 402)
                        | placeholder  offline coloured cards, for tests and UI work
"""
import hashlib
import io
import os
import textwrap
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont


class ImageGenError(RuntimeError):
    pass


@dataclass
class ImageRequest:
    prompt: str
    seed: int = 0
    width: int = 512
    height: int = 512
    negative: str = "text, letters, words, speech bubbles, watermark, signature, blurry"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class LocalTurboBackend:
    """SD-Turbo (distilled Stable Diffusion 2.1) with diffusers, fp16 on CUDA.

    Adversarial diffusion distillation lets it make a usable 512x512 image in 1-4 steps
    with no classifier-free guidance, so it fits a 4 GB laptop GPU (~3 GB peak) at ~1 s/image.
    Note: SD-Turbo's licence is non-commercial.

    Optional LoRA (opt-in): PINCH_LORA=<dir from training/train_lora.py>, strength
    PINCH_LORA_SCALE (default 0.5; 1.0 blurs 4-step output). Its name and scale become part
    of `model`, so cached images made with and without it never mix. Off by default: the
    blind A/B test (reports/lora_ab.json) showed +7 points, not statistically significant.
    """
    name = "local"
    base = "stabilityai/sd-turbo"

    def __init__(self, steps: int = 4, lora: str | None = None, lora_scale: float | None = None):
        self.steps = steps
        self._pipe = None
        choice = lora if lora is not None else os.environ.get("PINCH_LORA", "none")
        self.lora = None if choice.lower() == "none" or not Path(choice).exists() else Path(choice)
        self.lora_scale = lora_scale if lora_scale is not None else float(os.environ.get("PINCH_LORA_SCALE", "0.5"))
        self.model = self.base if self.lora is None else f"{self.base}+{self.lora.name}@{self.lora_scale}"

    def _load(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        if not torch.cuda.is_available():
            raise ImageGenError("local backend needs a CUDA GPU; set PINCH_IMAGE_BACKEND to another backend")
        pipe = AutoPipelineForText2Image.from_pretrained(self.base, torch_dtype=torch.float16, variant="fp16")
        if self.lora is not None:
            pipe.load_lora_weights(str(self.lora))
            pipe.fuse_lora(lora_scale=self.lora_scale)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self._torch, self._pipe = torch, pipe

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cuda").manual_seed(req.seed)
        img = self._pipe(req.prompt, num_inference_steps=self.steps, guidance_scale=0.0,
                         generator=g, width=req.width, height=req.height).images[0]
        return _png(img)


class LocalLCMBackend:
    """Stable Diffusion 1.5 + LCM-LoRA (latent consistency distillation as a LoRA), fp16 on CUDA.

    Unlike SD-Turbo, SD 1.5 fine-tunes cleanly with ordinary LoRA training, and LCM-LoRA (stacked
    on top) still makes it draw in 4-6 steps. An expression LoRA trained on SD 1.5
    (training/train_lora.py --base sd15) is loaded from PINCH_LORA when set.
    """
    name = "lcm"
    base = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lcm = "latent-consistency/lcm-lora-sdv1-5"

    def __init__(self, steps: int = 6, lora: str | None = None, lora_scale: float | None = None):
        self.steps = steps
        self._pipe = None
        choice = lora if lora is not None else os.environ.get("PINCH_LORA", "none")
        self.lora = None if choice.lower() == "none" or not Path(choice).exists() else Path(choice)
        self.lora_scale = lora_scale if lora_scale is not None else float(os.environ.get("PINCH_LORA_SCALE", "1.0"))
        self.model = f"sd15+lcm" + ("" if self.lora is None else f"+{self.lora.name}@{self.lora_scale}")

    def _load(self):
        import torch
        from diffusers import LCMScheduler, StableDiffusionPipeline
        if not torch.cuda.is_available():
            raise ImageGenError("lcm backend needs a CUDA GPU")
        pipe = StableDiffusionPipeline.from_pretrained(self.base, torch_dtype=torch.float16, variant="fp16",
                                                       safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(self.lcm, adapter_name="lcm")
        names, weights = ["lcm"], [1.0]
        if self.lora is not None:
            pipe.load_lora_weights(str(self.lora), adapter_name="expr")
            names.append("expr")
            weights.append(self.lora_scale)
        pipe.set_adapters(names, adapter_weights=weights)
        pipe.fuse_lora()
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self._torch, self._pipe = torch, pipe

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cuda").manual_seed(req.seed)
        img = self._pipe(req.prompt, num_inference_steps=self.steps, guidance_scale=1.0,
                         generator=g, width=req.width, height=req.height).images[0]
        return _png(img)


class LocalSDXLTurboBackend:
    """SDXL-Turbo on a 4 GB GPU: UNet quantised to 4-bit NF4 (bitsandbytes, ~1.5 GB) and
    each component moved to the GPU only while it runs (model CPU offload).

    Benchmarked on an RTX 2050 (eval/bench_sdxl_turbo.py): NF4 + offload 8.5 s/image at
    1.9 GB peak; everything on the GPU spills past 4 GB and drops to 15 s; 8-bit UNet spills
    too (21 s) and produced broken images. Uses the fp16-fixed SDXL VAE (the stock one
    overflows in half precision). Licence: non-commercial (SDXL-Turbo research licence).
    """
    name = "sdxl"
    model = "stabilityai/sdxl-turbo@nf4"
    base = "stabilityai/sdxl-turbo"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"

    def __init__(self, steps: int = 4):
        self.steps = steps
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import AutoencoderKL, AutoPipelineForText2Image, BitsAndBytesConfig, UNet2DConditionModel
        if not torch.cuda.is_available():
            raise ImageGenError("sdxl backend needs a CUDA GPU; set PINCH_IMAGE_BACKEND to another backend")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(self.base, subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float16, quantization_config=q)
        vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=torch.float16)
        pipe = AutoPipelineForText2Image.from_pretrained(self.base, unet=unet, vae=vae,
                                                         torch_dtype=torch.float16, variant="fp16")
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self._torch, self._pipe = torch, pipe

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cpu").manual_seed(req.seed)
        img = self._pipe(req.prompt, num_inference_steps=self.steps, guidance_scale=0.0,
                         generator=g, width=req.width, height=req.height).images[0]
        return _png(img)


class HuggingFaceBackend:
    """FLUX.1-schnell through Hugging Face's serverless inference (free monthly credits)."""
    name = "huggingface"
    model = "black-forest-labs/FLUX.1-schnell"

    def __init__(self, token: str | None = None, timeout: float = 180.0,
                 transport: httpx.BaseTransport | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ImageGenError("huggingface backend needs HF_TOKEN (huggingface.co/settings/tokens)")
        self.timeout, self.transport = timeout, transport

    def generate(self, req: ImageRequest) -> bytes:
        url = f"https://router.huggingface.co/hf-inference/models/{self.model}"
        body = {"inputs": req.prompt, "parameters": {
            "width": req.width, "height": req.height, "seed": req.seed,
            "num_inference_steps": 4, "negative_prompt": req.negative}}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                r = http.post(url, json=body, headers={"Authorization": f"Bearer {self.token}"})
        except httpx.HTTPError as e:
            raise ImageGenError(f"huggingface request failed: {e}") from e
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            raise ImageGenError(f"huggingface returned {r.status_code}: {r.text[:200]}")
        return r.content


class PollinationsBackend:
    """Hosted diffusion over plain HTTP GET. Anonymous requests now get 402, so it
    needs POLLINATIONS_TOKEN from a (free) Pollinations account."""
    name = "pollinations"
    model = "flux"

    def __init__(self, token: str | None = None, timeout: float = 180.0,
                 transport: httpx.BaseTransport | None = None):
        self.token = token or os.environ.get("POLLINATIONS_TOKEN")
        self.timeout = timeout
        self.transport = transport

    def generate(self, req: ImageRequest) -> bytes:
        url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(req.prompt, safe="")
        params = {"width": req.width, "height": req.height, "seed": req.seed,
                  "model": self.model, "nologo": "true", "negative": req.negative}
        try:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            with httpx.Client(timeout=self.timeout, transport=self.transport,
                              follow_redirects=True) as http:
                r = http.get(url, params=params, headers=headers)
        except httpx.HTTPError as e:
            raise ImageGenError(f"pollinations request failed: {e}") from e
        if r.status_code == 402:
            raise ImageGenError("pollinations returned 402 Payment Required: set POLLINATIONS_TOKEN "
                                "or use PINCH_IMAGE_BACKEND=local")
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            raise ImageGenError(f"pollinations returned {r.status_code}: {r.text[:200]}")
        return r.content


class PlaceholderBackend:
    """Offline stand-in: a flat-coloured card with the prompt written on it."""
    name = "placeholder"
    model = "placeholder"

    def generate(self, req: ImageRequest) -> bytes:
        digest = hashlib.sha256(f"{req.prompt}|{req.seed}".encode()).digest()
        img = Image.new("RGB", (req.width, req.height), tuple(80 + b % 150 for b in digest[:3]))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default(size=22)
        d.multiline_text((20, 20), "\n".join(textwrap.wrap(req.prompt, 50)[:20]),
                         font=font, fill=(255, 255, 255))
        return _png(img)


BACKENDS = {"local": LocalTurboBackend, "sdxl": LocalSDXLTurboBackend, "lcm": LocalLCMBackend, "huggingface": HuggingFaceBackend,
            "pollinations": PollinationsBackend, "placeholder": PlaceholderBackend}


def backend_from_env():
    name = os.environ.get("PINCH_IMAGE_BACKEND", "local").lower()
    if name not in BACKENDS:
        raise ImageGenError(f"unknown image backend {name!r}; pick one of {sorted(BACKENDS)}")
    return BACKENDS[name]()
