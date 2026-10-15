"""Image generation backends. All return PNG/JPEG bytes for a prompt.

    PINCH_IMAGE_BACKEND = local        SD-Turbo on your GPU (default; free, ~1 s/panel on 4 GB)
                        | sdxl         SDXL-Turbo, 4-bit, on your GPU (free, ~8.5 s/panel, better art)
                        | anime        Animagine XL 4.0, 4-bit, on your GPU (Danbooru expression tags; slow)
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
    emotion: str | None = None   # used by backends that control the face directly (faceshape)
    shot: str | None = None


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

    def boost_for(self, emotion: str | None) -> float:
        """Per-emotion steering strength: PINCH_EMOTION_BOOSTS='{"fear": 0.3, ...}' (JSON),
        else the single PINCH_EMOTION_BOOST, else 0 (off)."""
        import json
        if not emotion or emotion == "neutral":
            return 0.0
        per = json.loads(os.environ.get("PINCH_EMOTION_BOOSTS", "{}") or "{}")
        return float(per.get(emotion, os.environ.get("PINCH_EMOTION_BOOST", "0")))

    def direction(self, emotion: str):
        """Emotion direction in text-embedding space ("terrified face" - "calm face"), cached."""
        if not hasattr(self, "_dirs"):
            self._dirs = {}
        if emotion not in self._dirs:
            from .style import ART_STYLE, EXPRESSION
            emb = lambda t: self._pipe.encode_prompt(t, "cuda", 1, False)[0]  # noqa: E731
            self._dirs[emotion] = emb(f"{ART_STYLE}, close-up, a person, {EXPRESSION[emotion]}") -                 emb(f"{ART_STYLE}, close-up, a person, {EXPRESSION['neutral']}")
        return self._dirs[emotion]

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cuda").manual_seed(req.seed)
        boost = self.boost_for(req.emotion)
        if boost:
            emb = self._pipe.encode_prompt(req.prompt, "cuda", 1, False)[0] + boost * self.direction(req.emotion)
            img = self._pipe(prompt_embeds=emb, num_inference_steps=self.steps, guidance_scale=0.0,
                             generator=g, width=req.width, height=req.height).images[0]
        else:
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

    def __init__(self, steps: int = 6, lora: str | None = None, lora_scale: float | None = None,
                 boost: float | None = None):
        self.steps = steps
        self._pipe = None
        choice = lora if lora is not None else os.environ.get("PINCH_LORA", "none")
        self.lora = None if choice.lower() == "none" or not Path(choice).exists() else Path(choice)
        self.lora_scale = lora_scale if lora_scale is not None else float(os.environ.get("PINCH_LORA_SCALE", "1.0"))
        # Emotion steering: add boost * (embedding of the emotion's face cues - embedding of a calm
        # face) to the prompt embedding, the "king - man + woman" trick applied to expressions.
        self.boost = boost if boost is not None else float(os.environ.get("PINCH_EMOTION_BOOST", "0"))
        self._dirs: dict[str, object] = {}
        self.model = f"sd15+lcm" + ("" if self.lora is None else f"+{self.lora.name}@{self.lora_scale}")             + (f"+boost{self.boost}" if self.boost else "")

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

    def _embed(self, text: str):
        emb, _ = self._pipe.encode_prompt(text, "cuda", 1, False)
        return emb

    def direction(self, emotion: str):
        """Emotion direction in text-embedding space (cached per emotion)."""
        if emotion not in self._dirs:
            from .style import ART_STYLE, EXPRESSION
            calm = self._embed(f"{ART_STYLE}, close-up, a person, {EXPRESSION['neutral']}")
            felt = self._embed(f"{ART_STYLE}, close-up, a person, {EXPRESSION[emotion]}")
            self._dirs[emotion] = felt - calm
        return self._dirs[emotion]

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cuda").manual_seed(req.seed)
        if self.boost and req.emotion and req.emotion != "neutral":
            emb = self._embed(req.prompt) + self.boost * self.direction(req.emotion)
            img = self._pipe(prompt_embeds=emb, num_inference_steps=self.steps, guidance_scale=1.0,
                             generator=g, width=req.width, height=req.height).images[0]
        else:
            img = self._pipe(req.prompt, num_inference_steps=self.steps, guidance_scale=1.0,
                             generator=g, width=req.width, height=req.height).images[0]
        return _png(img)


class LocalFaceControlBackend(LocalLCMBackend):
    """SD 1.5 + LCM-LoRA + a face-landmark ControlNet: the wanted expression is drawn as a
    landmark sketch (pinch/face_control.py, templates from our labelled panels) and the
    ControlNet makes the face follow it. Falls back to plain prompting without an emotion."""
    name = "faceshape"
    controlnet = "CrucibleAI/ControlNetMediaPipeFace"

    def __init__(self, steps: int = 6, lora: str | None = None, lora_scale: float | None = None,
                 control_scale: float | None = None):
        super().__init__(steps, lora, lora_scale)
        self.control_scale = control_scale if control_scale is not None else             float(os.environ.get("PINCH_FACE_CONTROL", "0.9"))
        self.model = self.model.replace("sd15+lcm", f"sd15+lcm+faceshape@{self.control_scale}")

    def _load(self):
        import torch
        from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline
        if not torch.cuda.is_available():
            raise ImageGenError("faceshape backend needs a CUDA GPU")
        cn = ControlNetModel.from_pretrained(self.controlnet, subfolder="diffusion_sd15", torch_dtype=torch.float16,
                                             variant="fp16")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
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
        import random
        from PIL import Image
        from .face_control import load_template, sketch
        if self._pipe is None:
            self._load()
        tmpl = load_template(req.emotion, random.Random(req.seed)) if req.emotion else None
        cond = sketch(tmpl, req.width, req.shot or "close-up") if tmpl is not None else             Image.new("RGB", (req.width, req.height))
        scale = self.control_scale if tmpl is not None else 0.0
        g = self._torch.Generator("cuda").manual_seed(req.seed)
        img = self._pipe(req.prompt, image=cond, controlnet_conditioning_scale=scale,
                         num_inference_steps=self.steps, guidance_scale=1.0, generator=g,
                         width=req.width, height=req.height).images[0]
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


class LocalAnimeBackend:
    """Animagine XL 4.0 (anime SDXL fine-tune, trained on Danbooru-tagged images), 4-bit NF4
    UNet + model CPU offload, like the sdxl backend. Licence: openrail++ (commercial use ok).

    Why: SD-Turbo draws mild faces for surprise/disgust/fear (2/8, 4/8, 5/8 in the blind test).
    Danbooru taggers label expressions precisely ("disgust", "constricted pupils", "anger vein"),
    so a model trained on those tags has seen thousands of examples of each. The emotion's tags
    go first so they survive CLIP's 77-token cut; quality tags follow the model card.
    Not distilled: needs ~20 steps with guidance, so it is much slower than SD-Turbo.
    PINCH_ANIME_STEPS (default 20), PINCH_ANIME_SIZE (default 768; SDXL breaks down at 512).
    """
    name = "anime"
    base = "cagliostrolab/animagine-xl-4.0"
    quality = "anime coloring, masterpiece, high score, great score, absurdres"
    negative = ("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, "
                "cropped, worst quality, low quality, low score, bad score, average score, signature, "
                "watermark, username, blurry, speech bubble, multiple views")

    def __init__(self, steps: int | None = None, size: int | None = None, **_):
        self.steps = steps or int(os.environ.get("PINCH_ANIME_STEPS", "20"))
        self.size = size or int(os.environ.get("PINCH_ANIME_SIZE", "768"))
        self.model = f"{self.base}@nf4/{self.steps}steps/{self.size}"
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import (BitsAndBytesConfig, EulerAncestralDiscreteScheduler,
                               StableDiffusionXLPipeline, UNet2DConditionModel)
        if not torch.cuda.is_available():
            raise ImageGenError("anime backend needs a CUDA GPU; set PINCH_IMAGE_BACKEND to another backend")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        # The hub's single 5.1 GB UNet file can't be memory-mapped with a small pagefile on
        # Windows; training/shard_unet.py re-saves it as 1 GB shards, used when present.
        local = Path(__file__).resolve().parent.parent / "models" / "animagine_unet"
        src = {"pretrained_model_name_or_path": str(local)} if (local / "config.json").exists() else             {"pretrained_model_name_or_path": self.base, "subfolder": "unet"}
        unet = UNet2DConditionModel.from_pretrained(**src, torch_dtype=torch.float16, quantization_config=q)
        pipe = StableDiffusionXLPipeline.from_pretrained(self.base, unet=unet, torch_dtype=torch.float16)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._torch, self._pipe = torch, pipe

    def prompt(self, req: ImageRequest) -> str:
        from .style import ANIME_EXPRESSION, ART_STYLE
        rest = req.prompt.removeprefix(ART_STYLE).lstrip(", ")
        tags = ANIME_EXPRESSION.get(req.emotion or "", "")
        return ", ".join(p for p in (self.gender_tag(rest), "solo", tags, rest, self.quality) if p)

    @staticmethod
    def gender_tag(text: str) -> str:
        """Danbooru models need "1boy"/"1girl": without it a "young man" was drawn as a girl
        in 3 of 7 test images."""
        import re
        words = set(re.findall(r"[a-z]+", text.lower()))
        if words & {"woman", "girl", "she", "her", "lady", "female"}:
            return "1girl"
        if words & {"man", "boy", "he", "his", "guy", "male"}:
            return "1boy"
        return ""

    def generate(self, req: ImageRequest) -> bytes:
        if self._pipe is None:
            self._load()
        g = self._torch.Generator("cpu").manual_seed(req.seed)
        img = self._pipe(self.prompt(req), negative_prompt=self.negative, num_inference_steps=self.steps,
                         guidance_scale=5.0, generator=g, width=self.size, height=self.size).images[0]
        if img.size != (req.width, req.height):
            img = img.resize((req.width, req.height), Image.LANCZOS)
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


BACKENDS = {"local": LocalTurboBackend, "sdxl": LocalSDXLTurboBackend, "lcm": LocalLCMBackend,
            "anime": LocalAnimeBackend,
            "faceshape": LocalFaceControlBackend, "huggingface": HuggingFaceBackend,
            "pollinations": PollinationsBackend, "placeholder": PlaceholderBackend}


def backend_from_env():
    name = os.environ.get("PINCH_IMAGE_BACKEND", "local").lower()
    if name not in BACKENDS:
        raise ImageGenError(f"unknown image backend {name!r}; pick one of {sorted(BACKENDS)}")
    return BACKENDS[name]()
