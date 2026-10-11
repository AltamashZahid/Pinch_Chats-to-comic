"""Image generation backends. All return PNG/JPEG bytes for a prompt.

    PINCH_IMAGE_BACKEND = local        SD-Turbo on your GPU (default; free, ~1 s/panel on 4 GB)
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
    """
    name = "local"
    model = "stabilityai/sd-turbo"

    def __init__(self, steps: int = 4):
        self.steps = steps
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        if not torch.cuda.is_available():
            raise ImageGenError("local backend needs a CUDA GPU; set PINCH_IMAGE_BACKEND to another backend")
        pipe = AutoPipelineForText2Image.from_pretrained(self.model, torch_dtype=torch.float16, variant="fp16")
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


BACKENDS = {"local": LocalTurboBackend, "huggingface": HuggingFaceBackend,
            "pollinations": PollinationsBackend, "placeholder": PlaceholderBackend}


def backend_from_env():
    name = os.environ.get("PINCH_IMAGE_BACKEND", "local").lower()
    if name not in BACKENDS:
        raise ImageGenError(f"unknown image backend {name!r}; pick one of {sorted(BACKENDS)}")
    return BACKENDS[name]()
