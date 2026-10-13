"""Score how clearly a generated panel shows each emotion.

CLIP ViT-L/14 image embedding + a logistic head trained on 700 blind-labelled Pinch panels
(training/train_pool_judge.py). 5-fold CV accuracy 64% vs 56% for zero-shot CLIP; used for
best-of-N selection it raised correct expressions from 55% (1 image) to 74% (best of 4).
Device: PINCH_JUDGE_DEVICE (default cpu). On a 4 GB GPU, putting it beside SD-Turbo pushes
memory to 3.8 GB and slows generation (16 s/panel for best of 4); on the CPU it scores the
4 candidates in one ~5 s batch.
"""
import io
from pathlib import Path

import numpy as np

import os

_MODELS = Path(__file__).resolve().parent.parent / "models"
# A judge must be trained on the same image model it judges (PINCH_JUDGE_HEAD overrides).
HEADS = {"local": _MODELS / "pool_judge" / "head.npz", "lcm": _MODELS / "pool_judge_lcm" / "head.npz"}
HEAD = HEADS["local"]
CLIP = "openai/clip-vit-large-patch14"


class ExpressionJudge:
    def __init__(self, head: Path = HEAD):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._torch = torch
        import os
        want = os.environ.get("PINCH_JUDGE_DEVICE", "cpu")
        self.device = want if want == "cpu" or torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(CLIP, torch_dtype=self.dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP)
        h = np.load(head)
        self.coef, self.intercept = h["coef"], h["intercept"]
        self.classes = [str(c) for c in h["classes"]]

    def probs(self, images: list[bytes]) -> list[dict[str, float]]:
        from PIL import Image
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        with self._torch.no_grad():
            px = self.proc(images=pil, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            e = self._torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        z = e @ self.coef.T + self.intercept
        z = np.exp(z - z.max(1, keepdims=True))
        z /= z.sum(1, keepdims=True)
        return [dict(zip(self.classes, map(float, row))) for row in z]

    def best(self, images: list[bytes], emotion: str) -> tuple[int, float]:
        """Index of the image most likely to show `emotion`, and that probability."""
        scores = [p.get(emotion, 0.0) for p in self.probs(images)]
        i = int(np.argmax(scores))
        return i, scores[i]


_judges: dict[str, ExpressionJudge] = {}


def judge(backend_name: str = "local") -> ExpressionJudge | None:
    """Shared judge for an image backend, or None if none has been trained for it."""
    head = Path(os.environ.get("PINCH_JUDGE_HEAD") or HEADS.get(backend_name, HEAD))
    if not head.exists():
        return None
    if str(head) not in _judges:
        _judges[str(head)] = ExpressionJudge(head)
    return _judges[str(head)]
