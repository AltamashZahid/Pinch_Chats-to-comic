"""Score generated comic panels so prompt changes and fine-tuning are measured, not eyeballed.

    python eval/evaluate_panels.py --tag baseline

For each conversation in eval/conversations.json: classify emotions, write the script,
render the panels, then score them with CLIP (ViT-L/14, the model used by most papers'
"CLIP score"):

- clip_score:      100 * cosine(image, prompt). Does the picture show what was asked for?
- diversity:       1 - mean pairwise cosine between panel images of the same comic.
                   Higher = panels look less alike (the "same room three times" problem).
- expression_acc:  CLIP zero-shot picks which of 7 facial expressions the panel shows;
                   share of panels where that matches the emotion the panel should convey.

Writes reports/panel_eval_<tag>.json and a contact sheet of every comic to eval/out/<tag>/.
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pinch.cache import ImageCache  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from pinch.imagegen import backend_from_env  # noqa: E402
from pinch.layout import compose_page  # noqa: E402
from pinch.llm import LLMClient  # noqa: E402
from pinch.pipeline import classify  # noqa: E402
from pinch.render import render_panels  # noqa: E402
from pinch.scenes import Line, write_script  # noqa: E402

CLIP_MODEL = "openai/clip-vit-large-patch14"
FACE = {"anger": "angry", "disgust": "disgusted", "fear": "scared", "joy": "happy",
        "sadness": "sad", "surprise": "shocked", "neutral": "calm"}


def intended_emotion(lines: list[Line]) -> str:
    felt = [l for l in lines if l.emotion != "neutral"]
    return max(felt, key=lambda l: l.confidence).emotion if felt else "neutral"


class ClipScorer:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).eval()  # CPU: the GPU holds SD
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        faces = [f"a comic drawing of a person with a {FACE[l]} face" for l in LABELS]
        self.face_emb = self.text(faces)

    @torch.no_grad()
    def image(self, paths: list[Path]) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        e = self.model.get_image_features(**self.proc(images=imgs, return_tensors="pt"))
        return torch.nn.functional.normalize(e, dim=-1).numpy()

    @torch.no_grad()
    def text(self, texts: list[str]) -> np.ndarray:
        enc = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        e = self.model.get_text_features(**enc)
        return torch.nn.functional.normalize(e, dim=-1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this run, e.g. baseline, shots, lora")
    args = ap.parse_args()

    convos = json.loads((ROOT / "eval" / "conversations.json").read_text(encoding="utf-8"))
    backend = backend_from_env()
    cache = ImageCache(ROOT / "eval" / "cache")
    llm = LLMClient.from_env()
    out_dir = ROOT / "eval" / "out" / args.tag

    comics, start = [], time.time()
    for c in convos:
        lines = [Line(s, t) for s, t in c["lines"]]
        for l, (e, p) in zip(lines, classify([l.text for l in lines])):
            l.emotion, l.confidence = e, p
        script = write_script(lines, llm)
        rendered = render_panels(script, backend, cache)
        compose_page(rendered, out_dir / f"{c['id']}.png")
        comics.append((c["id"], script.source, rendered))
        print(f"rendered {c['id']}: {len(rendered)} panels")
    render_minutes = (time.time() - start) / 60

    # Free the GPU before loading CLIP.
    del backend
    torch.cuda.empty_cache()
    clip = ClipScorer()

    per_comic, clip_scores, diversities, hits, total = [], [], [], 0, 0
    for cid, source, rendered in comics:
        img = clip.image([r.image for r in rendered])
        txt = clip.text([r.panel.prompt for r in rendered])
        scores = 100 * np.clip((img * txt).sum(1), 0, None)
        pair = [float(img[i] @ img[j]) for i, j in combinations(range(len(img)), 2)]
        div = 1 - float(np.mean(pair)) if pair else 0.0
        predicted = [LABELS[int(i)] for i in (img @ clip.face_emb.T).argmax(1)]
        wanted = [intended_emotion(r.panel.lines) for r in rendered]
        correct = sum(p == w for p, w in zip(predicted, wanted))
        hits, total = hits + correct, total + len(wanted)
        clip_scores += scores.tolist()
        diversities.append(div)
        per_comic.append({"id": cid, "script": source, "clip_score": round(float(scores.mean()), 2),
                          "diversity": round(div, 3), "wanted": wanted, "clip_saw": predicted})
        print(f"{cid:10} clip {scores.mean():5.1f}  diversity {div:.3f}  faces {correct}/{len(wanted)}")

    report = {
        "tag": args.tag,
        "panels": total,
        "clip_score": round(float(np.mean(clip_scores)), 2),
        "diversity": round(float(np.mean(diversities)), 3),
        "expression_acc": round(hits / total, 3),
        "render_minutes": round(render_minutes, 1),
        "comics": per_comic,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"panel_eval_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\n{args.tag}: clip_score {report['clip_score']}  diversity {report['diversity']}  "
          f"expression_acc {report['expression_acc']}  ({total} panels)")


if __name__ == "__main__":
    main()
