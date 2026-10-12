"""Build a LoRA training set of comic panels whose facial expression is clearly correct.

    python training/build_expression_dataset.py --per-emotion 200 --keep 60
    python training/build_expression_dataset.py --probe          # hit rate per emotion only

Self-training (generate -> filter -> fine-tune): SD-Turbo draws many candidates per
emotion with varied characters, shots and places; a CLIP judge keeps only those where the
intended expression clearly wins. A LoRA trained on the kept images learns to produce
those expressions more reliably.

The filter judge (CLIP ViT-B/16) is deliberately a different model from the evaluation
judge (ViT-L/14 in eval/evaluate_panels.py), so the LoRA isn't graded by the same model
that picked its training data.

Output: data/expressions/<emotion>/<n>.png plus data/expressions/metadata.jsonl in the
{"file_name", "text"} format diffusers' LoRA training script expects.
"""
import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from pinch.imagegen import ImageRequest, LocalTurboBackend  # noqa: E402
from pinch.style import ART_STYLE, EXPRESSION, POSE  # noqa: E402

FILTER_MODEL = "openai/clip-vit-base-patch16"
FACE = {"anger": "angry", "disgust": "disgusted", "fear": "scared", "joy": "happy",
        "sadness": "sad", "surprise": "shocked", "neutral": "calm"}
TRIGGER = "pinchface"  # rare token the LoRA binds the "clear expression" look to

LOOKS = [
    "young man with short black hair and a blue hoodie",
    "young woman with curly red hair and a yellow sweater",
    "teenage boy with messy brown hair and a green t-shirt",
    "woman with a long black braid and a denim jacket",
    "older man with grey beard and a flannel shirt",
    "young woman with short blonde bob and a red coat",
    "man with dark skin, shaved head and a white shirt",
    "girl with glasses, ponytail and a striped sweater",
    "man with long hair in a bun and a leather jacket",
    "woman with hijab and a purple cardigan",
]
SHOTS = ["close-up", "extreme close-up", "close-up portrait", "medium close-up"]
PLACES = ["bedroom at night", "coffee shop", "city street", "office desk", "kitchen",
          "bus seat", "living room couch", "park bench", "school hallway", "train station"]


def caption(emotion: str, look: str, shot: str, place: str) -> str:
    return ", ".join([f"{TRIGGER} {ART_STYLE}", shot, f"single character, {look}",
                      EXPRESSION[emotion], POSE[emotion], place])


class Judge:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(FILTER_MODEL).eval()
        self.proc = CLIPProcessor.from_pretrained(FILTER_MODEL)
        enc = self.proc(text=[f"a comic drawing of a person with a {FACE[l]} face" for l in LABELS],
                        return_tensors="pt", padding=True)
        with torch.no_grad():
            t = self.model.text_projection(self.model.text_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output)
        self.text = torch.nn.functional.normalize(t, dim=-1)

    @torch.no_grad()
    def probs(self, img: Image.Image) -> torch.Tensor:
        px = self.proc(images=[img], return_tensors="pt")["pixel_values"]
        e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
        e = torch.nn.functional.normalize(e, dim=-1)
        return (100 * e @ self.text.T).softmax(-1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", type=int, default=200, help="candidates generated per emotion")
    ap.add_argument("--keep", type=int, default=60, help="max images kept per emotion")
    ap.add_argument("--min-prob", type=float, default=0.35, help="judge probability needed to keep")
    ap.add_argument("--probe", action="store_true", help="20 candidates per emotion, report hit rate")
    ap.add_argument("--out", default=str(ROOT / "data" / "expressions"))
    args = ap.parse_args()
    if args.probe:
        args.per_emotion = 20

    rng = random.Random(0)
    backend, judge = LocalTurboBackend(), Judge()
    out = Path(args.out)
    rows, stats = [], {}
    for emotion in LABELS:
        scored = []
        for _ in range(args.per_emotion):
            look, shot, place = rng.choice(LOOKS), rng.choice(SHOTS), rng.choice(PLACES)
            text = caption(emotion, look, shot, place)
            data = backend.generate(ImageRequest(text.replace(f"{TRIGGER} ", ""), rng.randrange(2**31)))
            img = Image.open(io.BytesIO(data)).convert("RGB")
            p = judge.probs(img)
            scored.append((float(p[LABELS.index(emotion)]), LABELS[int(p.argmax())], img, text))

        hits = [s for s in scored if s[1] == emotion and s[0] >= args.min_prob]
        stats[emotion] = {"generated": len(scored), "passed": len(hits),
                          "hit_rate": round(len(hits) / len(scored), 3),
                          "judge_saw": dict(Counter(s[1] for s in scored).most_common(3))}
        print(f"{emotion:9} {len(hits):3}/{len(scored)} passed  judge saw {stats[emotion]['judge_saw']}")
        if args.probe:
            continue
        hits.sort(key=lambda s: -s[0])
        (out / emotion).mkdir(parents=True, exist_ok=True)
        for i, (prob, _, img, text) in enumerate(hits[:args.keep]):
            name = f"{emotion}/{i:03d}.png"
            img.save(out / name)
            rows.append({"file_name": name, "text": text, "emotion": emotion, "judge_prob": round(prob, 3)})

    if not args.probe:
        with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"kept {len(rows)} images in {out}")


if __name__ == "__main__":
    main()
