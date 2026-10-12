"""Generate a small image set for validating expression judges against human labels.

    python eval/make_judge_set.py

Writes eval/judge_set/<id>.png, eval/judge_set/items.json (the requested emotion and
settings per image, hidden from the labeller), and shuffled, unlabelled contact sheets
eval/judge_set/sheet_<n>.png so labelling is blind to the requested emotion.
"""
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from pinch.imagegen import LocalTurboBackend  # noqa: E402
from training.build_expression_dataset import LOOKS, PLACES, SHOTS, caption  # noqa: E402

OUT = ROOT / "eval" / "judge_set"
PER_EMOTION = 10
NEG = "calm, neutral expression, expressionless, blank stare, relaxed face"
SETTINGS = [{"cfg": 0.0, "steps": 4, "neg": None}, {"cfg": 4.0, "steps": 8, "neg": NEG}]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    backend = LocalTurboBackend()
    backend._load()
    items = []
    for emotion in LABELS:
        for k in range(PER_EMOTION):
            s = SETTINGS[k % 2]
            prompt = caption(emotion, rng.choice(LOOKS), rng.choice(SHOTS), rng.choice(PLACES))
            g = torch.Generator("cuda").manual_seed(rng.randrange(2**31))
            img = backend._pipe(prompt, negative_prompt=s["neg"], num_inference_steps=s["steps"],
                                guidance_scale=s["cfg"], generator=g, width=512, height=512).images[0]
            items.append({"emotion": emotion, "cfg": s["cfg"], "prompt": prompt, "img": img})
        print(f"generated {emotion}")

    rng.shuffle(items)  # blind: sheet order says nothing about the requested emotion
    meta = []
    for i, it in enumerate(items):
        it["img"].save(OUT / f"{i:02d}.png")
        meta.append({"id": f"{i:02d}", **{k: v for k, v in it.items() if k != "img"}})
    (OUT / "items.json").write_text(json.dumps(meta, indent=2))

    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
    for n in range(0, len(items), 10):
        sheet = Image.new("RGB", (5 * 300, 2 * 330), "white")
        d = ImageDraw.Draw(sheet)
        for j, it in enumerate(items[n:n + 10]):
            x, y = (j % 5) * 300, (j // 5) * 330
            sheet.paste(it["img"].resize((290, 290)), (x + 5, y + 35))
            d.text((x + 10, y + 2), f"#{n + j:02d}", font=font, fill="black")
        sheet.save(OUT / f"sheet_{n // 10}.png")
    print(f"{len(items)} images, sheets in {OUT}")


if __name__ == "__main__":
    main()
