"""An in-domain expression dataset: Pinch-style panels, blind-labelled by the emotion they show.

    python training/expression_pool.py generate --per-emotion 100
    python training/expression_pool.py status        # how many images are labelled

Generates panels with the app's own image backend and prompt format (the same characters,
places, shots and emotion cues the comic builder uses), shuffles them, and writes numbered
contact sheets data/expr_pool/sheets/sheet_<n>.png. The requested emotion is stored in
data/expr_pool/items.json but not shown on the sheets, so labelling is blind.

Labels go in data/expr_pool/labels.json: {"labels": {"0000": "joy", ...}}. Use "unclear" for
faces that show no readable expression (no face visible, several people, broken image).
They become the training/test data for an expression judge (training/train_pool_judge.py)
and for LoRA fine-tuning.
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "expr_pool"
LOOKS = [
    "young man with short black hair and a blue hoodie", "young woman with curly red hair and a yellow sweater",
    "teenage boy with messy brown hair and a green t-shirt", "woman with a long black braid and a denim jacket",
    "older man with grey beard and a flannel shirt", "young woman with short blonde bob and a red coat",
    "man with dark skin, shaved head and a white shirt", "girl with glasses, ponytail and a striped sweater",
    "man with long hair in a bun and a leather jacket", "woman with hijab and a purple cardigan",
    "Indian college student with a mustache and a checked shirt", "young Indian woman in a kurta with a long braid",
    "chubby man with round glasses and a sweater vest", "old woman with grey hair bun and a shawl",
]
PLACES = ["cozy living room", "college canteen", "busy street", "office", "kitchen", "bedroom at night",
          "park bench", "classroom", "cafe", "train platform", "rooftop at sunset", "rainy bus stop"]
SHOTS = ["close-up", "extreme close-up", "medium shot", "medium close-up", "wide shot"]


def prompt_for(emotion: str, rng: random.Random) -> str:
    """Same structure as pinch.scenes.build_prompt in story mode."""
    from pinch.style import ART_STYLE, EXPRESSION, STORY_POSE
    return ", ".join([ART_STYLE, rng.choice(SHOTS), rng.choice(LOOKS), EXPRESSION[emotion],
                      STORY_POSE[emotion], rng.choice(PLACES)])


def generate(per_emotion: int, seed: int) -> None:
    from pinch.pipeline import load_env
    load_env()
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import ImageRequest, backend_from_env

    rng = random.Random(seed)
    backend = backend_from_env()
    jobs = [(e, prompt_for(e, rng), rng.randrange(2**31)) for e in LABELS for _ in range(per_emotion)]
    rng.shuffle(jobs)  # shuffled before numbering: the id says nothing about the emotion
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    items = []
    for i, (emotion, text, seed_i) in enumerate(jobs):
        pid = f"{i:04d}"
        img = Image.open(io.BytesIO(backend.generate(ImageRequest(text, seed_i, 512, 512)))).convert("RGB")
        img.save(OUT / "images" / f"{pid}.png")
        items.append({"id": pid, "emotion": emotion, "prompt": text, "seed": seed_i, "backend": backend.model})
        if (i + 1) % 50 == 0:
            print(f"generated {i + 1}/{len(jobs)}", flush=True)
    (OUT / "items.json").write_text(json.dumps(items, indent=1))

    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    per_sheet, cols, size = 30, 6, 210
    for n in range(0, len(items), per_sheet):
        chunk = items[n:n + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (size + 10), rows * (size + 34)), "white")
        d = ImageDraw.Draw(sheet)
        for j, it in enumerate(chunk):
            x, y = (j % cols) * (size + 10), (j // cols) * (size + 34)
            sheet.paste(Image.open(OUT / "images" / f"{it['id']}.png").resize((size, size)), (x + 5, y + 30))
            d.text((x + 8, y + 2), it["id"], font=font, fill="black")
        sheet.save(OUT / "sheets" / f"sheet_{n // per_sheet:02d}.png")
    print(f"{len(items)} images, {len(range(0, len(items), per_sheet))} sheets in {OUT / 'sheets'}")


def status() -> None:
    from collections import Counter
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())}
    lf = OUT / "labels.json"
    labels = json.loads(lf.read_text())["labels"] if lf.exists() else {}
    hit = Counter()
    tot = Counter()
    for i, lab in labels.items():
        tot[items[i]["emotion"]] += 1
        hit[items[i]["emotion"]] += lab == items[i]["emotion"]
    print(f"labelled {len(labels)}/{len(items)}; unclear {sum(1 for l in labels.values() if l == 'unclear')}")
    for e in sorted(tot):
        print(f"  requested {e:9} drawn correctly {hit[e]}/{tot[e]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--per-emotion", type=int, default=100)
    g.add_argument("--seed", type=int, default=2026)
    sub.add_parser("status")
    args = ap.parse_args()
    generate(args.per_emotion, args.seed) if args.cmd == "generate" else status()


if __name__ == "__main__":
    main()
