"""Build a curated LoRA training set of comic faces that clearly show each emotion.

Self-training with a human-style curator (generate -> curate -> fine-tune):

    python training/build_expression_dataset.py generate --per-emotion 48
        SD-Turbo draws candidates per emotion with varied characters, shots and places,
        using the same prompt format as the app. Writes data/expressions/candidates/ and
        contact sheets data/expressions/sheets/<emotion>_<n>.png for review.

    python training/build_expression_dataset.py finalize data/expressions/keep.json
        keep.json maps emotion -> list of candidate numbers that clearly show it.
        Copies the keepers to data/expressions/train/ and writes train/metadata.jsonl in
        the {"file_name", "text"} format train_lora.py reads.

Automatic judges were tried as the filter and rejected: CLIP ViT-B/16 and ViT-L/14 and a
Qwen VLM agreed with blind labels only 47-58% of the time (reports/judge_agreement.json).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    "chubby man with a mustache and a brown vest",
    "old woman with grey curly hair and pearl necklace",
]
SHOTS = ["close-up", "extreme close-up", "close-up portrait", "medium close-up", "medium shot"]
PLACES = ["bedroom at night", "coffee shop", "city street", "office", "kitchen", "bus",
          "living room", "park", "school hallway", "train station", "restaurant", "beach"]
OUT = ROOT / "data" / "expressions"


def caption(emotion: str, look: str, shot: str, place: str) -> str:
    """Same structure the app's build_prompt() produces, so the LoRA trains on what it will see."""
    from pinch.style import ART_STYLE, EXPRESSION, STORY_POSE
    return ", ".join([ART_STYLE, shot, look, EXPRESSION[emotion], STORY_POSE[emotion], place])


def generate(per_emotion: int, seed: int, emotions: list[str] | None = None, start: int = 0) -> None:
    from pinch.pipeline import load_env
    load_env()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import LocalTurboBackend

    rng = random.Random(seed)
    backend = LocalTurboBackend()
    backend._load()
    cand, sheets = OUT / "candidates", OUT / "sheets"
    cand.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    meta_file = OUT / "candidates.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    for emotion in emotions or LABELS:
        imgs = []
        for k in range(start, start + per_emotion):
            text = caption(emotion, rng.choice(LOOKS), rng.choice(SHOTS), rng.choice(PLACES))
            g = torch.Generator("cuda").manual_seed(rng.randrange(2**31))
            img = backend._pipe(text, num_inference_steps=4, guidance_scale=0.0, generator=g,
                                width=512, height=512).images[0]
            name = f"{emotion}_{k:03d}.png"
            img.save(cand / name)
            meta[name] = {"emotion": emotion, "text": text}
            imgs.append(img)
        for s in range(0, len(imgs), 24):  # 6x4 sheets with candidate numbers
            sheet = Image.new("RGB", (6 * 220, 4 * 245), "white")
            d = ImageDraw.Draw(sheet)
            for j, img in enumerate(imgs[s:s + 24]):
                x, y = (j % 6) * 220, (j // 6) * 245
                sheet.paste(img.resize((210, 210)), (x + 5, y + 30))
                d.text((x + 8, y + 2), f"{start + s + j}", font=font, fill="black")
            sheet.save(sheets / f"{emotion}_{(start + s) // 24}.png")
        print(f"generated {emotion}")
    meta_file.write_text(json.dumps(meta, indent=1))


def finalize(keep_file: Path) -> None:
    meta = json.loads((OUT / "candidates.json").read_text())
    keep = json.loads(Path(keep_file).read_text())
    train = OUT / "train"
    if train.exists():
        shutil.rmtree(train)
    train.mkdir(parents=True)
    rows = []
    for emotion, numbers in keep.items():
        for n in numbers:
            name = f"{emotion}_{n:03d}.png"
            shutil.copy(OUT / "candidates" / name, train / name)
            rows.append({"file_name": name, "text": meta[name]["text"], "emotion": emotion})
    with open(train / "metadata.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    counts = {e: len(v) for e, v in keep.items()}
    print(f"{len(rows)} training images: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--per-emotion", type=int, default=48)
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--emotions", nargs="*", help="only these emotions (default: all)")
    g.add_argument("--start", type=int, default=0, help="first candidate number, to add to an existing set")
    f = sub.add_parser("finalize")
    f.add_argument("keep_file")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.per_emotion, args.seed, args.emotions, args.start)
    else:
        finalize(Path(args.keep_file))


if __name__ == "__main__":
    main()
