"""Quick look at the anime backend: one panel per emotion, timed, on one contact sheet.

    python eval/anime_smoke.py [--steps 20] [--size 768]
"""
import argparse
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from pinch.imagegen import ImageRequest, LocalAnimeBackend  # noqa: E402
from pinch.style import ART_STYLE, DEFAULT_CHARACTERS, EXPRESSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default=str(ROOT / "eval" / "anime_smoke.png"))
    args = ap.parse_args()
    b = LocalAnimeBackend(steps=args.steps, size=args.size)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    sheet = Image.new("RGB", (4 * 260, 2 * 290), "white")
    d = ImageDraw.Draw(sheet)
    for i, e in enumerate(LABELS):
        prompt = f"{ART_STYLE}, medium shot, {DEFAULT_CHARACTERS['me']}, {EXPRESSION[e]}, cozy living room"
        t = time.perf_counter()
        img = Image.open(io.BytesIO(b.generate(ImageRequest(prompt, 1234 + i, emotion=e, shot="medium shot"))))
        print(f"{e:9} {time.perf_counter() - t:5.1f} s", flush=True)
        x, y = (i % 4) * 260, (i // 4) * 290
        sheet.paste(img.resize((250, 250)), (x + 5, y + 35))
        d.text((x + 8, y + 5), e, font=font, fill="black")
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
