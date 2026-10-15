"""Draw the panel library (pinch/library.py) with the anime backend. Resumable: existing
images are skipped, so it can be stopped and restarted any time.

    python training/build_library.py                 # every preset, 1 variant
    python training/build_library.py --variants 3    # add variants v1, v2
    python training/build_library.py --presets m1 f1 --sheets-only

Review: each preset gets a contact sheet (data/library/sheets/<preset>.png). Put the file
names of bad images in data/library/rejects.json; the next run redraws them with a new seed
and keeps the rejected file as *.rejected.png.

Full set: 12 presets x 7 emotions x 3 framings x 2 settings = 504 images per variant,
about 4 hours at ~28 s/image on an RTX 2050.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

from pinch.emotions import LABELS  # noqa: E402
from pinch.library import (FRAMING_SHOT, LIBRARY_DIR, PRESETS, SETTINGS, image_path,  # noqa: E402
                           library_prompt, rejects)

SIZE = 512  # what the app shows; drawn at 768 and downscaled by the backend


def seed_for(path: Path, attempt: int) -> int:
    return int(hashlib.sha256(f"{path.name}|{path.parent.name}|{attempt}".encode()).hexdigest()[:8], 16)


def jobs(presets: list[str], variants: int):
    for pid in presets:
        for e in LABELS:
            for framing in FRAMING_SHOT:
                for setting in SETTINGS:
                    for v in range(variants):
                        yield pid, e, framing, setting, v


def generate_with_retry(backend, req, attempts: int = 3) -> bytes:
    """A first run died with "CUDA error: out of memory" after 390 images (4 GB GPU, hours of
    allocations). Free the cache and retry instead of losing the night."""
    import torch
    for k in range(attempts):
        try:
            return backend.generate(req)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e) or k == attempts - 1:
                raise
            print(f"  out of GPU memory, retrying ({k + 1}/{attempts - 1})", flush=True)
            torch.cuda.empty_cache()
            time.sleep(30)


def contact_sheet(pid: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    files = sorted((LIBRARY_DIR / pid).glob("*_v*.png"))
    files = [f for f in files if not f.name.endswith(".rejected.png")]
    T, cols = 160, 6
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (T + 6) + 6, rows * (T + 22) + 6), "white")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        x, y = 6 + (i % cols) * (T + 6), 6 + (i // cols) * (T + 22)
        sheet.paste(Image.open(f).resize((T, T)), (x, y + 16))
        d.text((x, y), f.stem, font=font, fill="black")
    out = LIBRARY_DIR / "sheets" / f"{pid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="*", default=list(PRESETS))
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--sheets-only", action="store_true")
    args = ap.parse_args()

    if not args.sheets_only:
        from pinch.imagegen import ImageRequest, LocalAnimeBackend
        backend = LocalAnimeBackend(ip_scale=0)
        bad = rejects()
        todo = []
        for pid, e, framing, setting, v in jobs(args.presets, args.variants):
            path = image_path(pid, e, framing, setting, v)
            if (path.name in bad or f"{pid}/{path.name}" in bad) and path.exists():
                n = len(list(path.parent.glob(f"{path.stem}*.rejected.png")))
                path.rename(path.with_name(f"{path.stem}.{n}.rejected.png"))
            if not path.exists():
                todo.append((pid, e, framing, setting, v, path))
        if bad:  # consumed: the rejected files are now *.rejected.png and queued for redrawing,
            # so a redrawn image (same name) isn't rejected again on the next run
            (LIBRARY_DIR / "rejects.json").write_text("[]")
        print(f"{len(todo)} images to draw", flush=True)
        start = time.time()
        for k, (pid, e, framing, setting, v, path) in enumerate(todo, 1):
            attempt = len(list(path.parent.glob(f"{path.stem}*.rejected.png"))) if path.parent.exists() else 0
            req = ImageRequest(library_prompt(PRESETS[pid], e, framing, setting), seed_for(path, attempt),
                               SIZE, SIZE, emotion=e, shot=FRAMING_SHOT[framing])
            img = generate_with_retry(backend, req)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img)
            if k % 10 == 0 or k == len(todo):
                left = (time.time() - start) / k * (len(todo) - k) / 60
                print(f"  {k}/{len(todo)} ({path.parent.name}/{path.name}), ~{left:.0f} min left", flush=True)
    for pid in args.presets:
        if (LIBRARY_DIR / pid).exists():
            print("sheet:", contact_sheet(pid), flush=True)
    index = {pid: sorted(f.name for f in (LIBRARY_DIR / pid).glob("*_v*.png") if not f.name.endswith(".rejected.png"))
             for pid in PRESETS if (LIBRARY_DIR / pid).exists()}
    (LIBRARY_DIR / "index.json").write_text(json.dumps(index, indent=1))
    print(f"library: {sum(map(len, index.values()))} images", flush=True)


if __name__ == "__main__":
    main()
