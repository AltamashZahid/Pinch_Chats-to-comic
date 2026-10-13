"""Extract emotion-labelled manga faces: Manga109-s images + VisEmoComic emotion labels.

    python training/manga_faces.py

Inputs (data/, git-ignored; Manga109-s is licensed for research and must not be shared):
- data/manga109/Manga109s_released_2026_05_21/  pages + annotations/<book>.xml (face boxes)
- data/visemocomic/visemocomic_csvs/manga109_short_{train,test}.csv  3 annotators per face

Keeps faces where >= 2 of 3 annotators agree on one of our 7 emotions, crops a square around
each face (with margin), and writes data/manga_faces/{train,test}/<emotion>/<book>_<face>.png
plus data/manga_faces/{train,test}.csv. Train/test follow VisEmoComic's own split.
"""
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from training.train_expression_judge import majority  # noqa: E402

M109 = ROOT / "data" / "manga109" / "Manga109s_released_2026_05_21"
LABELS_CSV = ROOT / "data" / "visemocomic" / "visemocomic_csvs"
OUT = ROOT / "data" / "manga_faces"
MIN_SIDE = 48  # smaller faces are too low-resolution to teach expressions


def face_boxes(book: str) -> dict[str, tuple[int, int, int, int, int]]:
    """face id -> (page index, xmin, ymin, xmax, ymax)"""
    path = M109 / "annotations" / f"{book}.xml"
    if not path.exists():
        return {}
    out = {}
    for page in ET.parse(path).getroot().iter("page"):
        idx = int(page.get("index"))
        for f in page.iter("face"):
            out[f.get("id")] = (idx, int(f.get("xmin")), int(f.get("ymin")), int(f.get("xmax")), int(f.get("ymax")))
    return out


def crop(img: Image.Image, x0, y0, x1, y1, margin=0.3, size=256) -> Image.Image:
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return img.crop(tuple(int(v) for v in box)).convert("L").convert("RGB").resize((size, size), Image.LANCZOS)


def main() -> None:
    boxes_cache: dict[str, dict] = {}
    for split in ("train", "test"):
        rows_out, stats, skipped = [], Counter(), Counter()
        with open(LABELS_CSV / f"manga109_short_{split}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                label = majority(r)
                if label is None:
                    skipped["no agreement"] += 1
                    continue
                book = r["manga_name"]
                if book not in boxes_cache:
                    boxes_cache[book] = face_boxes(book)
                box = boxes_cache[book].get(r["face_id"])
                if box is None:
                    skipped["book/face not in Manga109-s"] += 1
                    continue
                page, x0, y0, x1, y1 = box
                if min(x1 - x0, y1 - y0) < MIN_SIDE:
                    skipped["face too small"] += 1
                    continue
                img_path = M109 / "images" / book / f"{page:03d}.jpg"
                if not img_path.exists():
                    skipped["page image missing"] += 1
                    continue
                name = f"{book}_{r['face_id']}.png"
                out = OUT / split / label / name
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    crop(Image.open(img_path), x0, y0, x1, y1).save(out)
                rows_out.append({"file": f"{split}/{label}/{name}", "label": label, "book": book})
                stats[label] += 1
        with open(OUT / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "book"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"{split}: {len(rows_out)} faces  " + " ".join(f"{l}={stats[l]}" for l in LABELS)
              + f"  | skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
