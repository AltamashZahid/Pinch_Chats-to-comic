"""Train an automatic judge of comic facial expressions on human-labelled comic faces.

    python training/train_expression_judge.py

Data: VisEmoComic annotations of EmoRecCom panels (public-domain Golden Age American
comics; data/visemocomic/, see README). Each face has 3 annotators; we keep faces where at
least 2 agree on one of our 7 emotions (2,187 train / 934 test).

Model: frozen CLIP ViT-L/14 image embeddings of the face crop + a logistic-regression head
(class-balanced). Compared on the same test faces against zero-shot CLIP, the judge the
project used before.

Then the real question, domain transfer: does it judge *our generated* SD-Turbo faces
well? It's scored against the blind labels already collected in eval/judge_set (70 images)
and eval/lora_ab (112 images), where zero-shot CLIP and a VLM agreed only 47-58%.

Writes models/expression_judge/head.npz and reports/expression_judge.json.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from eval.evaluate_panels import FACE, wilson  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402

DATA = ROOT / "data" / "visemocomic"
OUT = ROOT / "models" / "expression_judge"
CLIP = "openai/clip-vit-large-patch14"


def majority(row) -> str | None:
    votes = Counter(e.strip() for k in ("ann1_emotions", "ann2_emotions", "ann3_emotions")
                    for e in row[k].split(",") if e.strip())
    top = [(e, c) for e, c in votes.most_common() if e in LABELS]
    if top and top[0][1] >= 2 and (len(top) == 1 or top[1][1] < top[0][1]):
        return top[0][0]
    return None


def face_crop(img: Image.Image, box, margin: float = 0.25) -> Image.Image:
    x0, y0, x1, y1 = box
    m = margin * max(x1 - x0, y1 - y0)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 + m
    return img.crop((cx - half, cy - half, cx + half, cy + half))


def load_split(name: str) -> tuple[list[Image.Image], list[str]]:
    crops, labels = [], []
    for r in csv.DictReader(open(DATA / "visemocomic_csvs" / f"emoreccom_{name}.csv", encoding="utf-8")):
        lab = majority(r)
        if lab is None:
            continue
        img = Image.open(DATA / "emoreccom" / r["filename"]).convert("RGB")
        crops.append(face_crop(img, [float(r[f"face_{a}"]) for a in ("xmin", "ymin", "xmax", "ymax")]))
        labels.append(lab)
    return crops, labels


class Encoder:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(CLIP, torch_dtype=dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP)
        self.dtype = dtype

    @torch.no_grad()
    def images(self, imgs) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), 32):
            px = self.proc(images=imgs[i:i + 32], return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def texts(self, texts) -> np.ndarray:
        enc = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
        e = self.model.text_projection(self.model.text_model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def blind_sets() -> list[tuple[str, list[Path], list[str]]]:
    """Generated SD-Turbo images with Claude's blind labels (collected earlier)."""
    sets = []
    js = ROOT / "eval" / "judge_set"
    labels = json.loads((js / "labels_claude.json").read_text())["labels"]
    sets.append(("judge_set", [js / f"{i}.png" for i in sorted(labels)], [labels[i] for i in sorted(labels)]))
    ab = ROOT / "eval" / "lora_ab"
    labels = json.loads((ab / "labels.json").read_text())["labels"]
    sets.append(("lora_ab", [ab / f"{i}.png" for i in sorted(labels)], [labels[i] for i in sorted(labels)]))
    return [(n, p, l) for n, p, l in sets if all(x.exists() for x in p)]


def scores(y_true, y_pred) -> dict:
    k = sum(a == b for a, b in zip(y_true, y_pred))
    return {"accuracy": round(k / len(y_true), 3), "ci95": wilson(k, len(y_true)),
            "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 3)}


def main() -> None:
    enc = Encoder()
    tr_img, tr_y = load_split("train")
    te_img, te_y = load_split("test")
    print(f"faces: train {len(tr_y)} {dict(Counter(tr_y))}, test {len(te_y)}")
    Xtr, Xte = enc.images(tr_img), enc.images(te_img)

    head = LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced").fit(Xtr, tr_y)
    zs = enc.texts([f"a comic drawing of a person with a {FACE[l]} face" for l in LABELS])
    zero_shot = lambda X: [LABELS[i] for i in (X @ zs.T).argmax(1)]  # noqa: E731

    report = {"train_faces": len(tr_y), "test_faces": len(te_y),
              "comic_test": {"trained_judge": scores(te_y, list(head.predict(Xte))),
                             "zero_shot_clip": scores(te_y, zero_shot(Xte))},
              "generated_images_vs_blind_labels": {}}
    for name, paths, labels in blind_sets():
        X = enc.images([Image.open(p).convert("RGB") for p in paths])
        report["generated_images_vs_blind_labels"][name] = {
            "n": len(labels), "trained_judge": scores(labels, list(head.predict(X))),
            "zero_shot_clip": scores(labels, zero_shot(X))}

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", coef=head.coef_, intercept=head.intercept_, classes=np.array(head.classes_))
    (ROOT / "reports" / "expression_judge.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
