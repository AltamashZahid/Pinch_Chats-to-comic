"""Tune per-emotion logit biases of the text classifier with the Grey Wolf Optimizer.

    python training/tune_text_biases.py

The classifier under-predicts rare emotions like fear in chat. Adding a small bias per
emotion to its logits shifts those decisions without retraining. GWO searches the 7 biases
(range -2..2, starting from all zeros) to maximise mean macro-F1 over the English
*validation* splits; the result is reported on the untouched *test* splits, per emotion.
Saves the biases into the model folder (pinch_biases.json), which the classifier applies.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from pinch.classifier import DEFAULT_MODEL_DIR  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from pinch.gwo import gwo  # noqa: E402
from training.text_datasets import load  # noqa: E402

SOURCES = ["goemotions", "dailydialog", "meld", "dairai"]  # English only


@torch.no_grad()
def logits(model, tok, texts) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=64, padding=True, return_tensors="pt")
        out.append(model(**enc).logits.float().numpy())
    return np.concatenate(out)


def split_data(model, tok, split, cache: Path):
    """Logits per source, cached one file per source so a crash only loses one source."""
    d = {}
    for s in SOURCES:
        f = cache / f"logits_{split}_{s}.npz"
        if not f.exists():
            rows = load(s, split)
            z = logits(model, tok, [t for t, _ in rows])
            np.savez(f, z=z, y=np.array([LABELS.index(l) for _, l in rows]))
            print(f"  {split} {s}: {len(rows)} lines", flush=True)
        n = np.load(f)
        d[s] = (n["z"], n["y"])
    return d


def scores(data, bias) -> dict:
    out = {}
    for s, (z, y) in data.items():
        p = (z + bias).argmax(1)
        present = sorted(set(y))
        out[s] = {"macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
                  "per_class": dict(zip([LABELS[i] for i in present],
                                        map(float, f1_score(y, p, labels=present, average=None, zero_division=0))))}
    out["mean"] = float(np.mean([out[s]["macro_f1"] for s in data]))
    return out


def main() -> None:
    model_dir = Path(DEFAULT_MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS
    cache = ROOT / "data" / "text"
    torch.set_num_threads(4)
    val = split_data(model, tok, "validation", cache)
    test = split_data(model, tok, "test", cache)

    f = lambda b: scores(val, b)["mean"]  # noqa: E731
    best, val_score, hist = gwo(f, [-2] * 7, [2] * 7, wolves=16, iters=60, seed=0, init=np.zeros(7),
                                log=lambda t, v: print(f"  iter {t:2d} val mean macro-F1 {v:.4f}", flush=True) if t % 10 == 0 else None)
    before, after = scores(test, np.zeros(7)), scores(test, best)
    report = {"biases": dict(zip(LABELS, map(float, best.round(3)))), "val_before": scores(val, np.zeros(7))["mean"],
              "val_after": val_score, "test_before": before, "test_after": after}
    print("biases:", report["biases"])
    print(f"test mean macro-F1 {before['mean']:.3f} -> {after['mean']:.3f}")
    for s in SOURCES:
        b, a = before[s]["per_class"], after[s]["per_class"]
        print(f"  {s:12} {before[s]['macro_f1']:.3f} -> {after[s]['macro_f1']:.3f}   fear "
              f"{b.get('fear', float('nan')):.2f} -> {a.get('fear', float('nan')):.2f}")
    (model_dir / "pinch_biases.json").write_text(json.dumps(report["biases"], indent=1))
    (ROOT / "reports" / "text_biases_gwo.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
