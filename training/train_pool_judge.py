"""Expression judge trained on Pinch's own panels (data/expr_pool, blind-labelled), and a
measured simulation of best-of-N selection.

    python training/train_pool_judge.py

1. CLIP ViT-L/14 image embeddings of the 700 pool images; "unclear" labels dropped.
2. Logistic-regression head, scored with 5-fold cross-validation (every image is predicted by a
   head that never saw it) against zero-shot CLIP, the judge used before.
3. Best-of-N simulation, using only cross-validated predictions: for each requested emotion,
   draw N of the images generated for it, keep the one the judge rates most likely to show that
   emotion, and check the human label. Repeated many times. This is what "generate N candidates
   per panel and keep the best" would do in the app, measured on real labels.
Saves the head fitted on all data to models/pool_judge/head.npz and writes
reports/pool_judge.json.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from eval.evaluate_panels import FACE, wilson  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from training.train_expression_judge import Encoder  # noqa: E402

POOL = ROOT / "data" / "expr_pool"
OUT = ROOT / "models" / "pool_judge"


def main() -> None:
    import argparse
    global POOL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default="pool_judge")
    args = ap.parse_args()
    POOL, OUT = Path(args.pool), Path(args.out)
    items = {it["id"]: it for it in json.loads((POOL / "items.json").read_text())}
    labels = json.loads((POOL / "labels.json").read_text())["labels"]
    ids = sorted(i for i in labels if labels[i] in LABELS)
    y = np.array([LABELS.index(labels[i]) for i in ids])
    requested = np.array([items[i]["emotion"] for i in ids])

    enc = Encoder()
    cache = POOL / "clip_l14.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = enc.images([Image.open(POOL / "images" / f"{i}.png").convert("RGB") for i in ids])
        np.save(cache, X)

    # Cross-validated probabilities: each image scored by a head trained without it.
    cv_prob = np.zeros((len(ids), len(LABELS)))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        head = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(X[train], y[train])
        cv_prob[np.ix_(test, head.classes_)] = head.predict_proba(X[test])
    zs = enc.texts([f"a comic drawing of a person with a {FACE[l]} face" for l in LABELS])
    zs_prob = np.exp(100 * X @ zs.T)
    zs_prob /= zs_prob.sum(1, keepdims=True)

    def acc(prob):
        k = int((prob.argmax(1) == y).sum())
        per = {LABELS[c]: round(float((prob.argmax(1)[y == c] == c).mean()), 3) for c in range(len(LABELS))}
        return {"accuracy": round(k / len(y), 3), "ci95": wilson(k, len(y)), "per_class_recall": per}

    report = {"images": len(ids), "trained_judge_cv": acc(cv_prob), "zero_shot_clip": acc(zs_prob),
              "best_of_n": {}}
    print(f"judge (5-fold CV) accuracy {report['trained_judge_cv']['accuracy']}  "
          f"vs zero-shot {report['zero_shot_clip']['accuracy']}")

    # Best-of-N, by requested emotion.
    rng = random.Random(0)
    for n in (1, 2, 3, 4, 6):
        per = {}
        for e in LABELS:
            idx = np.where(requested == e)[0]
            c = LABELS.index(e)
            hits = 0
            for _ in range(2000):
                pick = rng.sample(list(idx), n)
                best = max(pick, key=lambda j: cv_prob[j, c])
                hits += y[best] == c
            per[e] = round(hits / 2000, 3)
        report["best_of_n"][n] = {"mean": round(float(np.mean(list(per.values()))), 3), "per_emotion": per}
        print(f"best of {n}: mean {report['best_of_n'][n]['mean']:.3f}  " +
              "  ".join(f"{e[:4]} {v:.2f}" for e, v in per.items()))

    OUT.mkdir(parents=True, exist_ok=True)
    head = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(X, y)
    np.savez(OUT / "head.npz", coef=head.coef_, intercept=head.intercept_,
             classes=np.array([LABELS[c] for c in head.classes_]))
    (ROOT / "reports" / f"{args.report}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
