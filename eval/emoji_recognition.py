"""How well does the emoji recogniser handle an emoji art style it has no references for,
at the sizes emoji actually appear in screenshots?

    python eval/emoji_recognition.py

Leave-one-style-out over Segoe (Windows), Twemoji and OpenMoji: the gallery holds two
styles (each at full size and at 64/40/24 px, like the app), the third style is the query
set, shrunk to 64, 40 and 24 px to mimic big emoji, in-bubble emoji and tiny ones. The
held-out style stands in for Apple/Instagram/WhatsApp emoji, which the gallery never has.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eval.evaluate_panels import wilson  # noqa: E402
from pinch.emoji import EMOTION_OF, GALLERY_SIZES, EmojiRecognizer, at_size, reference_images  # noqa: E402

QUERY_SIZES = (64, 40, 24)


def main() -> None:
    rec = EmojiRecognizer(styles=())  # model only; galleries built per fold below
    styles = {}
    for s in ("segoe", "twemoji", "openmoji"):
        chars, imgs = reference_images((s,))
        styles[s] = dict(zip(chars, imgs))
    chars = [c for c in EMOTION_OF if all(c in v for v in styles.values())]
    emb = {(s, px): rec.embed([at_size(styles[s][c], px) for c in chars])
           for s in styles for px in set(GALLERY_SIZES) | set(QUERY_SIZES)}

    report, ok_all, n_all = {"chars": len(chars), "by_query_size": {}, "folds": {}}, 0, 0
    per_size = {px: [0, 0] for px in QUERY_SIZES}
    for q in styles:
        gal = [s for s in styles if s != q]
        G = np.concatenate([emb[(s, px)] for s in gal for px in GALLERY_SIZES])
        gc = [c for _ in gal for _ in GALLERY_SIZES for c in chars]
        fold = {}
        for px in QUERY_SIZES:
            pred = [gc[i] for i in (emb[(q, px)] @ G.T).argmax(1)]
            ok = sum(EMOTION_OF[p] == EMOTION_OF[c] for p, c in zip(pred, chars))
            fold[px] = round(ok / len(chars), 3)
            per_size[px][0] += ok
            per_size[px][1] += len(chars)
            ok_all, n_all = ok_all + ok, n_all + len(chars)
        report["folds"][q] = fold
    for px, (k, n) in per_size.items():
        report["by_query_size"][px] = {"emotion_acc": round(k / n, 3), "ci95": wilson(k, n)}
    report["emotion_acc"] = round(ok_all / n_all, 3)
    report["emotion_ci95"] = wilson(ok_all, n_all)
    print(json.dumps(report, indent=1))
    (ROOT / "reports" / "emoji_recognition.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
