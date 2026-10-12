"""How well does the emoji recogniser handle an emoji art style it has no references for?

    python eval/emoji_recognition.py

Leave-one-style-out over Segoe (Windows), Twemoji and OpenMoji: the gallery holds two
styles, the third is the query set (standing in for Apple/Instagram/WhatsApp emoji, which
the gallery never contains). Reports exact-emoji and emotion accuracy per held-out style,
plus the similarity of correct matches, used to set MIN_SIMILARITY.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eval.evaluate_panels import wilson  # noqa: E402
from pinch.emoji import EMOTION_OF, GALLERY_DIR, EmojiRecognizer, _load_png, render_reference  # noqa: E402


def main() -> None:
    rec = EmojiRecognizer(styles=())  # model only; galleries built per fold below
    chars = list(EMOTION_OF)
    styles = {"segoe": [render_reference(c) for c in chars]}
    for name in ("twemoji", "openmoji"):
        styles[name] = [_load_png(GALLERY_DIR / name / f"{ord(c):x}.png") for c in chars]
    emb = {s: rec.embed(imgs) for s, imgs in styles.items()}

    report, total_ok, total_n, sims_ok = {"folds": {}}, 0, 0, []
    for q in styles:
        gal = [s for s in styles if s != q]
        G = np.concatenate([emb[s] for s in gal])
        gc = [c for _ in gal for c in chars]
        S = emb[q] @ G.T
        pred = [gc[i] for i in S.argmax(1)]
        ok = [EMOTION_OF[p] == EMOTION_OF[c] for p, c in zip(pred, chars)]
        sims_ok += [float(S[i].max()) for i, o in enumerate(ok) if o]
        total_ok, total_n = total_ok + sum(ok), total_n + len(ok)
        report["folds"][q] = {"exact": round(float(np.mean([p == c for p, c in zip(pred, chars)])), 3),
                              "emotion": round(sum(ok) / len(ok), 3)}
    report["emotion_acc"] = round(total_ok / total_n, 3)
    report["emotion_ci95"] = wilson(total_ok, total_n)
    report["similarity_of_correct_matches"] = {"p5": round(float(np.percentile(sims_ok, 5)), 3),
                                               "median": round(float(np.median(sims_ok)), 3)}
    print(json.dumps(report, indent=1))
    (ROOT / "reports" / "emoji_recognition.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
