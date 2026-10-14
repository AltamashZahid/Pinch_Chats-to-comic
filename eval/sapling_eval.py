"""Compare Sapling's Tone API with our classifier on the same held-out English lines.

    python eval/sapling_eval.py --per-emotion 100

Lines: MELD test set (TV dialogue), stratified: up to N per emotion so rare emotions
(fear, disgust) are actually measured. ~50 chars/line, so 700 lines fit Sapling's free
50,000 characters/day. Sapling's 28 tones are summed into our 7 emotions (mapping below,
same grouping idea as GoEmotions -> Ekman). Sapling sees the single line, as does our
current classifier. Answers are cached in data/text/sapling_cache.jsonl, so reruns are free.
Writes reports/sapling_vs_ours.json.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from eval.llm_tone_eval import report  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from training.text_datasets import load  # noqa: E402

TONE_TO_EMOTION = {
    "angry": "anger", "annoyed": "anger", "disapproving": "anger",
    "repulsed": "disgust",
    "fearful": "fear", "worried": "fear",
    "admiring": "joy", "amused": "joy", "approving": "joy", "excited": "joy", "grateful": "joy", "joyful": "joy",
    "loving": "joy", "optimistic": "joy", "relieved": "joy", "eager": "joy", "confident": "joy",
    "sad": "sadness", "mournful": "sadness", "disappointed": "sadness", "remorseful": "sadness",
    "embarrassed": "sadness",
    "surprised": "surprise", "confused": "surprise", "curious": "surprise", "aware": "surprise",
    "neutral": "neutral", "sympathetic": "neutral",
}
CACHE = ROOT / "data" / "text" / "sapling_cache.jsonl"


def sapling(text: str, key: str) -> list:
    for attempt in range(5):
        r = httpx.post("https://api.sapling.ai/api/v1/tone", json={"key": key, "text": text}, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()["overall"]
    raise RuntimeError("Sapling rate limit")


def to_emotion(overall: list) -> str:
    score = dict.fromkeys(LABELS, 0.0)
    for p, tone, *_ in overall:
        score[TONE_TO_EMOTION.get(tone, "neutral")] += float(p)
    return max(score, key=score.get)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", type=int, default=100)
    args = ap.parse_args()
    key = os.environ["SAPLING_API_KEY"]
    rows = load("meld", "test")
    rng = random.Random(0)
    rng.shuffle(rows)
    picked, count = [], dict.fromkeys(LABELS, 0)
    for t, l in rows:
        if count[l] < args.per_emotion and 3 <= len(t) <= 200:
            picked.append((t, l))
            count[l] += 1
    print("lines per emotion:", count, "| characters:", sum(len(t) for t, _ in picked))

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["text"]] = d["overall"]
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, (t, _) in enumerate(picked, 1):
            if t not in cache:
                cache[t] = sapling(t, key)
                f.write(json.dumps({"text": t, "overall": cache[t]}, ensure_ascii=False) + "\n")
                f.flush()
            if i % 100 == 0:
                print(f"  sapling {i}/{len(picked)}", flush=True)

    from pinch.classifier import EmotionClassifier
    clf = EmotionClassifier()
    texts, y = [t for t, _ in picked], [l for _, l in picked]
    ours = [lab for lab, _ in clf.predict(texts)]
    theirs = [to_emotion(cache[t]) for t in texts]
    rep = {"lines": len(y), "per_emotion_lines": count, "sapling": report(y, theirs), "ours": report(y, ours),
           "agreement": round(float(np.mean([a == b for a, b in zip(ours, theirs)])), 3)}
    for name in ("sapling", "ours"):
        r = rep[name]
        print(f"{name:8} accuracy {r['accuracy']:.2f}  macro-F1 {r['macro_f1']:.2f}  | " +
              " ".join(f"{k[:5]} {v:.2f}" for k, v in r["per_class_f1"].items()))
    print("they agree on", rep["agreement"])
    (ROOT / "reports" / "sapling_vs_ours.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
