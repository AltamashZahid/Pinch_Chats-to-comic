"""Does rewriting chat messages before classifying them improve tone detection?

    python eval/rewrite_eval.py --source dailydialog --dialogues 60

Idea (from the project owner): normalise every message first, then classify the clean
version. Each test dialogue is rewritten by the LLM in one call, line by line, in two styles:
- clean:  a clear, grammatical, standalone English sentence with the same meaning and tone
          (short replies like "same" are expanded using the conversation)
- tweet:  the same, written as a first-person tweet (the style of the dair-ai tweet set,
          which our classifier scores 0.90 on)
The classifier labels raw, clean and tweet versions of the same lines; all three are scored
against the human labels. Rewrites are cached in data/text/rewrite_cache/.
Writes reports/rewrite_<source>.json.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import httpx  # noqa: E402

from eval.llm_tone_eval import dialogues, report  # noqa: E402

CACHE = ROOT / "data" / "text" / "rewrite_cache"
MODEL = os.environ.get("PINCH_REWRITE_MODEL", "openai/gpt-oss-20b")

STYLES = {
    "clean": ("Rewrite every numbered chat line as one clear, grammatical, standalone English sentence "
              "that keeps EXACTLY the same meaning and emotional tone. Expand short or slang replies "
              "(\"same\", \"lol ok\", \"wtf\") using the conversation. Do not add emotions that are not there."),
    "tweet": ("Rewrite every numbered chat line as a short first-person tweet by the person who said it, "
              "keeping EXACTLY the same meaning and emotional tone. Expand short or slang replies using the "
              "conversation. Do not add emotions that are not there."),
}


def rewrite(dialogue, style: str) -> list[str] | None:
    lines = "\n".join(f"{i}. {s}: {t}" for i, (s, t, _) in enumerate(dialogue, 1))
    key = hashlib.sha256((MODEL + STYLES[style] + lines).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": STYLES[style] +
                          ' Reply with JSON only: {"lines": ["<line 1>", ...]}, one entry per input line.'},
                         {"role": "user", "content": f"{lines}\n\nReturn exactly {len(dialogue)} lines."}]}
    for attempt in range(6):
        try:
            r = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=120,
                           headers={"Authorization": f"Bearer {os.environ['PINCH_LLM_API_KEY']}"})
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 20)))
                continue
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            out = [str(x) for x in json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["lines"]]
            if len(out) != len(dialogue):
                raise ValueError(f"{len(out)} lines for {len(dialogue)}")
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
            return out
        except (httpx.HTTPError, ValueError, KeyError, AttributeError) as e:
            print(f"    retry ({str(e)[:80]})")
            time.sleep(3)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dailydialog", choices=["dailydialog", "meld"])
    ap.add_argument("--dialogues", type=int, default=60)
    args = ap.parse_args()
    ds = dialogues(args.source)
    random.Random(1).shuffle(ds)
    ds = [d for d in ds if 3 <= len(d) <= 25][:args.dialogues]

    from pinch.classifier import EmotionClassifier
    clf = EmotionClassifier()
    y, preds, examples = [], {"raw": [], "clean": [], "tweet": []}, []
    for i, d in enumerate(ds, 1):
        got = {s: rewrite(d, s) for s in STYLES}
        if any(v is None for v in got.values()):
            print(f"  dialogue {i}: skipped")
            continue
        raw = [t for _, t, _ in d]
        versions = {"raw": raw, **got}
        for name, texts in versions.items():
            prevs = [""] + texts[:-1]
            preds[name] += [lab for lab, _ in clf.predict(texts, prevs)]
        y += [l for _, _, l in d]
        if len(examples) < 8:
            examples += [{"raw": a, "clean": b, "tweet": c, "label": l}
                         for a, b, c, (_, _, l) in zip(raw, got["clean"], got["tweet"], d)][:2]
        print(f"  dialogue {i}/{len(ds)}", flush=True)
    rep = {"source": args.source, "lines": len(y), "rewrite_model": MODEL,
           **{name: report(y, p) for name, p in preds.items()}, "examples": examples}
    for name in preds:
        print(f"{name:6} macro-F1 {rep[name]['macro_f1']:.3f}  accuracy {rep[name]['accuracy']:.3f}  "
              + " ".join(f"{k[:4]} {v:.2f}" for k, v in rep[name]["per_class_f1"].items()))
    (ROOT / "reports" / f"rewrite_{args.source}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
