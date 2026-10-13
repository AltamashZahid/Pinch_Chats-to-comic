"""Does an LLM reading the whole conversation judge tone better than our classifier?

    python eval/llm_tone_eval.py --source hinglish --dialogues 46
    python eval/llm_tone_eval.py --source dailydialog --dialogues 80

Each test dialogue is sent whole to the configured LLM (Groq gpt-oss-120b by default), which
labels every line with one of our 7 emotions. The classifier labels the same lines. Both
are scored (macro-F1 and per-emotion F1) against the dataset's human labels on exactly the
same lines. LLM answers are cached in data/text/llm_cache/ so reruns don't re-query.
Writes reports/llm_tone_<source>.json.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from pinch.llm import LLMClient, LLMError  # noqa: E402
from training.text_datasets import DD, HINGLISH, MELD, TEXT  # noqa: E402

CACHE = ROOT / "data" / "text" / "llm_cache"

PROMPT = """You label the emotion of every line in a chat conversation. Read the WHOLE
conversation first: short replies like "same", "me 2" or "haan yaar" take their emotion
from context. Lines may be in Hinglish (romanised Hindi-English) or slang.
For each numbered line choose exactly one of: anger, disgust, fear, joy, sadness, surprise, neutral.
Use neutral only when the line really shows no emotion.
Reply with JSON only: {"labels": ["<label for line 1>", "<label for line 2>", ...]} with
exactly one label per line, in order."""


def dialogues(source: str) -> list[list[tuple[str, str, str]]]:
    """[[(speaker, text, label), ...] per dialogue] from the source's test split."""
    out = []
    if source == "hinglish":
        df = pd.read_csv(TEXT / "semeval_hinglish" / "val.csv")
        for _, g in df.groupby("id", sort=False):
            d = [(str(s), str(u).strip(), HINGLISH[e]) for s, u, e in zip(g["speaker"], g["utterance"], g["emotion"])
                 if isinstance(u, str) and u.strip() and e in HINGLISH]
            if d:
                out.append(d)
    elif source == "dailydialog":
        df = pd.read_parquet(TEXT / "dailydialog" / "test.parquet")
        for us, es in zip(df["utterances"], df["emotions"]):
            d = [("AB"[i % 2], u.strip(), DD[int(e)]) for i, (u, e) in enumerate(zip(us, es)) if u.strip()]
            if d:
                out.append(d)
    elif source == "meld":
        with open(TEXT / "meld" / "test_sent_emo.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by: dict[str, list] = {}
        for r in sorted(rows, key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"]))):
            by.setdefault(r["Dialogue_ID"], []).append((r["Speaker"], r["Utterance"].strip(), MELD[r["Emotion"]]))
        out = list(by.values())
    return out


def llm_labels(llm: LLMClient, dialogue: list[tuple[str, str, str]]) -> list[str] | None:
    lines = "\n".join(f"{i}. {s}: {t}" for i, (s, t, _) in enumerate(dialogue, 1))
    key = hashlib.sha256((llm.model + PROMPT + lines).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    msgs = [{"role": "system", "content": PROMPT},
            {"role": "user", "content": f"{lines}\n\nReturn exactly {len(dialogue)} labels."}]
    for attempt in range(6):
        try:
            raw = llm.chat(msgs, max_tokens=4000)
            m = re.search(r"\{.*\}", raw, re.S)
            labels = [str(l).lower().strip() for l in json.loads(m.group(0))["labels"]]
            if len(labels) != len(dialogue):
                raise ValueError(f"{len(labels)} labels for {len(dialogue)} lines")
            labels = [l if l in LABELS else "neutral" for l in labels]
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(labels))
            return labels
        except LLMError as e:
            wait = 20 if "429" in str(e) else 5
            print(f"    LLM error ({str(e)[:80]}), retrying in {wait}s")
            time.sleep(wait)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"    bad reply ({e}), retrying")
    return None


def report(y, p) -> dict:
    present = sorted({LABELS.index(l) for l in y})
    yi, pi = [LABELS.index(l) for l in y], [LABELS.index(l) for l in p]
    r = classification_report(yi, pi, labels=present, target_names=[LABELS[i] for i in present],
                              output_dict=True, zero_division=0)
    return {"macro_f1": round(r["macro avg"]["f1-score"], 3), "accuracy": round(r["accuracy"], 3),
            "per_class_f1": {LABELS[i]: round(r[LABELS[i]]["f1-score"], 3) for i in present}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hinglish", choices=["hinglish", "dailydialog", "meld"])
    ap.add_argument("--dialogues", type=int, default=46)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = LLMClient.from_env()
    if llm is None:
        raise SystemExit("needs PINCH_LLM_API_KEY")
    ds = dialogues(args.source)
    random.Random(args.seed).shuffle(ds)
    ds = ds[:args.dialogues]

    from pinch.classifier import EmotionClassifier
    clf = EmotionClassifier()
    y, p_llm, p_clf = [], [], []
    for i, d in enumerate(ds, 1):
        got = llm_labels(llm, d)
        if got is None:
            print(f"  dialogue {i}: skipped (LLM failed)")
            continue
        texts = [t for _, t, _ in d]
        prevs = [""] + texts[:-1]
        y += [l for _, _, l in d]
        p_llm += got
        p_clf += [lab for lab, _ in clf.predict(texts, prevs)]
        print(f"  dialogue {i}/{len(ds)}: {len(d)} lines", flush=True)

    rep = {"source": args.source, "dialogues": len(ds), "lines": len(y), "llm_model": llm.model,
           "classifier": str(Path(clf.tokenizer.name_or_path).name),
           "llm": report(y, p_llm), "classifier_scores": report(y, p_clf)}
    print(json.dumps(rep, indent=1))
    (ROOT / "reports" / f"llm_tone_{args.source}.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
