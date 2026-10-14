"""Load every text-emotion dataset into one shape with our 7 labels.

Sources (downloaded to data/text/, see README):
- goemotions   Reddit comments, 27 labels -> 7 (pinch/emotions.py). Apache-2.0.
- dailydialog  everyday two-person chat, 7 labels (no emotion -> neutral). CC BY-NC-SA 4.0.
- meld         Friends TV dialogue, 7 labels. GPL-3.0.
- hinglish     SemEval-2024 Task 10 (MaSaC, Sarabhai vs Sarabhai), romanised Hindi-English
               dialogue, 8 labels; contempt -> disgust.
- dairai       16k tweets, 6 labels (love -> joy; no disgust/neutral).
- user         your own labelled chat lines (data/text/user/labels.jsonl, from the labelling page).

Each source keeps its own test split so models are compared per domain.

load_ctx() also returns the previous line of the same conversation, the context a reader
uses ("Same" after "Missing u" is sad, not neutral). Single posts (goemotions, dairai) get "".
"""
import csv
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "text"

DD = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
MELD = {"neutral": "neutral", "anger": "anger", "disgust": "disgust", "fear": "fear",
        "joy": "joy", "sadness": "sadness", "surprise": "surprise"}
HINGLISH = dict(MELD, contempt="disgust")
DAIRAI = {0: "sadness", 1: "joy", 2: "joy", 3: "anger", 4: "fear", 5: "surprise"}  # 2 = love

Row = tuple[str, str, str]  # (previous line, text, label)


def goemotions(split: str) -> list[Row]:
    from datasets import load_dataset
    from pinch.emotions import FINE_TO_COARSE
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    names = ds.features["labels"].feature.names
    out = []
    for row in ds:
        coarse = {FINE_TO_COARSE[names[i]] for i in row["labels"]}
        if len(coarse) == 1:
            out.append(("", row["text"], coarse.pop()))
    return out


def dailydialog(split: str) -> list[Row]:
    df = pd.read_parquet(TEXT / "dailydialog" / f"{split}.parquet")
    out = []
    for us, es in zip(df["utterances"], df["emotions"]):
        prev = ""
        for u, e in zip(us, es):
            if u.strip():
                out.append((prev, u.strip(), DD[int(e)]))
                prev = u.strip()
    return out


def meld(split: str) -> list[Row]:
    name = {"train": "train", "validation": "dev", "test": "test"}[split]
    with open(TEXT / "meld" / f"{name}_sent_emo.csv", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
    out, prev, dialogue = [], "", None
    for r in rows:
        if r["Dialogue_ID"] != dialogue:
            prev, dialogue = "", r["Dialogue_ID"]
        u = r["Utterance"].replace(chr(0x92), "'").strip()
        if u:
            out.append((prev, u, MELD[r["Emotion"]]))
            prev = u
    return out


def hinglish(split: str) -> list[Row]:
    """Train file = train; the official validation file is our test (the repo's test file may
    hold model predictions, not gold labels). Every 10th train dialogue is held out as
    validation, so no conversation is split across the two."""
    if split == "test":
        df = pd.read_csv(TEXT / "semeval_hinglish" / "val.csv")
    else:
        df = pd.read_csv(TEXT / "semeval_hinglish" / "train.csv")
        val_ids = set(sorted(df["id"].unique())[::10])
        df = df[df["id"].isin(val_ids)] if split == "validation" else df[~df["id"].isin(val_ids)]
    out, prev, dialogue = [], "", None
    for d, u, e in zip(df["id"], df["utterance"], df["emotion"]):
        if d != dialogue:
            prev, dialogue = "", d
        if isinstance(u, str) and u.strip():
            if e in HINGLISH:
                out.append((prev, u.strip(), HINGLISH[e]))
            prev = u.strip()
    return out


def dairai(split: str) -> list[Row]:
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split", split=split)
    return [("", r["text"], DAIRAI[r["label"]]) for r in ds]


def user(split: str) -> list[Row]:
    """Lines labelled by the user. Split by conversation: 80% train, 10% validation, 10% test."""
    path = TEXT / "user" / "labels.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    convs = sorted({r["conversation"] for r in rows})
    random.Random(0).shuffle(convs)
    n = len(convs)
    parts = {"test": set(convs[:max(1, n // 10)]) if n >= 3 else set(),
             "validation": set(convs[max(1, n // 10):max(2, n // 5)]) if n >= 3 else set()}
    parts["train"] = set(convs) - parts["test"] - parts["validation"]
    return [(r.get("prev", ""), r["text"], r["label"]) for r in rows if r["conversation"] in parts[split]]


SOURCES = {"goemotions": goemotions, "dailydialog": dailydialog, "meld": meld,
           "hinglish": hinglish, "dairai": dairai, "user": user}


def load_ctx(source: str, split: str) -> list[Row]:
    """[(previous line, text, label)]"""
    return SOURCES[source](split)


def load(source: str, split: str) -> list[tuple[str, str]]:
    """[(text, label)] without context."""
    return [(t, l) for _, t, l in load_ctx(source, split)]


if __name__ == "__main__":
    import sys
    from collections import Counter
    sys.path.insert(0, str(ROOT))
    from pinch.pipeline import load_env
    load_env()
    for s in SOURCES:
        for split in ("train", "validation", "test"):
            rows = load_ctx(s, split)
            c = Counter(l for _, _, l in rows)
            with_ctx = sum(1 for p, _, _ in rows if p)
            print(f"{s:12} {split:10} {len(rows):6} (context {with_ctx:6})  " + " ".join(
                f"{k}={c[k]}" for k in ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")))


# --- whole dialogues with speakers (for EmoBERTa-style context models) --------------------

Turn = tuple[str, str, str]  # (speaker, text, label)


def dialogues(source: str, split: str) -> list[list[Turn]]:
    """Conversations as lists of (speaker, text, label). Single-post sources are 1-turn dialogues."""
    if source == "dailydialog":
        df = pd.read_parquet(TEXT / "dailydialog" / f"{split}.parquet")
        return [[("A" if i % 2 == 0 else "B", u.strip(), DD[int(e)]) for i, (u, e) in enumerate(zip(us, es)) if u.strip()]
                for us, es in zip(df["utterances"], df["emotions"])]
    if source == "meld":
        name = {"train": "train", "validation": "dev", "test": "test"}[split]
        with open(TEXT / "meld" / f"{name}_sent_emo.csv", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
        out: dict[str, list] = {}
        for r in rows:
            u = r["Utterance"].replace(chr(0x92), "'").strip()
            if u:
                out.setdefault(r["Dialogue_ID"], []).append((r["Speaker"].strip().title(), u, MELD[r["Emotion"]]))
        return list(out.values())
    if source == "hinglish":
        if split == "test":
            df = pd.read_csv(TEXT / "semeval_hinglish" / "val.csv")
        else:
            df = pd.read_csv(TEXT / "semeval_hinglish" / "train.csv")
            val_ids = set(sorted(df["id"].unique())[::10])
            df = df[df["id"].isin(val_ids)] if split == "validation" else df[~df["id"].isin(val_ids)]
        out = {}
        for d, s, u, e in zip(df["id"], df["speaker"], df["utterance"], df["emotion"]):
            if isinstance(u, str) and u.strip() and e in HINGLISH:
                out.setdefault(d, []).append((str(s).strip().title(), u.strip(), HINGLISH[e]))
        return list(out.values())
    if source == "augment":  # LLM-written chats for rare emotions: train only, label only the target turn
        if split != "train":
            return []
        out = []
        for path in sorted((TEXT / "augment").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                out.append([(s.title(), t, l if l == d["target"] else None) for s, t, l in d["turns"]])
        return out
    return [[("", t, l)] for _, t, l in load_ctx(source, split)]
