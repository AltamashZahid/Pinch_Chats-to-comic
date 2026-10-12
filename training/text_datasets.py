"""Load every text-emotion dataset into one shape: (text, label, source) with our 7 labels.

Sources (downloaded to data/text/, see README):
- goemotions   Reddit comments, 27 labels -> 7 (pinch/emotions.py). Apache-2.0.
- dailydialog  everyday two-person chat, 7 labels (no emotion -> neutral). CC BY-NC-SA 4.0.
- meld         Friends TV dialogue, 7 labels. GPL-3.0.
- hinglish     SemEval-2024 Task 10 (MaSaC, Sarabhai vs Sarabhai), romanised Hindi-English
               dialogue, 8 labels; contempt -> disgust.
- dairai       416k tweets, 6 labels (love -> joy; no disgust/neutral). Training only.

Each source keeps its own official test split so models are compared per domain.
"""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "text"

DD = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
MELD = {"neutral": "neutral", "anger": "anger", "disgust": "disgust", "fear": "fear",
        "joy": "joy", "sadness": "sadness", "surprise": "surprise"}
HINGLISH = dict(MELD, contempt="disgust")
DAIRAI = {0: "sadness", 1: "joy", 2: "joy", 3: "anger", 4: "fear", 5: "surprise"}  # 2 = love


def goemotions(split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset
    from pinch.emotions import FINE_TO_COARSE
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    names = ds.features["labels"].feature.names
    out = []
    for row in ds:
        coarse = {FINE_TO_COARSE[names[i]] for i in row["labels"]}
        if len(coarse) == 1:
            out.append((row["text"], coarse.pop()))
    return out


def dailydialog(split: str) -> list[tuple[str, str]]:
    df = pd.read_parquet(TEXT / "dailydialog" / f"{split}.parquet")
    return [(u.strip(), DD[int(e)]) for us, es in zip(df["utterances"], df["emotions"])
            for u, e in zip(us, es) if u.strip()]


def meld(split: str) -> list[tuple[str, str]]:
    name = {"train": "train", "validation": "dev", "test": "test"}[split]
    rows = csv.DictReader(open(TEXT / "meld" / f"{name}_sent_emo.csv", encoding="utf-8", errors="replace"))
    return [(r["Utterance"].replace("\x92", "'").strip(), MELD[r["Emotion"]]) for r in rows if r["Utterance"].strip()]


def hinglish(split: str) -> list[tuple[str, str]]:
    """Train file = train; the official validation file is our test (the repo's test file may
    hold model predictions, not gold labels). A slice of train is held out as validation."""
    if split == "test":
        df = pd.read_csv(TEXT / "semeval_hinglish" / "val.csv")
    else:
        df = pd.read_csv(TEXT / "semeval_hinglish" / "train.csv")
        dialogues = sorted(df["id"].unique())
        val_ids = set(dialogues[::10])  # every 10th dialogue, so no dialogue is split
        df = df[df["id"].isin(val_ids)] if split == "validation" else df[~df["id"].isin(val_ids)]
    return [(str(u).strip(), HINGLISH[e]) for u, e in zip(df["utterance"], df["emotion"])
            if isinstance(u, str) and u.strip() and e in HINGLISH]


def dairai(split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split", split=split)
    return [(r["text"], DAIRAI[r["label"]]) for r in ds]


SOURCES = {"goemotions": goemotions, "dailydialog": dailydialog, "meld": meld,
           "hinglish": hinglish, "dairai": dairai}


def load(source: str, split: str) -> list[tuple[str, str]]:
    return SOURCES[source](split)


if __name__ == "__main__":
    import sys
    from collections import Counter
    sys.path.insert(0, str(ROOT))
    from pinch.pipeline import load_env
    load_env()
    for s in SOURCES:
        for split in ("train", "validation", "test"):
            rows = load(s, split)
            c = Counter(l for _, l in rows)
            print(f"{s:12} {split:10} {len(rows):6}  " + " ".join(f"{k}={c[k]}" for k in
                  ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")))
