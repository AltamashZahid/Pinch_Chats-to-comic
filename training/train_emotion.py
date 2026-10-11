"""Fine-tune DistilBERT to classify chat messages into 7 emotions.

    python training/train_emotion.py               # full run
    python training/train_emotion.py --quick       # tiny smoke test

Data: GoEmotions (58k Reddit comments, 27 emotions), collapsed to Ekman's 7 classes.
Comments whose labels map to more than one coarse class are dropped so each example
has one target. Rare classes (disgust, fear) are up-weighted in the loss.

Also trains a TF-IDF + logistic regression baseline so the fine-tuned model's
gain is measured, not assumed. Writes the model to models/emotion/ and metrics to
reports/emotion_metrics.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()  # before importing HF libraries, so HF_HOME from .env decides the download folder

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from pinch.emotions import FINE_TO_COARSE, LABELS  # noqa: E402

BASE_MODEL = "distilbert-base-uncased"


def load_split(split: str, names: list[str]) -> tuple[list[str], list[int]]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    texts, labels = [], []
    for row in ds:
        coarse = {FINE_TO_COARSE[names[i]] for i in row["labels"]}
        if len(coarse) == 1:
            texts.append(row["text"])
            labels.append(LABELS.index(coarse.pop()))
    return texts, labels


def baseline(train, test) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(train[0]), train[1])
    pred = clf.predict(vec.transform(test[0]))
    return {"macro_f1": f1_score(test[1], pred, average="macro"),
            "accuracy": float(np.mean(np.array(pred) == np.array(test[1])))}


def batches(tokenizer, texts, labels, batch_size, shuffle):
    pairs = list(zip(texts, labels))

    def collate(chunk):
        enc = tokenizer([t for t, _ in chunk], truncation=True, max_length=64,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in chunk])
        return enc

    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--quick", action="store_true", help="train on 2k examples for a smoke test")
    ap.add_argument("--out", default=str(ROOT / "models" / "emotion"))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    names = load_dataset("google-research-datasets/go_emotions", "simplified",
                         split="train").features["labels"].feature.names
    train, val, test = (load_split(s, names) for s in ("train", "validation", "test"))
    if args.quick:
        train, val, test = ((t[:n], y[:n]) for (t, y), n in zip((train, val, test), (2000, 500, 500)))
        args.epochs = 1
    print(f"examples: train {len(train[0])}, val {len(val[0])}, test {len(test[0])}")
    print("class counts:", dict(zip(LABELS, np.bincount(train[1], minlength=len(LABELS)).tolist())))

    print("training TF-IDF baseline...")
    base = baseline(train, test)
    print(f"baseline macro-F1 {base['macro_f1']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    # Inverse-sqrt-frequency class weights: helps rare classes without wrecking precision.
    counts = np.bincount(train[1], minlength=len(LABELS))
    weights = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train_dl = batches(tokenizer, *train, args.batch_size, shuffle=True)
    val_dl = batches(tokenizer, *val, 128, shuffle=False)
    test_dl = batches(tokenizer, *test, 128, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, start = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {running / 200:.4f}")
                running = 0.0

        val_f1 = f1_score(val[1], predict(model, val_dl, device), average="macro")
        print(f"epoch {epoch}: val macro-F1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate the best epoch on the held-out test set, then save it.
    model.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    pred = predict(model, test_dl, device)
    report = classification_report(test[1], pred, target_names=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "model": BASE_MODEL,
        "quick": args.quick,
        "epochs": args.epochs,
        "train_minutes": round((time.time() - start) / 60, 1),
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_accuracy": report["accuracy"],
        "baseline_tfidf_logreg": base,
        "per_class_f1": {l: report[l]["f1-score"] for l in LABELS},
    }
    print(classification_report(test[1], pred, target_names=LABELS, zero_division=0))
    print(f"test macro-F1 {metrics['test_macro_f1']:.3f} vs baseline {base['macro_f1']:.3f}")
    if not args.quick:
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / "emotion_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
