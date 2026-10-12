"""Text emotion model v2: more data (chat, TV dialogue, Hinglish), evaluated per domain.

    python training/train_emotion_v2.py --eval-only models/emotion          # score an existing model
    python training/train_emotion_v2.py --base distilbert-base-uncased --out models/emotion_v2_distil
    python training/train_emotion_v2.py --base l3cube-pune/hing-roberta --out models/emotion_v2_hing

Training mixes goemotions, dailydialog, meld, hinglish and dairai (training/text_datasets.py).
DailyDialog is 83% "no emotion", so neutral examples are capped per source. Hinglish is the
smallest but matters most for real chats, so it's repeated (--hinglish-repeat).

Every model is scored on each source's own test split, per emotion, so a gain on one
domain can't hide a loss on another. Writes reports/<name>_eval.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from training.text_datasets import SOURCES, load  # noqa: E402

TEST_SOURCES = ["goemotions", "dailydialog", "meld", "hinglish", "dairai"]


def build_train(neutral_cap: int, hinglish_repeat: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    rows = []
    for s in SOURCES:
        data = load(s, "train")
        neutral = [r for r in data if r[1] == "neutral"]
        other = [r for r in data if r[1] != "neutral"]
        rng.shuffle(neutral)
        data = other + neutral[:neutral_cap]
        if s == "hinglish":
            data = data * hinglish_repeat
        rows += [(t, LABELS.index(l)) for t, l in data]
        print(f"  train {s:12} {len(data)}")
    rng.shuffle(rows)
    return rows


def batches(tok, rows, bs, shuffle):
    def collate(chunk):
        enc = tok([t for t, _ in chunk], truncation=True, max_length=64, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in chunk])
        return enc
    return DataLoader(rows, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, tok, texts, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), 128):
        enc = tok(texts[i:i + 128], truncation=True, max_length=64, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**enc).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, tok, device, split: str = "test") -> dict:
    report = {}
    for s in TEST_SOURCES:
        data = load(s, split)
        y = [LABELS.index(l) for _, l in data]
        present = sorted(set(y))
        p = predict(model, tok, [t for t, _ in data], device)
        r = classification_report(y, p, labels=present, target_names=[LABELS[i] for i in present],
                                  output_dict=True, zero_division=0)
        report[s] = {"n": len(y), "macro_f1": round(r["macro avg"]["f1-score"], 3),
                     "accuracy": round(r["accuracy"], 3) if "accuracy" in r else None,
                     "per_class_f1": {LABELS[i]: round(r[LABELS[i]]["f1-score"], 3) for i in present}}
    report["mean_macro_f1"] = round(float(np.mean([report[s]["macro_f1"] for s in TEST_SOURCES])), 3)
    return report


def show(name: str, rep: dict) -> None:
    print(f"\n{name}: mean macro-F1 over sources {rep['mean_macro_f1']}")
    print(f"  {'source':12} {'macroF1':>7}  " + " ".join(f"{l[:7]:>7}" for l in LABELS))
    for s in TEST_SOURCES:
        pc = rep[s]["per_class_f1"]
        print(f"  {s:12} {rep[s]['macro_f1']:7.3f}  " + " ".join(f"{pc[l]:7.3f}" if l in pc else f"{'-':>7}" for l in LABELS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", help="path of an existing model to score")
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "models" / "emotion_v2"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--neutral-cap", type=int, default=6000)
    ap.add_argument("--hinglish-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="don't train the word-embedding table (XLM-R style models have a 250k vocab; "
                         "its AdamW state alone overflows a 4 GB GPU)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(args.eval_only)
        model = AutoModelForSequenceClassification.from_pretrained(args.eval_only).to(device)
        rep = evaluate(model, tok, device)
        name = Path(args.eval_only).name
        show(name, rep)
        (ROOT / "reports" / f"{name}_eval.json").write_text(json.dumps(rep, indent=2))
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("building training set:")
    train = build_train(args.neutral_cap, args.hinglish_repeat, args.seed)
    counts = np.bincount([y for _, y in train], minlength=len(LABELS))
    print("  total", len(train), dict(zip(LABELS, counts.tolist())))

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, ignore_mismatched_sizes=True).to(device)
    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable params {sum(p.numel() for p in trainable) / 1e6:.0f}M "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    weights = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    dl = batches(tok, train, args.batch_size, True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, best_state, start = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step}/{len(dl)} loss {running / 500:.4f}", flush=True)
                running = 0.0
        val = evaluate(model, tok, device, "validation")["mean_macro_f1"]
        print(f"epoch {epoch}: validation mean macro-F1 {val:.3f}", flush=True)
        if val > best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    rep = evaluate(model, tok, device)
    rep["train"] = {"base": args.base, "examples": len(train), "epochs": args.epochs,
                    "minutes": round((time.time() - start) / 60, 1), "best_val_mean_macro_f1": best}
    name = Path(args.out).name
    show(name, rep)
    (ROOT / "reports" / f"{name}_eval.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
