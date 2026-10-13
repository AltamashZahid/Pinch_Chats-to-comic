"""Text emotion model v2: more data (chat, TV dialogue, Hinglish), evaluated per domain.

    python training/train_emotion_v2.py --eval-only models/emotion          # score an existing model
    python training/train_emotion_v2.py --eval-ensemble models/a models/b   # score an average of models
    python training/train_emotion_v2.py --base distilbert-base-uncased --out models/emotion_v2_distil
    python training/train_emotion_v2.py --base l3cube-pune/hing-roberta --freeze-embeddings \\
        --context --out models/emotion_v2_hing_ctx

Training mixes goemotions, dailydialog, meld, hinglish, dairai and the user's own labels
(training/text_datasets.py). DailyDialog is 83% "no emotion", so neutral examples are capped
per source. Hinglish and the user's labels are small but closest to real chats, so they're
repeated (--hinglish-repeat, --user-repeat).

--context: the model reads "current message </s> previous line" (if too long, the longer
of the two is trimmed), trained and evaluated that way; the flag is saved in the model's config as
`pinch_context` so inference knows to pass the previous message.

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
from training.text_datasets import SOURCES, load_ctx  # noqa: E402

MAX_LEN = 64
MAX_LEN_CTX = 96


def test_sources() -> list[str]:
    base = ["goemotions", "dailydialog", "meld", "hinglish", "dairai"]
    return base + (["user"] if load_ctx("user", "test") else [])


def build_train(neutral_cap: int, hinglish_repeat: int, user_repeat: int, seed: int) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    rows = []
    for s in SOURCES:
        data = load_ctx(s, "train")
        neutral = [r for r in data if r[2] == "neutral"]
        other = [r for r in data if r[2] != "neutral"]
        rng.shuffle(neutral)
        data = other + neutral[:neutral_cap]
        data *= {"hinglish": hinglish_repeat, "user": user_repeat}.get(s, 1)
        rows += [(p, t, LABELS.index(l)) for p, t, l in data]
        print(f"  train {s:12} {len(data)}")
    rng.shuffle(rows)
    return rows


def encode(tok, texts, prevs, context: bool):
    if context:
        return tok(texts, [p or "" for p in prevs], truncation="longest_first", max_length=MAX_LEN_CTX,
                   padding=True, return_tensors="pt")
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


def batches(tok, rows, bs, shuffle, context: bool):
    def collate(chunk):
        enc = encode(tok, [t for _, t, _ in chunk], [p for p, _, _ in chunk], context)
        enc["labels"] = torch.tensor([y for _, _, y in chunk])
        return enc
    return DataLoader(rows, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def probs(model, tok, texts, prevs, device, context: bool) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), 128):
        enc = encode(tok, texts[i:i + 128], prevs[i:i + 128], context).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**enc).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def score(y: list[int], p: np.ndarray) -> dict:
    present = sorted(set(y))
    r = classification_report(y, p, labels=present, target_names=[LABELS[i] for i in present],
                              output_dict=True, zero_division=0)
    return {"n": len(y), "macro_f1": round(r["macro avg"]["f1-score"], 3),
            "per_class_f1": {LABELS[i]: round(r[LABELS[i]]["f1-score"], 3) for i in present}}


def evaluate(members: list[tuple], device, split: str = "test") -> dict:
    """members: [(model, tokenizer, uses_context)]; probabilities are averaged."""
    report = {}
    srcs = test_sources()
    for s in srcs:
        data = load_ctx(s, split)
        if not data:
            continue
        prevs, texts, y = [p for p, _, _ in data], [t for _, t, _ in data], [LABELS.index(l) for _, _, l in data]
        avg = sum(probs(m, t, texts, prevs, device, c) for m, t, c in members) / len(members)
        report[s] = score(y, avg.argmax(1))
    scored = [s for s in srcs if s in report]
    report["mean_macro_f1"] = round(float(np.mean([report[s]["macro_f1"] for s in scored])), 3)
    return report


def show(name: str, rep: dict) -> None:
    print(f"\n{name}: mean macro-F1 over sources {rep['mean_macro_f1']}")
    print(f"  {'source':12} {'macroF1':>7}  " + " ".join(f"{l[:7]:>7}" for l in LABELS))
    for s in test_sources():
        if s in rep:
            pc = rep[s]["per_class_f1"]
            print(f"  {s:12} {rep[s]['macro_f1']:7.3f}  " +
                  " ".join(f"{pc[l]:7.3f}" if l in pc else f"{'-':>7}" for l in LABELS))


def load_member(path: str, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    return model, tok, bool(getattr(model.config, "pinch_context", False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", help="path of an existing model to score")
    ap.add_argument("--eval-ensemble", nargs="+", help="paths of models whose probabilities are averaged")
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "models" / "emotion_v2"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--neutral-cap", type=int, default=6000)
    ap.add_argument("--hinglish-repeat", type=int, default=3)
    ap.add_argument("--user-repeat", type=int, default=10)
    ap.add_argument("--context", action="store_true", help="also read the previous line of the conversation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="don't train the word-embedding table (XLM-R style models have a 250k vocab; "
                         "its AdamW state alone overflows a 4 GB GPU)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only or args.eval_ensemble:
        paths = [args.eval_only] if args.eval_only else args.eval_ensemble
        rep = evaluate([load_member(p, device) for p in paths], device)
        name = Path(paths[0]).name if len(paths) == 1 else "ensemble_" + "+".join(Path(p).name for p in paths)
        show(name, rep)
        (ROOT / "reports" / f"{name}_eval.json").write_text(json.dumps(rep, indent=2))
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("building training set:")
    train = build_train(args.neutral_cap, args.hinglish_repeat, args.user_repeat, args.seed)
    counts = np.bincount([y for _, _, y in train], minlength=len(LABELS))
    print("  total", len(train), dict(zip(LABELS, counts.tolist())))

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, ignore_mismatched_sizes=True).to(device)
    model.config.pinch_context = args.context
    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable params {sum(p.numel() for p in trainable) / 1e6:.0f}M "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M, context={args.context}")
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    dl = batches(tok, train, args.batch_size, True, args.context)
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
        val = evaluate([(model, tok, args.context)], device, "validation")["mean_macro_f1"]
        print(f"epoch {epoch}: validation mean macro-F1 {val:.3f}", flush=True)
        if val > best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    rep = evaluate([(model, tok, args.context)], device)
    rep["train"] = {"base": args.base, "examples": len(train), "epochs": args.epochs, "context": args.context,
                    "minutes": round((time.time() - start) / 60, 1), "best_val_mean_macro_f1": best}
    name = Path(args.out).name
    show(name, rep)
    (ROOT / "reports" / f"{name}_eval.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
