"""Conversation-aware text emotion model, following EmoBERTa (Kim & Vossen, 2021).

    python training/train_emotion_erc.py --base roberta-base --out models/emotion_erc_base
    python training/train_emotion_erc.py --base roberta-large --out models/emotion_erc_large --adam8bit

Input per utterance (EmoBERTa, arXiv 2108.12009):
    <s> past turns </s></s> Speaker: current turn </s></s> future turns </s>
with every turn prefixed "Speaker: ". Context grows outward from the current turn (one past,
one future, ...) until the token budget is used. Their MELD ablation: no context 63.5, past
64.6, past+future 65.6 weighted F1 (roberta-large).

Training data: English conversation sets (MELD, DailyDialog) plus single-post sets
(GoEmotions, tweets) as one-turn "dialogues". Scored per source with the metrics papers
report: MELD weighted F1, DailyDialog micro-F1 without neutral, plus macro-F1 and per-emotion
F1 everywhere. Writes reports/<name>_eval.json.
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
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from pinch.emotions import LABELS  # noqa: E402
from training.text_datasets import dialogues  # noqa: E402

TRAIN_SOURCES = ["meld", "dailydialog", "goemotions", "dairai"]
EVAL_SOURCES = ["meld", "dailydialog", "goemotions", "dairai"]
NEUTRAL = LABELS.index("neutral")


def turn_text(speaker: str, text: str) -> str:
    return f"{speaker}: {text}" if speaker else text


def build_input(tok, dialogue, i: int, max_len: int, future: bool = True) -> list[int]:
    """Token ids for turn i with as much past (and future) context as fits (EmoBERTa Alg. 1)."""
    sep = [tok.sep_token_id]
    cur = tok.encode(turn_text(*dialogue[i][:2]), add_special_tokens=False)[: max_len - 8]
    past, fut = [], []
    p, f = i - 1, i + 1
    budget = max_len - 2 - len(cur) - 4  # <s> </s> and two double separators
    while budget > 0 and (p >= 0 or (future and f < len(dialogue))):
        grew = False
        if p >= 0:
            t = tok.encode(turn_text(*dialogue[p][:2]), add_special_tokens=False)
            if len(t) + 1 <= budget:
                past = t + [tok.sep_token_id] + past if past else t
                budget -= len(t) + 1
                grew = True
            p -= 1
        if future and f < len(dialogue):
            t = tok.encode(turn_text(*dialogue[f][:2]), add_special_tokens=False)
            if len(t) + 1 <= budget:
                fut = fut + [tok.sep_token_id] + t if fut else t
                budget -= len(t) + 1
                grew = True
            f += 1
        if not grew and (p < 0 or budget <= 0) and (not future or f >= len(dialogue) or budget <= 0):
            break
    ids = [tok.cls_token_id]
    if past:
        ids += past + sep + sep
    ids += cur
    if fut:
        ids += sep + sep + fut
    return ids + sep


def examples(tok, source: str, split: str, max_len: int, neutral_cap: int | None, rng) -> list[tuple[list[int], int]]:
    out = []
    for d in dialogues(source, split):
        for i, (_, _, label) in enumerate(d):
            out.append((build_input(tok, d, i, max_len), LABELS.index(label)))
    if neutral_cap is not None:
        neu = [e for e in out if e[1] == NEUTRAL]
        rng.shuffle(neu)
        out = [e for e in out if e[1] != NEUTRAL] + neu[:neutral_cap]
    return out


def collate(batch, pad_id):
    n = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (x, _) in enumerate(batch):
        ids[k, :len(x)], mask[k, :len(x)] = torch.tensor(x), 1
    return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([y for _, y in batch])}


@torch.no_grad()
def predict(model, data, pad_id, device, bs=32) -> np.ndarray:
    model.eval()
    order = sorted(range(len(data)), key=lambda k: len(data[k][0]))  # length-sorted batches: much faster
    logits = np.zeros((len(data), len(LABELS)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = collate([data[k] for k in idx], pad_id)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            z = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
        logits[idx] = z.float().cpu().numpy()
    return logits


def score(y, p) -> dict:
    present = sorted(set(y))
    nn = [c for c in present if c != NEUTRAL]
    return {"n": len(y), "accuracy": round(float(np.mean(np.array(y) == p)), 3),
            "weighted_f1": round(f1_score(y, p, average="weighted"), 3),
            "macro_f1": round(f1_score(y, p, labels=present, average="macro"), 3),
            "micro_f1_no_neutral": round(f1_score(y, p, labels=nn, average="micro"), 3),
            "per_class_f1": {LABELS[c]: round(v, 3) for c, v in zip(present, f1_score(y, p, labels=present, average=None))}}


def evaluate(model, tok, sets, device) -> dict:
    rep = {}
    for s, data in sets.items():
        z = predict(model, data, tok.pad_token_id, device)
        rep[s] = score([y for _, y in data], z.argmax(1))
    rep["mean_macro_f1"] = round(float(np.mean([rep[s]["macro_f1"] for s in sets])), 3)
    return rep


def show(name, rep):
    print(f"\n{name}: mean macro-F1 {rep['mean_macro_f1']}")
    print(f"  {'source':12} {'acc':>5} {'wF1':>5} {'macro':>5} {'microNN':>7} | " + " ".join(f"{l[:5]:>5}" for l in LABELS))
    for s in EVAL_SOURCES:
        r = rep[s]
        pc = r["per_class_f1"]
        print(f"  {s:12} {r['accuracy']:5.2f} {r['weighted_f1']:5.2f} {r['macro_f1']:5.2f} {r['micro_f1_no_neutral']:7.2f} | "
              + " ".join(f"{pc[l]:5.2f}" if l in pc else "    -" for l in LABELS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--out", default=str(ROOT / "models" / "emotion_erc_base"))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neutral-cap", type=int, default=8000, help="cap DailyDialog neutral turns (83%% of it)")
    ap.add_argument("--adam8bit", action="store_true", help="8-bit AdamW (bitsandbytes) to fit roberta-large in 4 GB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.base)
    t0 = time.time()
    train = []
    for s in TRAIN_SOURCES:
        ex = examples(tok, s, "train", args.max_len, args.neutral_cap if s == "dailydialog" else None, rng)
        print(f"  train {s:12} {len(ex)}", flush=True)
        train += ex
    val = {s: examples(tok, s, "validation", args.max_len, None, rng) for s in EVAL_SOURCES}
    test = {s: examples(tok, s, "test", args.max_len, None, rng) for s in EVAL_SOURCES}
    counts = np.bincount([y for _, y in train], minlength=len(LABELS))
    print(f"  built {len(train)} training inputs in {time.time() - t0:.0f}s; mean tokens "
          f"{np.mean([len(x) for x, _ in train]):.0f}; classes {dict(zip(LABELS, counts.tolist()))}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    model.config.pinch_erc = {"max_len": args.max_len, "future": True}
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    # Length-bucketed shuffling: batches of similar length waste less padding.
    def batches():
        idx = list(range(len(train)))
        rng.shuffle(idx)
        chunks = [sorted(idx[k:k + args.batch_size * 50], key=lambda j: len(train[j][0]))
                  for k in range(0, len(idx), args.batch_size * 50)]
        bs = [c[k:k + args.batch_size] for c in chunks for k in range(0, len(c), args.batch_size)]
        rng.shuffle(bs)
        return bs
    steps_per_epoch = (len(train) // args.batch_size) // args.accum
    sched = get_linear_schedule_with_warmup(opt, int(0.2 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler()

    best, best_state, start = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for k, idx in enumerate(batches(), 1):
            b = collate([train[j] for j in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            loss = loss_fn(z.float(), b["labels"].to(device)) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            n += 1
            if k % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if k % 1000 == 0:
                print(f"  epoch {epoch} batch {k} loss {running / n:.4f} "
                      f"({(time.time() - start) / 60:.0f} min, {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)", flush=True)
                running, n = 0.0, 0
        rep = evaluate(model, tok, val, device)
        print(f"epoch {epoch}: validation mean macro-F1 {rep['mean_macro_f1']} | meld wF1 {rep['meld']['weighted_f1']}", flush=True)
        if rep["mean_macro_f1"] > best:
            best = rep["mean_macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    rep = evaluate(model, tok, test, device)
    rep["train"] = {"base": args.base, "examples": len(train), "epochs": args.epochs, "max_len": args.max_len,
                    "minutes": round((time.time() - start) / 60, 1), "best_val_mean_macro_f1": best}
    name = Path(args.out).name
    show(name, rep)
    (ROOT / "reports" / f"{name}_eval.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
