"""Collect the user's own emotion labels for chat messages (fed into text-model training).

Input: a WhatsApp "Export chat" .txt (fastest way to get hundreds of real messages), pasted
"Name: message" lines, or a screenshot (via OCR). Output: data/text/user/labels.jsonl, one
line per message: {"conversation", "speaker", "prev", "text", "label", "model_pred"}.
Each conversation is saved as a unit, so training/text_datasets.py can split train/test by
conversation and never test on a chat it trained on.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .emotions import LABELS

STORE = Path(__file__).resolve().parent.parent / "data" / "text" / "user" / "labels.jsonl"

# WhatsApp exports: "12/10/26, 9:41 pm - Name: text" (Android) or "[12/10/26, 9:41:05 PM] Name: text" (iOS).
_WA = re.compile(r"^\[?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP]\.?[mM]\.?)?)\]?\s*(?:-\s*)?([^:]{1,40}):\s(.*)$")
_PLAIN = re.compile(r"^([^:]{1,30}):\s+(.+)$")
_SKIP = ("<Media omitted>", "This message was deleted", "Messages and calls are end-to-end encrypted",
         "image omitted", "video omitted", "sticker omitted", "<attached:", "null")


def parse_chat_text(raw: str) -> list[dict]:
    """[{"speaker", "text"}] from a WhatsApp export or "Name: message" lines. Multi-line
    WhatsApp messages are joined; system lines and media placeholders are dropped."""
    lines = raw.replace("‎", "").replace(" ", " ").splitlines()
    msgs: list[dict] = []
    wa = sum(1 for l in lines if _WA.match(l.strip())) >= max(1, len(lines) // 4)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _WA.match(line) if wa else _PLAIN.match(line)
        if m:
            speaker, text = (m.group(3), m.group(4)) if wa else (m.group(1), m.group(2))
            msgs.append({"speaker": speaker.strip(), "text": text.strip()})
        elif wa and msgs and not re.match(r"^\[?\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}", line):
            msgs[-1]["text"] += " " + line  # continuation of a multi-line message
    return [m for m in msgs if m["text"] and not any(s.lower() in m["text"].lower() for s in _SKIP)]


def with_context(msgs: list[dict]) -> list[dict]:
    prev = ""
    out = []
    for m in msgs:
        out.append({**m, "prev": prev})
        prev = m["text"]
    return out


def conversation_id(msgs: list[dict]) -> str:
    return hashlib.sha256(json.dumps([m["text"] for m in msgs]).encode()).hexdigest()[:12]


def load_all(store: Path = STORE) -> list[dict]:
    if not store.exists():
        return []
    return [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_conversation(conv: str, items: list[dict], store: Path = STORE) -> int:
    """Replace any earlier labels for this conversation with `items`. Returns lines saved."""
    for it in items:
        if it["label"] not in LABELS:
            raise ValueError(f"unknown label {it['label']!r}")
    keep = [r for r in load_all(store) if r["conversation"] != conv]
    rows = keep + [{"conversation": conv, "speaker": it.get("speaker", ""), "prev": it.get("prev", ""),
                    "text": it["text"], "label": it["label"], "model_pred": it.get("model_pred", "")}
                   for it in items]
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(items)


def stats(store: Path = STORE) -> dict:
    rows = load_all(store)
    c = Counter(r["label"] for r in rows)
    agreed = sum(1 for r in rows if r.get("model_pred") == r["label"])
    return {"messages": len(rows), "conversations": len({r["conversation"] for r in rows}),
            "per_label": {l: c[l] for l in LABELS},
            "model_agreement": round(agreed / len(rows), 3) if rows else None}
