"""Does reading future messages help or hurt? Same trained model, different inference inputs.

    python eval/context_ablation.py

Prompted by a chat where "I am sooo happpy" was labelled sadness because later messages
said "cry". Compares past+future vs past-only context, and collapsing stretched letters
("happpy" -> "happy"), on the four English test sets. Writes reports/context_ablation.json.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from pinch.classifier import DEFAULT_MODEL_DIR  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from pinch.erc import build_input  # noqa: E402
from training.text_datasets import dialogues  # noqa: E402
from training.train_emotion_erc import collate, score  # noqa: E402

STRETCH = re.compile(r"(\w)\1{2,}")


def unstretch(text: str) -> str:
    return STRETCH.sub(r"\1\1", text)


def main() -> None:
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_DIR).to(device).eval()
    rep = {}
    for future in (True, False):
        for norm in (False, True):
            name = f"{'past+future' if future else 'past-only'}{' +unstretch' if norm else ''}"
            rep[name] = {}
            for src in ("meld", "dailydialog", "goemotions", "dairai"):
                data = []
                for d in dialogues(src, "test"):
                    d2 = [(s, unstretch(t) if norm else t, l) for s, t, l in d]
                    for i, (_, _, l) in enumerate(d2):
                        if l is not None:
                            data.append((build_input(tok, d2, i, 256, future=future), LABELS.index(l)))
                order = sorted(range(len(data)), key=lambda k: len(data[k][0]))
                pred = np.zeros(len(data), int)
                with torch.no_grad():
                    for s in range(0, len(order), 64):
                        idx = order[s:s + 64]
                        b = collate([data[k] for k in idx], tok.pad_token_id)
                        with torch.autocast("cuda"):
                            z = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
                        pred[idx] = z.argmax(1).cpu().numpy()
                rep[name][src] = score([y for _, y in data], pred)
            rep[name]["mean_macro_f1"] = round(float(np.mean([rep[name][s]["macro_f1"] for s in ("meld", "dailydialog", "goemotions", "dairai")])), 3)
            print(f"{name:26} MELD wF1 {rep[name]['meld']['weighted_f1']:.3f} | mean macro {rep[name]['mean_macro_f1']:.3f} | "
                  f"DD macro {rep[name]['dailydialog']['macro_f1']:.3f}", flush=True)
    (ROOT / "reports" / "context_ablation.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
