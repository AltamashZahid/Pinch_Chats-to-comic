"""Is an explicit self-report ("I am so happy") more accurate than the conversation model?

    python eval/self_report_eval.py

On the four English test sets: for lines where pinch/self_report.py fires, compare its label
and the model's (past+future context) with the gold label. Writes reports/self_report.json.
"""
import json
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
from pinch.self_report import self_report  # noqa: E402
from training.text_datasets import dialogues  # noqa: E402
from training.train_emotion_erc import collate, score  # noqa: E402


def main() -> None:
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_DIR).to(device).eval()
    rep = {}
    for src in ("meld", "dailydialog", "goemotions", "dairai"):
        data, past_only, texts = [], [], []
        for d in dialogues(src, "test"):
            for i, (_, t, l) in enumerate(d):
                if l is not None:
                    data.append((build_input(tok, d, i, 256), LABELS.index(l)))
                    past_only.append((build_input(tok, d, i, 256, future=False), LABELS.index(l)))
                    texts.append(t)

        def run(items):
            order = sorted(range(len(items)), key=lambda k: len(items[k][0]))
            out = np.zeros(len(items), int)
            with torch.no_grad():
                for s in range(0, len(order), 64):
                    idx = order[s:s + 64]
                    b = collate([items[k] for k in idx], tok.pad_token_id)
                    with torch.autocast("cuda"):
                        z = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
                    out[idx] = z.argmax(1).cpu().numpy()
            return out
        pred, pred_past = run(data), run(past_only)
        gold = np.array([y for _, y in data])
        rule = [self_report(t) for t in texts]
        hit = [k for k, r in enumerate(rule) if r]
        with_rule = pred.copy()
        for k in hit:
            with_rule[k] = LABELS.index(rule[k])
        # Variant B: the self-report wins only if the model, reading without the future, agrees.
        agree = [k for k in hit if pred_past[k] == LABELS.index(rule[k])]
        with_agree = pred.copy()
        for k in agree:
            with_agree[k] = pred_past[k]
        rep[src] = {"lines": len(gold), "rule_fires": len(hit),
                    "on_those_lines": {"rule_accuracy": round(float(np.mean([LABELS.index(rule[k]) == gold[k] for k in hit])), 3) if hit else None,
                                       "model_accuracy": round(float(np.mean([pred[k] == gold[k] for k in hit])), 3) if hit else None},
                    "macro_f1_model": score(gold.tolist(), pred)["macro_f1"],
                    "macro_f1_with_rule": score(gold.tolist(), with_rule)["macro_f1"],
                    "weighted_f1_model": score(gold.tolist(), pred)["weighted_f1"],
                    "weighted_f1_with_rule": score(gold.tolist(), with_rule)["weighted_f1"],
                    "agreement_variant": {"fires": len(agree), "changed": int(sum(with_agree[k] != pred[k] for k in agree)),
                                          "changed_right": int(sum(with_agree[k] != pred[k] and with_agree[k] == gold[k] for k in agree)),
                                          "changed_wrong": int(sum(with_agree[k] != pred[k] and pred[k] == gold[k] for k in agree)),
                                          "macro_f1": score(gold.tolist(), with_agree)["macro_f1"],
                                          "weighted_f1": score(gold.tolist(), with_agree)["weighted_f1"]}}
        r = rep[src]
        print(f"{src:12} fires on {r['rule_fires']:4}/{r['lines']} lines | there: rule {r['on_those_lines']['rule_accuracy']} vs model "
              f"{r['on_those_lines']['model_accuracy']} | macro F1 {r['macro_f1_model']} -> {r['macro_f1_with_rule']} | "
              f"weighted F1 {r['weighted_f1_model']} -> {r['weighted_f1_with_rule']}", flush=True)
        a = r["agreement_variant"]
        print(f"   agreement variant: fires {a['fires']}, changes {a['changed']} labels ({a['changed_right']} fixed, "
              f"{a['changed_wrong']} broken) | macro F1 {r['macro_f1_model']} -> {a['macro_f1']} | weighted F1 -> {a['weighted_f1']}", flush=True)
    (ROOT / "reports" / "self_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
