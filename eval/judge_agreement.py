"""How good is each automatic expression judge? Compare against blind human-style labels.

    python eval/judge_agreement.py

Uses eval/judge_set (from make_judge_set.py) and eval/judge_set/labels_*.json. Reports:
- each judge's accuracy against the labels (which judge to trust),
- how often SD-Turbo actually drew the requested emotion according to the labels, per
  emotion and per sampler setting (the true generation quality, independent of any judge).
Writes reports/judge_agreement.json.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from eval.evaluate_panels import FACE, wilson  # noqa: E402
from pinch.emotions import LABELS  # noqa: E402
from pinch.vlm_judge import VLMJudge  # noqa: E402

SET = ROOT / "eval" / "judge_set"


class ClipJudge:
    def __init__(self, name: str):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        enc = self.proc(text=[f"a comic drawing of a person with a {FACE[l]} face" for l in LABELS],
                        return_tensors="pt", padding=True)
        with torch.no_grad():
            t = self.model.text_projection(self.model.text_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output)
        self.text = torch.nn.functional.normalize(t, dim=-1)

    @torch.no_grad()
    def emotion(self, path: Path) -> str:
        px = self.proc(images=[Image.open(path).convert("RGB")], return_tensors="pt")["pixel_values"]
        e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
        return LABELS[int((torch.nn.functional.normalize(e, dim=-1) @ self.text.T).argmax())]


def main() -> None:
    items = {it["id"]: it for it in json.loads((SET / "items.json").read_text())}
    labels = json.loads((SET / "labels_claude.json").read_text())["labels"]
    ids = sorted(labels)

    judges = {"clip_b16": ClipJudge("openai/clip-vit-base-patch16"),
              "clip_l14": ClipJudge("openai/clip-vit-large-patch14"),
              "vlm_qwen": VLMJudge()}
    preds: dict[str, dict[str, str]] = {}
    for name, judge in judges.items():
        cache = SET / f"preds_{name}.json"  # resume without re-calling the API
        p = json.loads(cache.read_text()) if cache.exists() else {}
        for i in ids:
            if i not in p:
                p[i] = judge.emotion(SET / f"{i}.png")
                cache.write_text(json.dumps(p, indent=1))
        preds[name] = p
        print(f"judged with {name}")

    report = {"n": len(ids), "judges": {}}
    for name, p in preds.items():
        k = sum(p[i] == labels[i] for i in ids)
        report["judges"][name] = {"accuracy_vs_labels": round(k / len(ids), 3), "ci95": wilson(k, len(ids)),
                                  "predicted": dict(Counter(p.values()).most_common())}
        print(f"{name:9} agrees with labels {k}/{len(ids)} = {k / len(ids):.0%}  CI {wilson(k, len(ids))}  "
              f"predicts {dict(Counter(p.values()).most_common(3))}")

    # True generation quality, using the labels rather than any judge.
    by_emotion, by_cfg = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i in ids:
        ok = labels[i] == items[i]["emotion"]
        by_emotion[items[i]["emotion"]][0] += ok
        by_emotion[items[i]["emotion"]][1] += 1
        by_cfg[str(items[i]["cfg"])][0] += ok
        by_cfg[str(items[i]["cfg"])][1] += 1
    report["generation_by_emotion"] = {e: f"{a}/{n}" for e, (a, n) in by_emotion.items()}
    report["generation_by_cfg"] = {c: {"correct": f"{a}/{n}", "ci95": wilson(a, n)} for c, (a, n) in by_cfg.items()}
    print("SD-Turbo drew the requested emotion (per labels):",
          "  ".join(f"{e} {v}" for e, v in report["generation_by_emotion"].items()))
    print("by sampler:", "  ".join(f"cfg {c}: {v['correct']} {v['ci95']}" for c, v in report["generation_by_cfg"].items()))
    (ROOT / "reports" / "judge_agreement.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
