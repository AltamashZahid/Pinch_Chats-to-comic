"""End-to-end blind A/B of the full panel pipeline, best-of-N included.

    python eval/pipeline_ab_test.py generate
    python eval/pipeline_ab_test.py score eval/pipeline_ab/labels.json

Config A: SD-Turbo + its judge, best of 4 (the previous default).
Config B: SD 1.5 + LCM + expression LoRA + its own judge, best of 6 (hard emotions) / 2 (joy, neutral).
Prompts use the held-out characters/places of eval/lora_ab_test.py, which neither judge was
trained on. Same seeds on both sides. Images are shuffled into unlabelled sheets for blind
labelling. Writes reports/pipeline_ab.json.
"""
import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.lora_ab_test import PER_EMOTION, TEST_LOOKS, TEST_PLACES, TEST_SHOTS, mcnemar_exact  # noqa: E402

OUT = ROOT / "eval" / "pipeline_ab"
CONFIGS = {
    "A_turbo_best4": {"backend": "local", "lora": "none", "candidates": "4", "easy": "4"},
    "B_sd15lora_best6": {"backend": "lcm", "lora": str(ROOT / "models" / "lora" / "expr_sd15"),
                         "candidates": "6", "easy": "2"},
}


def generate(seed: int) -> None:
    from pinch.pipeline import load_env
    load_env()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import BACKENDS
    from pinch.render import draw_best
    from pinch.scenes import Line, Panel
    from training.build_expression_dataset import caption

    rng = random.Random(seed)
    jobs = []
    for e in LABELS:
        for _ in range(PER_EMOTION):
            shot = rng.choice(TEST_SHOTS)
            jobs.append((e, shot, caption(e, rng.choice(TEST_LOOKS), shot, rng.choice(TEST_PLACES)), rng.randrange(2**31)))
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for name, cfg in CONFIGS.items():
        os.environ["PINCH_CANDIDATES"], os.environ["PINCH_CANDIDATES_EASY"] = cfg["candidates"], cfg["easy"]
        cls = BACKENDS[cfg["backend"]]
        backend = cls(lora=cfg["lora"], lora_scale=1.0) if cfg["lora"] != "none" else cls(lora="none")
        for p, (emotion, shot, text, s) in enumerate(jobs):
            panel = Panel([Line("me", "x", emotion, 0.9)], "me", shot=shot, prompt=text)
            img_bytes, tried, prob = draw_best(backend, panel, s, 512, 512)
            images.append({"pair": p, "variant": name, "emotion": emotion, "prompt": text, "tried": tried,
                           "img": Image.open(io.BytesIO(img_bytes)).convert("RGB")})
            if (p + 1) % 14 == 0:
                print(f"  {name}: {p + 1}/{len(jobs)}", flush=True)
        del backend
        torch.cuda.empty_cache()
        print(f"rendered {name}", flush=True)

    rng.shuffle(images)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    items = []
    for i, it in enumerate(images):
        it["img"].save(OUT / f"{i:03d}.png")
        items.append({"id": f"{i:03d}", **{k: v for k, v in it.items() if k != "img"}})
    for n in range(0, len(images), 20):
        sheet = Image.new("RGB", (5 * 260, 4 * 285), "white")
        d = ImageDraw.Draw(sheet)
        for j, it in enumerate(images[n:n + 20]):
            x, y = (j % 5) * 260, (j // 5) * 285
            sheet.paste(it["img"].resize((250, 250)), (x + 5, y + 30))
            d.text((x + 8, y + 2), f"{n + j:03d}", font=font, fill="black")
        sheet.save(OUT / f"sheet_{n // 20}.png")
    (OUT / "items.json").write_text(json.dumps({"seed": seed, "configs": CONFIGS, "items": items}, indent=1))
    print(f"{len(images)} images in {len(range(0, len(images), 20))} sheets")


def score(labels_file: Path) -> None:
    from eval.evaluate_panels import wilson
    run = json.loads((OUT / "items.json").read_text())
    items = {it["id"]: it for it in run["items"]}
    labels = json.loads(Path(labels_file).read_text())["labels"]
    hit = {i: labels[i] == items[i]["emotion"] for i in labels}
    names = list(CONFIGS)
    report = {"variants": {}, "per_emotion": {}}
    for v in names:
        ids = [i for i in hit if items[i]["variant"] == v]
        k = sum(hit[i] for i in ids)
        report["variants"][v] = {"correct": k, "n": len(ids), "rate": round(k / len(ids), 3), "ci95": wilson(k, len(ids))}
    for e in sorted({it["emotion"] for it in items.values()}):
        report["per_emotion"][e] = {v: f"{sum(hit[i] for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                       f"/{sum(1 for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                    for v in names}
    pairs: dict[int, dict[str, bool]] = {}
    for i, ok in hit.items():
        pairs.setdefault(items[i]["pair"], {})[items[i]["variant"]] = ok
    a, b = names
    only_a = sum(1 for p in pairs.values() if p.get(a) and not p.get(b))
    only_b = sum(1 for p in pairs.values() if p.get(b) and not p.get(a))
    report["mcnemar"] = {f"only_{a}": only_a, f"only_{b}": only_b, "p_value": round(mcnemar_exact(only_a, only_b), 4)}
    for v, r in report["variants"].items():
        print(f"{v:18} {r['correct']}/{r['n']} = {r['rate']:.0%}  CI {r['ci95']}")
    for e, r in report["per_emotion"].items():
        print(f"  {e:9} " + "  ".join(f"{v} {r[v]}" for v in names))
    print(f"paired: only A {only_a}, only B {only_b}, McNemar p = {report['mcnemar']['p_value']}")
    (ROOT / "reports" / "pipeline_ab.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, default=31337)
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    generate(args.seed) if args.cmd == "generate" else score(Path(args.labels))


if __name__ == "__main__":
    main()
