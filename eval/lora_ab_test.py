"""Blind A/B test: does the expressions LoRA make SD-Turbo draw the requested emotion more often?

    python eval/lora_ab_test.py generate [--lora models/lora/expressions]
        For 8 prompts per emotion, built from characters and places that are NOT in the
        LoRA's training data, render the same prompt + seed with the base model and with the
        LoRA. Shuffle everything into unlabelled sheets eval/lora_ab/sheet_<n>.png; which image
        is which (variant, requested emotion) is stored in eval/lora_ab/items.json.

    python eval/lora_ab_test.py score eval/lora_ab/labels.json
        labels.json maps image id -> the emotion a blind labeller sees. Reports the hit rate
        (labelled == requested) for base vs LoRA with 95% CIs, per emotion, and a paired
        McNemar exact test over prompt pairs. Writes reports/lora_ab.json.
"""
import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "eval" / "lora_ab"
PER_EMOTION = 8
# Held out: none of these characters or places appear in training/build_expression_dataset.py.
TEST_LOOKS = [
    "freckled teenage girl with red pigtails and overalls",
    "tall man with a black beard and a beanie",
    "elderly woman with a white bun and round spectacles",
    "young man with dreadlocks and a yellow raincoat",
    "middle-aged woman with a pixie cut and a business suit",
    "boy with a buzz cut and a soccer jersey",
    "woman with long silver hair and a green scarf",
    "bald man with tattoos and a tank top",
]
TEST_PLACES = ["airport gate", "rainy rooftop", "hospital waiting room", "supermarket aisle",
               "library", "football stadium", "camping tent", "elevator"]
TEST_SHOTS = ["close-up", "medium close-up", "medium shot"]


def generate(lora: Path, seed: int) -> None:
    from pinch.pipeline import load_env
    load_env()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import LocalTurboBackend
    from training.build_expression_dataset import caption

    rng = random.Random(seed)
    prompts = [(e, caption(e, rng.choice(TEST_LOOKS), rng.choice(TEST_SHOTS), rng.choice(TEST_PLACES)),
                rng.randrange(2**31)) for e in LABELS for _ in range(PER_EMOTION)]
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for variant, lora_arg in (("base", "none"), ("lora", str(lora))):
        backend = LocalTurboBackend(lora=lora_arg)
        backend._load()
        for p, (emotion, text, s) in enumerate(prompts):
            g = torch.Generator("cuda").manual_seed(s)
            img = backend._pipe(text, num_inference_steps=4, guidance_scale=0.0, generator=g,
                                width=512, height=512).images[0]
            images.append({"pair": p, "variant": variant, "emotion": emotion, "prompt": text, "img": img})
        del backend
        torch.cuda.empty_cache()
        print(f"rendered {variant}")

    rng.shuffle(images)  # blind: position says nothing about variant or requested emotion
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
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    print(f"{len(images)} images in {len(range(0, len(images), 20))} sheets")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for discordant pairs b (only base right) vs c (only LoRA right)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score(labels_file: Path) -> None:
    from eval.evaluate_panels import wilson
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())}
    labels = json.loads(Path(labels_file).read_text())["labels"]
    hit = {i: labels[i] == items[i]["emotion"] for i in labels}

    report = {"variants": {}, "per_emotion": {}}
    for v in ("base", "lora"):
        ids = [i for i in hit if items[i]["variant"] == v]
        k = sum(hit[i] for i in ids)
        report["variants"][v] = {"correct": k, "n": len(ids), "rate": round(k / len(ids), 3), "ci95": wilson(k, len(ids))}
    emotions = sorted({it["emotion"] for it in items.values()})
    for e in emotions:
        report["per_emotion"][e] = {v: f"{sum(hit[i] for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                       f"/{sum(1 for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                    for v in ("base", "lora")}
    pairs: dict[int, dict[str, bool]] = {}
    for i, ok in hit.items():
        pairs.setdefault(items[i]["pair"], {})[items[i]["variant"]] = ok
    b = sum(1 for p in pairs.values() if p.get("base") and not p.get("lora"))
    c = sum(1 for p in pairs.values() if p.get("lora") and not p.get("base"))
    report["mcnemar"] = {"only_base_right": b, "only_lora_right": c, "p_value": round(mcnemar_exact(b, c), 4)}

    for v, r in report["variants"].items():
        print(f"{v:5} {r['correct']}/{r['n']} = {r['rate']:.0%}  CI {r['ci95']}")
    for e, r in report["per_emotion"].items():
        print(f"  {e:9} base {r['base']:5}  lora {r['lora']}")
    print(f"paired: only base right {b}, only LoRA right {c}, McNemar p = {report['mcnemar']['p_value']}")
    (ROOT / "reports" / "lora_ab.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--lora", default=str(ROOT / "models" / "lora" / "expressions"))
    g.add_argument("--seed", type=int, default=2024)
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(Path(args.lora), args.seed)
    else:
        score(Path(args.labels))


if __name__ == "__main__":
    main()
