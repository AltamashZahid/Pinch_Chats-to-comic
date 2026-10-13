"""Blind A/B test between two image backends: which draws the requested emotion more often?

    python eval/backend_ab_test.py generate --a local --b sdxl
    python eval/backend_ab_test.py score eval/backend_ab/labels.json

Same held-out prompts and protocol as eval/lora_ab_test.py (8 prompts per emotion from
characters and places not used anywhere else, same seed for both sides), so results are
comparable. Images are shuffled into unlabelled sheets; labels.json holds a blind
labeller's reading of each face. Scoring reports hit rate per backend with 95% CIs, per
emotion, and a paired McNemar test. Writes reports/backend_ab.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.lora_ab_test import PER_EMOTION, TEST_LOOKS, TEST_PLACES, TEST_SHOTS, mcnemar_exact  # noqa: E402

OUT = ROOT / "eval" / "backend_ab"


def generate(a: str, b: str, seed: int, b_lora: str = "none") -> None:
    import io
    from pinch.pipeline import load_env
    load_env()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import BACKENDS, ImageRequest
    from training.build_expression_dataset import caption

    rng = random.Random(seed)
    prompts = [(e, caption(e, rng.choice(TEST_LOOKS), rng.choice(TEST_SHOTS), rng.choice(TEST_PLACES)),
                rng.randrange(2**31)) for e in LABELS for _ in range(PER_EMOTION)]
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    import inspect
    for variant, lora in ((a, "none"), (b, b_lora)):
        cls = BACKENDS[variant]
        backend = cls(lora=lora) if "lora" in inspect.signature(cls).parameters else cls()
        for p, (emotion, text, s) in enumerate(prompts):
            shot = next((x for x in ("extreme close-up", "medium close-up", "close-up", "medium shot") if x in text), "close-up")
            img = Image.open(io.BytesIO(backend.generate(ImageRequest(text, s, 512, 512, emotion=emotion, shot=shot)))).convert("RGB")
            images.append({"pair": p, "variant": variant, "emotion": emotion, "prompt": text, "img": img})
        del backend
        torch.cuda.empty_cache()
        print(f"rendered {variant}")

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
    (OUT / "items.json").write_text(json.dumps({"a": a, "b": b, "seed": seed, "items": items}, indent=1))
    print(f"{len(images)} images in {len(range(0, len(images), 20))} sheets")


def score(labels_file: Path) -> None:
    from eval.evaluate_panels import wilson
    run = json.loads((OUT / "items.json").read_text())
    items = {it["id"]: it for it in run["items"]}
    labels = json.loads(Path(labels_file).read_text())["labels"]
    hit = {i: labels[i] == items[i]["emotion"] for i in labels}
    a, b = run["a"], run["b"]
    report = {"a": a, "b": b, "variants": {}, "per_emotion": {}}
    for v in (a, b):
        ids = [i for i in hit if items[i]["variant"] == v]
        k = sum(hit[i] for i in ids)
        report["variants"][v] = {"correct": k, "n": len(ids), "rate": round(k / len(ids), 3), "ci95": wilson(k, len(ids))}
    for e in sorted({it["emotion"] for it in items.values()}):
        report["per_emotion"][e] = {v: f"{sum(hit[i] for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                       f"/{sum(1 for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                    for v in (a, b)}
    pairs: dict[int, dict[str, bool]] = {}
    for i, ok in hit.items():
        pairs.setdefault(items[i]["pair"], {})[items[i]["variant"]] = ok
    only_a = sum(1 for p in pairs.values() if p.get(a) and not p.get(b))
    only_b = sum(1 for p in pairs.values() if p.get(b) and not p.get(a))
    report["mcnemar"] = {f"only_{a}_right": only_a, f"only_{b}_right": only_b,
                         "p_value": round(mcnemar_exact(only_a, only_b), 4)}
    for v, r in report["variants"].items():
        print(f"{v:6} {r['correct']}/{r['n']} = {r['rate']:.0%}  CI {r['ci95']}")
    for e, r in report["per_emotion"].items():
        print(f"  {e:9} {a} {r[a]:5}  {b} {r[b]}")
    print(f"paired: only {a} right {only_a}, only {b} right {only_b}, McNemar p = {report['mcnemar']['p_value']}")
    (ROOT / "reports" / "backend_ab.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--a", default="local")
    g.add_argument("--b", default="sdxl")
    g.add_argument("--seed", type=int, default=5151)
    g.add_argument("--b-lora", default="none", help="LoRA for backend b (a always runs without one)")
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.a, args.b, args.seed, args.b_lora)
    else:
        score(Path(args.labels))


if __name__ == "__main__":
    main()
