"""Blind A/B: does the manga-expression LoRA (trained on 4,530 labelled Manga109 faces) draw the
requested emotion more often than plain SD-Turbo given the same manga-style prompt?

    python eval/manga_ab_test.py generate --lora models/lora/manga_expr --scale 1.0
    python eval/manga_ab_test.py score eval/manga_ab/labels.json

Same protocol as eval/lora_ab_test.py: 8 prompts per emotion from held-out characters and
places, same seed for both sides, shuffled unlabelled sheets, McNemar test on the pairs.
Writes reports/manga_ab.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.lora_ab_test import PER_EMOTION, TEST_LOOKS, TEST_PLACES, mcnemar_exact  # noqa: E402

OUT = ROOT / "eval" / "manga_ab"
MANGA_STYLE = "manga panel, black and white ink, screentone"
SHOTS = ["close-up of a face", "medium close-up", "medium shot"]


def prompt(emotion: str, rng: random.Random) -> str:
    from pinch.style import EXPRESSION, STORY_POSE
    return ", ".join([MANGA_STYLE, rng.choice(SHOTS), rng.choice(TEST_LOOKS), EXPRESSION[emotion],
                      STORY_POSE[emotion], rng.choice(TEST_PLACES)])


def generate(lora: str, scale: float, seed: int) -> None:
    from pinch.pipeline import load_env
    load_env()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import LocalTurboBackend

    rng = random.Random(seed)
    jobs = [(e, prompt(e, rng), rng.randrange(2**31)) for e in LABELS for _ in range(PER_EMOTION)]
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for variant, lora_arg in (("base", "none"), ("manga_lora", lora)):
        b = LocalTurboBackend(lora=lora_arg, lora_scale=scale)
        b._load()
        for p, (emotion, text, s) in enumerate(jobs):
            g = torch.Generator("cuda").manual_seed(s)
            img = b._pipe(text, num_inference_steps=4, guidance_scale=0.0, generator=g, width=512, height=512).images[0]
            images.append({"pair": p, "variant": variant, "emotion": emotion, "prompt": text, "img": img})
        del b
        torch.cuda.empty_cache()
        print(f"rendered {variant}", flush=True)

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
    (OUT / "items.json").write_text(json.dumps({"lora": lora, "scale": scale, "seed": seed, "items": items}, indent=1))
    print(f"{len(images)} images in {len(range(0, len(images), 20))} sheets")


def score(labels_file: Path) -> None:
    from eval.evaluate_panels import wilson
    run = json.loads((OUT / "items.json").read_text())
    items = {it["id"]: it for it in run["items"]}
    labels = json.loads(Path(labels_file).read_text())["labels"]
    hit = {i: labels[i] == items[i]["emotion"] for i in labels}
    report = {"lora": run["lora"], "scale": run["scale"], "variants": {}, "per_emotion": {}}
    for v in ("base", "manga_lora"):
        ids = [i for i in hit if items[i]["variant"] == v]
        k = sum(hit[i] for i in ids)
        report["variants"][v] = {"correct": k, "n": len(ids), "rate": round(k / len(ids), 3), "ci95": wilson(k, len(ids))}
    for e in sorted({it["emotion"] for it in items.values()}):
        report["per_emotion"][e] = {v: f"{sum(hit[i] for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                       f"/{sum(1 for i in hit if items[i]['variant'] == v and items[i]['emotion'] == e)}"
                                    for v in ("base", "manga_lora")}
    pairs: dict[int, dict[str, bool]] = {}
    for i, ok in hit.items():
        pairs.setdefault(items[i]["pair"], {})[items[i]["variant"]] = ok
    b = sum(1 for p in pairs.values() if p.get("base") and not p.get("manga_lora"))
    c = sum(1 for p in pairs.values() if p.get("manga_lora") and not p.get("base"))
    report["mcnemar"] = {"only_base_right": b, "only_lora_right": c, "p_value": round(mcnemar_exact(b, c), 4)}
    for v, r in report["variants"].items():
        print(f"{v:10} {r['correct']}/{r['n']} = {r['rate']:.0%}  CI {r['ci95']}")
    for e, r in report["per_emotion"].items():
        print(f"  {e:9} base {r['base']:5}  manga_lora {r['manga_lora']}")
    print(f"paired: only base right {b}, only LoRA right {c}, McNemar p = {report['mcnemar']['p_value']}")
    (ROOT / "reports" / "manga_ab.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--lora", default=str(ROOT / "models" / "lora" / "manga_expr"))
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=909)
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    generate(args.lora, args.scale, args.seed) if args.cmd == "generate" else score(Path(args.labels))


if __name__ == "__main__":
    main()
