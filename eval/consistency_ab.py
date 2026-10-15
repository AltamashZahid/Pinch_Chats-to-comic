"""Blind test: does a reference portrait (IP-Adapter) keep a character's face the same across
panels, and does it cost expression accuracy?

    python eval/consistency_ab.py generate [--ip-scale 0.5]
    python eval/consistency_ab.py score eval/consistency_ab/labels.json

4 held-out characters x 7 emotions, drawn by the anime backend two ways:
  A: text only (the remembered description, one seed per character, as in the app)
  B: the same, plus the character's neutral portrait as an IP-Adapter reference
Each sheet row is one character under one config: the portrait first, then its 7 panels in
shuffled order. Rows are shuffled and the config is hidden. The labeller marks every panel
"same person as the portrait?" and which emotion it shows. Writes reports/consistency_ab.json.
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402
from pinch.render import portrait_prompt  # noqa: E402,F401  (same portrait as the app)

load_env()

OUT = ROOT / "eval" / "consistency_ab"
CHARACTERS = [  # not used anywhere else; each with distinct hair, eyes, clothes
    "young man with short spiky black hair, brown eyes, round glasses and a green hoodie",
    "young woman with long wavy red hair, green eyes, a yellow sweater and a star hairclip",
    "teenage boy with messy blonde hair, blue eyes, a red baseball cap and a white t-shirt",
    "young woman with a short purple bob haircut, grey eyes, a black leather jacket and silver earrings",
]
PLACES = ["cozy living room", "school hallway", "busy cafe", "city street at night",
          "kitchen", "park bench", "bedroom with posters"]


def panel_prompt(look: str, emotion: str, place: str) -> str:
    from pinch.style import ART_STYLE, EXPRESSION, STORY_POSE
    return f"{ART_STYLE}, medium shot, {look}, {EXPRESSION[emotion]}, {STORY_POSE[emotion]}, {place}"


def generate(ip_scale: float, seed: int) -> None:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pinch.emotions import LABELS
    from pinch.imagegen import ImageRequest, LocalAnimeBackend

    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    b = LocalAnimeBackend(ip_scale=ip_scale)
    portraits, items = [], []
    for c, look in enumerate(CHARACTERS):
        s = 1000 + 7919 * c  # one seed per character, like cast_seed in the app
        raw = b.generate(ImageRequest(portrait_prompt(look), s, emotion="neutral"))
        portraits.append(raw)
        Image.open(io.BytesIO(raw)).save(OUT / f"portrait_{c}.png")
        for e, place in zip(LABELS, PLACES):
            for cfg in ("A_text", "B_reference"):
                ref = raw if cfg == "B_reference" else None
                img = b.generate(ImageRequest(panel_prompt(look, e, place), s, emotion=e, reference=ref))
                items.append({"character": c, "emotion": e, "config": cfg, "img": img})
        print(f"character {c + 1}/{len(CHARACTERS)} done", flush=True)
    del b
    torch.cuda.empty_cache()

    rows = [(c, cfg) for c in range(len(CHARACTERS)) for cfg in ("A_text", "B_reference")]
    rng.shuffle(rows)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    T, n = 200, 7
    sheet = Image.new("RGB", ((n + 1) * (T + 10) + 20, len(rows) * (T + 40) + 10), "white")
    d = ImageDraw.Draw(sheet)
    labelled = []
    for r, (c, cfg) in enumerate(rows):
        y = 10 + r * (T + 40)
        sheet.paste(Image.open(io.BytesIO(portraits[c])).resize((T, T)), (10, y + 30))
        d.rectangle([8, y + 28, 12 + T, y + 32 + T], outline="black", width=3)
        d.text((10, y + 2), f"row {r}: reference", font=font, fill="black")
        row = [it for it in items if it["character"] == c and it["config"] == cfg]
        rng.shuffle(row)
        for k, it in enumerate(row):
            pid = f"{r}{'abcdefg'[k]}"
            x = 20 + (k + 1) * (T + 10)
            sheet.paste(Image.open(io.BytesIO(it["img"])).resize((T, T)), (x, y + 30))
            d.text((x, y + 2), pid, font=font, fill="black")
            Image.open(io.BytesIO(it["img"])).save(OUT / f"{pid}.png")
            labelled.append({"id": pid, **{k2: v for k2, v in it.items() if k2 != "img"}})
    for half in (0, 1):  # two sheets so each stays readable
        h = len(rows) // 2
        sheet.crop((0, half * h * (T + 40), sheet.width, (half + 1) * h * (T + 40) + 10)).save(OUT / f"sheet_{half}.png")
    (OUT / "items.json").write_text(json.dumps({"ip_scale": ip_scale, "seed": seed, "rows": rows,
                                               "items": labelled}, indent=1))
    print(f"{len(labelled)} panels in 2 sheets")


def score(labels_file: Path) -> None:
    from eval.evaluate_panels import wilson
    from eval.lora_ab_test import mcnemar_exact
    run = json.loads((OUT / "items.json").read_text())
    items = {it["id"]: it for it in run["items"]}
    labels = json.loads(Path(labels_file).read_text())["labels"]
    rep = {"ip_scale": run["ip_scale"], "configs": {}}
    for cfg in ("A_text", "B_reference"):
        ids = [i for i in labels if items[i]["config"] == cfg]
        same = sum(labels[i]["same"] for i in ids)
        emo = sum(labels[i]["emotion"] == items[i]["emotion"] for i in ids)
        rep["configs"][cfg] = {"n": len(ids), "same_person": same, "same_rate": round(same / len(ids), 3),
                               "same_ci95": wilson(same, len(ids)), "emotion_correct": emo,
                               "emotion_rate": round(emo / len(ids), 3), "emotion_ci95": wilson(emo, len(ids))}
    pair = {}
    for i, lab in labels.items():
        it = items[i]
        pair.setdefault((it["character"], it["emotion"]), {})[it["config"]] = lab
    for what in ("same", "emotion"):
        ok = (lambda lab, key: lab["same"]) if what == "same" else \
             (lambda lab, key: lab["emotion"] == key[1])
        only_a = sum(1 for k, p in pair.items() if ok(p["A_text"], k) and not ok(p["B_reference"], k))
        only_b = sum(1 for k, p in pair.items() if ok(p["B_reference"], k) and not ok(p["A_text"], k))
        rep[f"mcnemar_{what}"] = {"only_A": only_a, "only_B": only_b, "p_value": round(mcnemar_exact(only_a, only_b), 4)}
    for cfg, r in rep["configs"].items():
        print(f"{cfg:12} same person {r['same_person']}/{r['n']} = {r['same_rate']:.0%}   "
              f"emotion {r['emotion_correct']}/{r['n']} = {r['emotion_rate']:.0%}")
    print("same person: ", rep["mcnemar_same"], "\nemotion:     ", rep["mcnemar_emotion"])
    (ROOT / "reports" / "consistency_ab.json").write_text(json.dumps(rep, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ip-scale", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    generate(args.ip_scale, args.seed) if args.cmd == "generate" else score(Path(args.labels))


if __name__ == "__main__":
    main()
