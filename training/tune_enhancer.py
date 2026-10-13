"""Find per-emotion enhancer (emotion-direction steering) strengths with the Grey Wolf Optimizer.

    python training/tune_enhancer.py

Search space: one strength per non-neutral emotion, 0..0.8 (above ~0.5 images were seen to
break into texture). Fitness, measured on held-out characters/places (eval/lora_ab_test.py):
  mean over emotions of P(judge sees the intended emotion)
  minus a penalty when CLIP says the image stopped looking like a drawn person
  (P("abstract texture pattern") vs P("a comic drawing of a person's face")).
Same seeds for every candidate, so differences come from the strengths, not luck.
Judge-based fitness is a proxy; the winning setting must still pass a blind A/B before it
becomes the default. Writes reports/enhancer_gwo.json.
"""
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402

from eval.lora_ab_test import TEST_LOOKS, TEST_PLACES, TEST_SHOTS  # noqa: E402
from pinch.gwo import gwo  # noqa: E402
from pinch.imagegen import ImageRequest, LocalTurboBackend  # noqa: E402
from pinch.judge import judge  # noqa: E402
from training.build_expression_dataset import caption  # noqa: E402

EMOS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
PER_EMOTION = 3


def main() -> None:
    import os
    rng = random.Random(4242)
    jobs = [(e, caption(e, rng.choice(TEST_LOOKS), rng.choice(TEST_SHOTS), rng.choice(TEST_PLACES)), rng.randrange(2**31))
            for e in EMOS for _ in range(PER_EMOTION)]
    backend, j = LocalTurboBackend(lora="none"), judge("local")
    import torch
    sanity_txt = j.model.text_projection(j.model.text_model(**{k: v.to(j.device) for k, v in j.proc(
        text=["a comic drawing of a person's face", "an abstract texture pattern, noise"], return_tensors="pt",
        padding=True).items() if k in ("input_ids", "attention_mask")}).pooler_output)
    sanity_txt = torch.nn.functional.normalize(sanity_txt.float(), dim=-1).detach().cpu().numpy()
    evals = []

    def fitness(x) -> float:
        os.environ["PINCH_EMOTION_BOOSTS"] = json.dumps(dict(zip(EMOS, map(float, x))))
        imgs = [backend.generate(ImageRequest(t, s, 512, 512, emotion=e)) for e, t, s in jobs]
        probs = j.probs(imgs)
        target = np.mean([p[e] for p, (e, _, _) in zip(probs, jobs)])
        from PIL import Image
        with torch.no_grad():
            px = j.proc(images=[Image.open(io.BytesIO(b)).convert("RGB") for b in imgs], return_tensors="pt")["pixel_values"]
            emb = j.model.visual_projection(j.model.vision_model(pixel_values=px.to(j.device, j.dtype)).pooler_output)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
        z = np.exp(100 * emb @ sanity_txt.T)
        broken = float(np.mean(z[:, 1] / z.sum(1)))
        score = float(target - 2.0 * max(0.0, broken - 0.2))
        evals.append({"x": list(map(float, x)), "target": float(target), "broken": broken, "score": score})
        print(f"  eval {len(evals):3d} score {score:.3f} (judge {target:.3f}, broken {broken:.2f}) "
              + " ".join(f"{e[:4]}={v:.2f}" for e, v in zip(EMOS, x)), flush=True)
        return score

    base = fitness(np.zeros(len(EMOS)))
    best, best_score, hist = gwo(fitness, [0] * len(EMOS), [0.8] * len(EMOS), wolves=8, iters=6, seed=0,
                                 init=np.zeros(len(EMOS)))
    report = {"baseline_score": base, "best_score": best_score,
              "best_boosts": dict(zip(EMOS, map(float, best.round(3)))), "history": hist, "evals": evals}
    print("best boosts:", report["best_boosts"], f"score {base:.3f} -> {best_score:.3f}")
    (ROOT / "reports" / "enhancer_gwo.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
