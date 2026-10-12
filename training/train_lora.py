"""Fine-tune a LoRA on SD-Turbo so it draws clear comic facial expressions.

    python training/train_lora.py                     # data/expressions/train -> models/lora/expressions
    python training/train_lora.py --steps 200 --out D:/tmp/lora_smoke   # quick check

Fits a 4 GB laptop GPU:
- the VAE and text encoder run once up front: image latents (plus mirrored copies) and
  caption embeddings are cached, then both models leave the GPU;
- the UNet stays frozen in fp16 with gradient checkpointing; only the LoRA matrices
  (rank 16 on the attention projections, ~3M params) train, in fp32, under fp16 autocast.

Standard denoising objective: add noise at a random timestep, predict the noise (SD-Turbo's
scheduler is epsilon-prediction), MSE. Saves diffusers-format LoRA weights plus a
training log and loss curve.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

BASE = "stabilityai/sd-turbo"


@torch.no_grad()
def encode_dataset(data_dir: Path, device) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    rows = [json.loads(l) for l in (data_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l]
    # variant="fp16": reuse the half-precision files the app already downloaded (~2.5 GB, not ~5 GB).
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", variant="fp16", torch_dtype=torch.float16).to(device)
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", variant="fp16",
                                       torch_dtype=torch.float16).to(device)

    latents, embeds = [], []
    for r in rows:
        img = Image.open(data_dir / r["file_name"]).convert("RGB").resize((512, 512))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).half().div(127.5).sub(1).unsqueeze(0).to(device)
        pair = torch.cat([x, x.flip(-1)])  # mirrored copy: free augmentation, faces are symmetric enough
        latents.append((vae.encode(pair).latent_dist.sample() * vae.config.scaling_factor).half().cpu())
        ids = tok(r["text"], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        embeds.append(te(ids)[0].cpu())
    del vae, te
    torch.cuda.empty_cache()
    return torch.stack(latents), torch.cat(embeds), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "expressions" / "train"))
    ap.add_argument("--out", default=str(ROOT / "models" / "lora" / "expressions"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation (effective batch)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    latents, embeds, rows = encode_dataset(Path(args.data), device)
    print(f"cached {len(rows)} images ({len(rows) * 2} with mirrors) in {time.time() - t0:.0f}s")

    noise_sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    assert noise_sched.config.prediction_type == "epsilon"
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", variant="fp16",
                                                torch_dtype=torch.float16).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)  # LoRA params in fp32, frozen base stays fp16
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f}M")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    warmup = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    scaler = torch.amp.GradScaler()

    log, running, start = [], 0.0, time.time()
    unet.train()
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            i, flip = random.randrange(len(rows)), random.randrange(2)
            x0 = latents[i, flip:flip + 1].to(device, torch.float32)
            cond = embeds[i:i + 1].to(device)
            noise = torch.randn_like(x0)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device)
            noisy = noise_sched.add_noise(x0, noise, t)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy.half(), t, encoder_hidden_states=cond).sample
            loss = F.mse_loss(pred.float(), noise) / args.accum
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if step % 25 == 0:
            log.append({"step": step, "loss": round(running / 25, 5), "lr": sched.get_last_lr()[0]})
            running = 0.0
        if step % 100 == 0:
            el = time.time() - start
            print(f"step {step}/{args.steps} loss {log[-1]['loss']:.4f} "
                  f"({el / step:.2f}s/step, {torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak)")
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out if step == args.steps else out / f"step_{step}"
            StableDiffusionPipeline.save_lora_weights(ckpt, unet_lora_layers=get_peft_model_state_dict(unet),
                                                      safe_serialization=True)

    (out / "train_log.json").write_text(json.dumps({
        "base": BASE, "images": len(rows), "steps": args.steps, "rank": args.rank, "lr": args.lr,
        "accum": args.accum, "minutes": round((time.time() - start) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "log": log}, indent=1))
    print(f"saved LoRA to {out}")


if __name__ == "__main__":
    main()
