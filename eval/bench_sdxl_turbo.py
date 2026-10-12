"""Can SDXL-Turbo run on a 4 GB GPU, and how fast? Tries three memory strategies.

    python eval/bench_sdxl_turbo.py

SDXL-Turbo's UNet is ~5 GB in fp16, bigger than the GPU, so something has to give:
- sequential: diffusers moves each layer to the GPU just when it runs (fits anything, slow)
- int8: UNet weights quantised to 8-bit with bitsandbytes (~2.6 GB, stays on the GPU);
        text encoders and VAE offloaded per component
- nf4:  UNet in 4-bit NF4 (~1.5 GB)
Prints seconds per image and peak VRAM for each, and saves the same 3 prompts per strategy
to D:/tmp/sdxl_<strategy>.png for a visual check.
"""
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pinch.pipeline import load_env  # noqa: E402

load_env()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

BASE = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"
PROMPTS = [
    "comic panel, clean ink lines, flat colors, close-up, young woman with curly red hair, crying face, tears streaming down cheeks, trembling lip, head down, wiping away tears, cozy living room",
    "comic panel, clean ink lines, flat colors, medium shot, young man with short black hair and a blue hoodie, happy face, big smile, while everything around them falls apart, rain cloud over head, college campus",
    "comic panel, clean ink lines, flat colors, close-up, older man with grey beard, grossed out face, tongue sticking out, nose scrunched, recoiling, covering nose with hand, kitchen",
]


def load(strategy: str):
    from diffusers import AutoencoderKL, AutoPipelineForText2Image, UNet2DConditionModel
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    if strategy == "sequential":
        pipe = AutoPipelineForText2Image.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16, variant="fp16")
        pipe.enable_sequential_cpu_offload()
    else:
        from diffusers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_8bit=True) if strategy == "int8" else \
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float16, quantization_config=q)
        pipe = AutoPipelineForText2Image.from_pretrained(BASE, unet=unet, vae=vae, torch_dtype=torch.float16,
                                                         variant="fp16")
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    strategies = sys.argv[1:] or ["int8", "nf4", "sequential"]
    for strategy in strategies:
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            pipe = load(strategy)
            load_s = time.time() - t
            imgs, times = [], []
            for i, p in enumerate(PROMPTS):
                t = time.time()
                g = torch.Generator("cpu").manual_seed(i + 1)
                imgs.append(pipe(p, num_inference_steps=4, guidance_scale=0.0, width=512, height=512,
                                 generator=g).images[0])
                times.append(time.time() - t)
            sheet = Image.new("RGB", (3 * 512, 512))
            for i, im in enumerate(imgs):
                sheet.paste(im, (512 * i, 0))
            sheet.save(f"D:/tmp/sdxl_{strategy}.png")
            print(f"{strategy}: load {load_s:.0f}s, per image {sum(times[1:]) / 2:.1f}s "
                  f"(first {times[0]:.1f}s), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
            del pipe
        except Exception as e:  # report and try the next strategy
            print(f"{strategy}: FAILED {type(e).__name__}: {str(e)[:200]}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
