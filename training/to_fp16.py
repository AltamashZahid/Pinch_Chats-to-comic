"""Copy a model's safetensors file to fp16 without memory-mapping it.

    python training/to_fp16.py h94/IP-Adapter models/image_encoder models/ip_adapter/image_encoder

Copies <subfolder>/config.json and writes <subfolder>/model.safetensors as fp16 (half the
size). The hub's 2.5 GB fp32 IP-Adapter image encoder can't be memory-mapped with a small
Windows pagefile, the same failure as the Animagine UNet (see training/shard_unet.py).
"""
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pinch.tensor_io import read_tensors  # noqa: E402


def main(repo: str, subfolder: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(repo, f"{subfolder}/config.json"), out_dir / "config.json")
    src = hf_hub_download(repo, f"{subfolder}/model.safetensors")
    tensors = {k: (t.half() if t.is_floating_point() else t) for k, t in read_tensors(src)}
    save_file(tensors, out_dir / "model.safetensors", metadata={"format": "pt"})
    print(f"{len(tensors)} tensors -> {out_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
