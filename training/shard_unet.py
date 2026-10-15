"""Re-save a big diffusers UNet as ~1 GB shards, streaming one tensor at a time.

    python training/shard_unet.py cagliostrolab/animagine-xl-4.0 models/animagine_unet

safetensors memory-maps the file copy-on-write, and Windows reserves commit charge for the
whole mapping (5.1 GB for an SDXL UNet). With a small pagefile that fails as an access
violation. Here tensors are read with plain file reads, and the ~1 GB shards it writes are
small enough for diffusers to map one at a time.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

SHARD_BYTES = 1 << 30
DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def read_tensors(path: str):
    """Yield (name, tensor) from a .safetensors file without memory-mapping it."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for k, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            buf = bytearray(f.read(b - a))
            yield k, torch.frombuffer(buf, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).clone()


def main(repo: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo, "unet/diffusion_pytorch_model.safetensors")
    shutil.copy(hf_hub_download(repo, "unet/config.json"), out_dir / "config.json")
    weight_map, shard, size, n = {}, {}, 0, 0
    names = []

    def flush():
        nonlocal shard, size, n
        if shard:
            n += 1
            name = f"diffusion_pytorch_model-{n:05d}.safetensors"
            save_file(shard, out_dir / name)
            names.append(name)
            for k in shard:
                weight_map[k] = name
            shard, size = {}, 0

    for k, t in read_tensors(src):
        shard[k] = t
        size += t.numel() * t.element_size()
        if size >= SHARD_BYTES:
            flush()
    flush()
    total = len(names)
    for i, name in enumerate(names, 1):  # rename to the -0000i-of-0000N pattern diffusers expects
        final = name.replace(".safetensors", f"-of-{total:05d}.safetensors")
        (out_dir / name).rename(out_dir / final)
        for k, v in weight_map.items():
            if v == name:
                weight_map[k] = final
    (out_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    print(f"{len(weight_map)} tensors in {total} shards -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
