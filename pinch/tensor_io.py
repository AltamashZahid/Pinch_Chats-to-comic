"""Read .safetensors files without memory-mapping them.

safetensors (and so transformers/diffusers) memory-maps a weights file copy-on-write, and
Windows charges the whole file against the system commit limit (RAM + pagefile) up front.
On this laptop that limit is nearly used by other programs, so mapping a 1-5 GB file fails
("paging file is too small", or an access violation). Plain reads one tensor at a time need
only that tensor's memory, and each can go straight to the GPU.
"""
import json
import struct

import torch

DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
          "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}


def read_tensors(path):
    """Yield (name, tensor) from a .safetensors file, in file order."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for k, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            buf = bytearray(f.read(b - a))
            yield k, torch.frombuffer(buf, dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
