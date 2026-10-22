"""Streaming safetensors reader for BF16 models (no torch).

Reads tensors via numpy memmap so a 67 GB model never has to fit in RAM.
BF16 -> FP32 conversion is done by hand: bf16 is the top 16 bits of fp32,
so uint16 -> uint32 << 16 -> view float32.
"""

import json
import os
import struct

import numpy as np

_DTYPES = {
    "BF16": np.uint16,   # raw storage; converted on load
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
}


def _bf16_to_f32(arr_u16):
    """uint16 array of bf16 bit patterns -> float32 array (same shape)."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class ShardReader:
    """Memmaps one .safetensors shard; serves slices without loading the whole file."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.header_len = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(self.header_len))
        self.data_base = 8 + self.header_len
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def tensor_meta(self, name):
        info = self.header[name]
        return info["dtype"], tuple(info["shape"]), tuple(info["data_offsets"])

    def read(self, name, index=None, out_dtype=np.float32):
        """Read tensor `name` (or a numpy-style index into it) as out_dtype.

        index is applied on top of a memmap view, so only the requested
        pages are touched. BF16 storage is converted to float32.
        """
        dtype_s, shape, (start, end) = self.tensor_meta(name)
        np_dtype = _DTYPES[dtype_s]
        itemsize = np.dtype(np_dtype).itemsize
        assert (end - start) == itemsize * int(np.prod(shape)), name
        raw = self._mm[self.data_base + start: self.data_base + end]
        view = raw.view(np_dtype).reshape(shape)
        if index is not None:
            view = view[index]
        if dtype_s == "BF16":
            out = _bf16_to_f32(np.ascontiguousarray(view))
        else:
            out = view.astype(np.float32 if out_dtype == np.float32 else out_dtype)
        if out_dtype != np.float32:
            out = out.astype(out_dtype)
        return out


class ModelReader:
    """Index-aware reader over a sharded safetensors model directory."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        self.weight_map = idx["weight_map"]
        self._shards = {}

    def _shard(self, filename):
        if filename not in self._shards:
            self._shards[filename] = ShardReader(os.path.join(self.model_dir, filename))
        return self._shards[filename]

    def meta(self, name):
        return self._shard(self.weight_map[name]).tensor_meta(name)

    def read(self, name, index=None, out_dtype=np.float32):
        return self._shard(self.weight_map[name]).read(name, index, out_dtype)


if __name__ == "__main__":
    import sys
    mr = ModelReader(sys.argv[1] if len(sys.argv) > 1 else
                     r"d:\xdna_llm\models\Ornith-1.5-35B-A3B")
    # smoke test: slice one expert out of the fused tensor
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    print(name, mr.meta(name))
    gate = mr.read(name, (0, slice(0, 512), slice(None)))
    print("expert0 gate:", gate.shape, gate.dtype,
          "absmax", float(np.abs(gate).max()), "rms", float(np.sqrt((gate ** 2).mean())))
