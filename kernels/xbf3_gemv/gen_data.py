#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate XBF3 GEMV test vectors (2048x512) using the reference quantizer.

Writes to --out dir:
  wts.bin     uint8[401408]  XBF3 packed, row-major (2 superblocks/row, 98B each)
  act.bin     float32[512]
  out_ref.bin float32[2048]  reference y = dequant(w) @ x
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "quant"))
import xbf3

OUT, IN, SB = 2048, 512, 256
SB_BYTES = 98


def f32_to_bf16_bytes(x):
    """Round-to-nearest-even fp32 -> bf16, return raw uint16 array."""
    u = x.astype(np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experts", type=int, default=1,
                    help="replicate weight matrix E times (v2 bandwidth runs)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    w = (rng.standard_normal((OUT, IN)) * 0.05).astype(np.float32)
    x = rng.standard_normal(IN).astype(np.float32)

    qq = xbf3.quantize(w, rotate=False)
    q = qq["q"].reshape(OUT, IN // SB, SB)          # int8 codes
    s = qq["scales"]                                 # (OUT, 2) fp32

    # pack: per row, per superblock: [2B bf16 scale][96B packed]
    s_bf = f32_to_bf16_bytes(s)                      # (OUT, 2) uint16
    packed = xbf3.pack3(q).reshape(OUT, IN // SB, 96)  # uint8
    row = np.empty((OUT, IN // SB, SB_BYTES), dtype=np.uint8)
    row[:, :, 0] = (s_bf & 0xFF).astype(np.uint8)
    row[:, :, 1] = (s_bf >> 8).astype(np.uint8)
    row[:, :, 2:] = packed
    wts = row.reshape(-1)
    if a.experts > 1:
        wts = np.tile(wts, a.experts)
    wts.tofile(os.path.join(a.out, "wts.bin"))
    x.tofile(os.path.join(a.out, "act.bin"))

    # reference: dequant with the SAME bf16-rounded scales the kernel consumes
    s_kernel = (s_bf.astype(np.uint32) << 16).view(np.float32)  # (OUT, 2)
    w_dq = (q.astype(np.float32) * s_kernel[:, :, None]).reshape(OUT, IN)
    y_ref = (w_dq.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    if a.experts > 1:
        y_ref = np.tile(y_ref, a.experts)
    y_ref.tofile(os.path.join(a.out, "out_ref.bin"))
    print(f"wts={wts.nbytes}B act={x.nbytes}B out_ref={y_ref.nbytes}B -> {a.out}/")


if __name__ == "__main__":
    main()
