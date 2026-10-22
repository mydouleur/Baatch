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
ROW_B = 196


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
    ap.add_argument("--cols", type=int, default=1,
                    help="v3: split rows across N columns, wts laid out column-major")
    ap.add_argument("--bitplane", action="store_true",
                    help="v4: XBF3-BP 位平面打包 + fp32 相位平面激活")
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
    if a.bitplane:
        packed = xbf3.pack3_bp(q).reshape(OUT, IN // SB, 96)
    else:
        packed = xbf3.pack3(q).reshape(OUT, IN // SB, 96)  # uint8
    row = np.empty((OUT, IN // SB, SB_BYTES), dtype=np.uint8)
    row[:, :, 0] = (s_bf & 0xFF).astype(np.uint8)
    row[:, :, 1] = (s_bf >> 8).astype(np.uint8)
    row[:, :, 2:] = packed
    wts = row.reshape(-1)                    # OUT*196
    if a.experts > 1:
        wts = np.tile(wts, a.experts)        # E * OUT * 196
    if a.cols > 1:
        rpc = OUT // a.cols
        wts = wts.reshape(a.experts, a.cols, rpc * ROW_B)  # [E][col][rows]
        wts = np.transpose(wts, (1, 0, 2)).reshape(-1)     # [col][E][rows]
        # v3：每列 slab = [act 块(chunk_rows*196B, 前 2048B 是 fp32 x)][列权重]
        chunk_bytes = 512 * ROW_B   # = L2 块（act 独占一个 L2 块）
        actpad = np.zeros(chunk_bytes, dtype=np.uint8)
        actpad[: IN * 4] = x.view(np.uint8)
        col_bytes = wts.nbytes // a.cols
        slabs = b"".join(
            actpad.tobytes() + wts[c * col_bytes:(c + 1) * col_bytes].tobytes()
            for c in range(a.cols))
        wts = np.frombuffer(slabs, dtype=np.uint8).copy()
    wts.tofile(os.path.join(a.out, "wts.bin"))
    if a.bitplane:
        # xp[k][sb][i] = x[sb*256 + 8i + k]，fp32，8*2*32 = 512 floats
        xp = np.zeros((8, IN // SB, 32), dtype=np.float32)
        for k in range(8):
            for sb in range(IN // SB):
                xp[k, sb, :] = x[sb * SB + 8 * np.arange(32) + k]
        xa = np.tile(xp.reshape(-1), a.cols) if a.cols > 1 else xp.reshape(-1)
    else:
        xa = np.tile(x, a.cols) if a.cols > 1 else x
    xa.tofile(os.path.join(a.out, "act.bin"))

    # reference: dequant with the SAME bf16-rounded scales the kernel consumes
    s_kernel = (s_bf.astype(np.uint32) << 16).view(np.float32)  # (OUT, 2)
    w_dq = (q.astype(np.float32) * s_kernel[:, :, None]).reshape(OUT, IN)
    y_ref = (w_dq.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    if a.experts > 1:
        y_ref = np.tile(y_ref, a.experts)
    y_ref.tofile(os.path.join(a.out, "out_ref.bin"))
    print(f"wts={wts.nbytes}B act={xa.nbytes}B out_ref={y_ref.nbytes}B -> {a.out}/")


if __name__ == "__main__":
    main()
