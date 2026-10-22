#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU XBF3 GEMV 测试数据（M3'-1）。

输出（--out dir）：
  wts_lut.bin  : 规范 pack3 打包（每 sb: [2B bf16 scale][96B 24-bit 组]），E 个专家拼接
  wts_bp.bin   : pack3_bp 位平面打包（每 sb: [scale][P0 32B][P1][P2]）
  x.bin        : fp32 in 向量
  xp_lut.bin   : LUT 变体的相位重排 fp32——xp[k][sb][64] = x[sb*256 + 8*(16b+j) + k]
                 其中 lane = 16b+j（b=block 内组号, j=lane）… 见 gemv_avx512.cpp
  xp_bp.bin    : 位平面变体：xp[p][k][sb][lane32] = x[sb*256 + 8*lane + k] * {1,2,-4}[p]
  y_ref.bin    : 参考 y = dequant(W) @ x（fp64 累加，每专家独立）
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "quant"))
import xbf3

SB = 256
ROW_B = 98  # 2B scale + 96B packed

def f32_to_bf16_u16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_cpu")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--n-in", type=int, default=512, help="reduction dim (down_proj: 512)")
    ap.add_argument("--n-out", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    OUT, IN, E = a.n_out, a.n_in, a.experts
    assert IN % SB == 0

    rng = np.random.default_rng(a.seed)
    # 每专家独立随机（防 L3 缓存假象）；快速量化路径（跳过网格搜索，带宽测试不需要最优 scale）
    w = (rng.standard_normal((E, OUT, IN)) * 0.05).astype(np.float32)
    x = rng.standard_normal(IN).astype(np.float32)

    absmax = np.maximum(np.abs(w).reshape(E, OUT, IN // SB, SB).max(axis=-1), 1e-12)
    s_e = absmax / 4.0                                     # (E, OUT, IN//SB)
    q = np.clip(np.round(w.reshape(E, OUT, IN // SB, SB) / s_e[..., None]), -4, 3).astype(np.int8)
    q = q.reshape(E, OUT, IN)
    s = s_e.reshape(E, OUT, IN // SB)
    s_bf = f32_to_bf16_u16(s)

    # 规范 pack3 / 位平面 pack3_bp（E 个专家各自独立）
    packed_lut = xbf3.pack3(q).reshape(E, OUT, IN // SB, 96)
    # VNNI: offset-binary field = q+4
    packed_vnni = xbf3.pack3(q + 4).reshape(E, OUT, IN // SB, 96)
    packed_bp = xbf3.pack3_bp(q).reshape(E, OUT, IN // SB, 96)

    for packed, name in ((packed_lut, "wts_lut.bin"), (packed_bp, "wts_bp.bin"), (packed_vnni, "wts_vnni.bin")):
        row = np.empty((E, OUT, IN // SB, ROW_B), dtype=np.uint8)
        row[:, :, :, 0] = (s_bf & 0xFF).astype(np.uint8)
        row[:, :, :, 1] = (s_bf >> 8).astype(np.uint8)
        row[:, :, :, 2:] = packed
        row.reshape(-1).tofile(os.path.join(a.out, name))

    x.astype(np.float32).tofile(os.path.join(a.out, "x.bin"))

    # LUT 变体相位平面: 每 sb 2 个 16-lane 块；xp[k][sb][16b+j] = x[sb*256 + 8*(16b+j) + k]
    xp_lut = np.zeros((8, IN // SB, 2 * 16), dtype=np.float32)
    for k in range(8):
        for sb in range(IN // SB):
            for blk in range(2):
                for j in range(16):
                    xp_lut[k, sb, blk * 16 + j] = x[sb * SB + 8 * (16 * blk + j) + k]
    xp_lut.tofile(os.path.join(a.out, "xp_lut.bin"))

    # 位平面变体: xp[p][k][sb][lane] = x[sb*256 + 8*lane + k] * {1,2,-4}[p]
    pw = np.array([1.0, 2.0, -4.0], dtype=np.float32)
    xp_bp = np.zeros((3, 8, IN // SB, 32), dtype=np.float32)
    for p in range(3):
        for k in range(8):
            for sb in range(IN // SB):
                xp_bp[p, k, sb, :] = x[sb * SB + 8 * np.arange(32) + k] * pw[p]
    xp_bp.tofile(os.path.join(a.out, "xp_bp.bin"))

    # VNNI 数据：xq = round(x/sx)，sx = max|x|/127；布局 [sb][b][pg][lane][t]
    sx = float(np.abs(x).max() / 127.0)
    xq = np.round(x / sx).astype(np.int8)                       # (IN,)
    xq_layout = np.zeros((IN // SB, 2, 2, 16, 4), dtype=np.int8)
    for sb in range(IN // SB):
        for b in range(2):
            for pg in range(2):
                for j in range(16):
                    for t in range(4):
                        xq_layout[sb, b, pg, j, t] = xq[sb * SB + 8 * (16 * b + j) + 4 * pg + t]
    xq_layout.tofile(os.path.join(a.out, "xq_vnni.bin"))
    xq.tofile(os.path.join(a.out, "xq_plain.bin"))  # vnni2：连续自然序（multishift 免重排）
    sumxq = xq.reshape(IN // SB, SB).sum(axis=1).astype(np.float32)  # 每 sb
    with open(os.path.join(a.out, "vnni_meta.bin"), "wb") as f:
        f.write(np.array([sx], dtype=np.float32).tobytes())
        f.write(sumxq.tobytes())
    # 参考（vnni）：y = Σ_sb (Σ field·xq − 4·Σxq) · sx · scale
    s_kernel = (s_bf.astype(np.uint32) << 16).view(np.float32)
    fld = (q + 4).reshape(E, OUT, IN // SB, SB).astype(np.float64)
    xq_sb = xq.reshape(IN // SB, SB).astype(np.float64)
    yv = (np.einsum("eosb,sb->eos", fld, xq_sb)
          - 4.0 * sumxq.astype(np.float64)[None, None, :])          * (sx * s_kernel.astype(np.float64))
    y_vnni = yv.sum(axis=-1)
    y_vnni.astype(np.float32).tofile(os.path.join(a.out, "y_ref_vnni.bin"))

    # 参考：fp64，每专家独立；scale 用 bf16 舍入值（与 kernel 消费的一致）
    w_dq = (q.reshape(E, OUT, IN // SB, SB).astype(np.float32) * s_kernel[..., None]).reshape(E, OUT, IN)
    y = w_dq.astype(np.float64) @ x.astype(np.float64)
    y.astype(np.float32).tofile(os.path.join(a.out, "y_ref.bin"))
    print(f"E={E} OUT={OUT} IN={IN} -> {a.out}/  wts_lut={os.path.getsize(os.path.join(a.out,'wts_lut.bin'))}B")

if __name__ == "__main__":
    main()
