#!/usr/bin/env python3
# gen_dbg.py — dequant_dbg 输入生成 + 期望输出计算 + 比对
# 用法: python gen_dbg.py --pattern w5        → 生成 dbg_in.bin
#       python gen_dbg.py --check w5          → 比对 dbg.st{1,2,3}.bin
import argparse, os
import numpy as np

LT_K, LT_N = 128, 64   # 大瓦片
def f32_to_bf16_u16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def build(pattern):
    if pattern.startswith("w"):  # wN[:zM]：恒值
        parts = pattern[1:].split("z")
        w = np.full((LT_K, LT_N), int(parts[0]), dtype=np.uint8)
        z = np.full(LT_N, int(parts[1]) if len(parts) > 1 else 0, dtype=np.uint8)
        s = np.ones(LT_N, dtype=np.float32)
    elif pattern == "kmod":      # 位置编码：w[k,n] = k%16
        w = (np.arange(LT_K)[:, None] % 16).astype(np.uint8) * np.ones((LT_K, LT_N), dtype=np.uint8)
        z = np.zeros(LT_N, dtype=np.uint8); s = np.ones(LT_N, dtype=np.float32)
    elif pattern == "nmod":
        w = (np.arange(LT_N)[None, :] % 16).astype(np.uint8) * np.ones((LT_K, LT_N), dtype=np.uint8)
        z = np.zeros(LT_N, dtype=np.uint8); s = np.ones(LT_N, dtype=np.float32)
    # 打包（与 gen_data 相同：8x8 小瓦片行主序，lo|hi<<4）
    B = np.zeros(4352, dtype=np.uint8)
    blk = w.reshape(16, 8, 8, 8).transpose(0, 2, 1, 3)   # [str,stc,i,j]
    pairs = blk.reshape(16, 8, 8, 4, 2)
    B[:4096] = (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1)
    B[4096:4224] = f32_to_bf16_u16(s).view(np.uint8)
    B[4224:4352] = np.concatenate([z.reshape(8, 8), z.reshape(8, 8)], axis=1).reshape(-1)
    # 期望（kernel 车道序 = 小瓦片行主序、瓦片内行主序 lo 先）
    st1 = np.stack([pairs[..., 0], pairs[..., 1]], axis=-1).reshape(-1)          # unpack 后
    return B, w, z, s, st1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="w5")
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    B, w, z, s, st1 = build(a.pattern)
    if not a.check:
        B.tofile("dbg_in.bin")
        np.save("dbg_expect_w.npy", w); np.save("dbg_expect_z.npy", z); np.save("dbg_expect_s.npy", s)
        np.save("dbg_expect_st1.npy", st1.astype(np.uint8))
        print(f"dbg_in.bin written (pattern={a.pattern})")
        return
    # 比对
    got1 = np.fromfile("dbg.st1.bin", dtype=np.uint8)
    got2 = np.fromfile("dbg.st2.bin", dtype=np.int8)
    got3 = bf16_to_f32(np.fromfile("dbg.st3.bin", dtype=np.uint16))
    exp_w = np.load("dbg_expect_w.npy"); exp_z = np.load("dbg_expect_z.npy"); exp_s = np.load("dbg_expect_s.npy")
    exp1 = np.load("dbg_expect_st1.npy")
    print("== st1 (unpack):")
    neq = np.nonzero(got1 != exp1)[0]
    print(f"   mismatch lanes: {len(neq)}", neq[:20])
    if len(neq):
        for l in neq[:8]:
            print(f"   lane {l}: got {got1[l]} exp {exp1[l]}  (tile {l//64}, row {(l%64)//8}, col {l%8})")
    # st2 期望 = st1 − zp；车道序：tile t=(i,j) lane l → zp[j*8 + l%8]
    t_idx = np.arange(128 * 64) // 64
    l_idx = np.arange(128 * 64) % 64
    zp_lane = exp_z[(t_idx % 8) * 8 + (l_idx % 8)]
    exp2 = (exp1.astype(np.int16) - zp_lane).astype(np.int8)
    neq2 = np.nonzero(got2 != exp2)[0]
    print("== st2 (sub zp, 假设 lane l→z[l%8]):", len(neq2), "mismatch", neq2[:10])
    # st3 期望 = bf16(st2) × s
    exp3 = (exp2.reshape(128, 64).astype(np.float32) * 1.0).reshape(-1)
    d = np.abs(got3 - exp3)
    neq3 = np.nonzero(d > 0.01)[0]
    print("== st3 (cvt+scale):", len(neq3), "mismatch", neq3[:10])
    if len(neq3):
        for l in neq3[:8]:
            print(f"   lane {l}: got {got3[l]} exp {exp3[l]} (tile {l//64}, row {(l%64)//8}, col {l%8})")

if __name__ == "__main__":
    main()
