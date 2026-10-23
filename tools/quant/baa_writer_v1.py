#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
""".baa v1 写入器（XBF24 配比，DESIGN.md §10.2 / BAA_FORMAT.md v0.1+专家粒度不变量）。

scheme：0=BF16 1=INT8 2=XBF3 3=XBF4 4=XBF2 5=F32
配比（朴素对称量化，不做 imatrix——质量版重打包时引擎零改动）：
- 路由专家：层 10–29 XBF2，其余 XBF4；**按专家拆目录项**（...experts.gate_up_proj.{e}）
- 密集路径（attn/shexp/embed/lm_head/mtp 全权重）→ XBF4
- 路由器（mlp.gate / shared_expert_gate）、SSM 参数（A_log/dt_bias/conv1d）、全部 norm → F32
- MTP 专家 XBF4；MTP 路由器 BF16（规范 §10.2 表注）
布局：XBF4 sb=[2B bf16 scale][128B nibble lo|hi<<4]；XBF2 sb=[2B scale][64B 4值/字节]
"""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import ModelReader

BLK = 256
SCHEME_BF16, SCHEME_XBF3, SCHEME_XBF4, SCHEME_XBF2, SCHEME_F32 = 0, 2, 3, 4, 5

def f32_to_bf16_u16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def quant_pack4(w2d):
    """XBF4：rows×K fp32 → (rows, K/256, 130) bytes。朴素对称 absmax/8。"""
    rows, K = w2d.shape
    b = w2d.reshape(rows, K // BLK, BLK)
    s = (np.maximum(np.abs(b).max(axis=-1), 1e-12) / 8.0).astype(np.float32)
    q = np.clip(np.round(b / s[..., None]), -8, 7).astype(np.int16) + 8   # field 0..15
    u = q.reshape(rows, K // BLK, 128, 2).astype(np.uint8)
    nib = u[..., 0] | (u[..., 1] << 4)
    s_bf = f32_to_bf16_u16(s)
    out = np.empty((rows, K // BLK, 130), dtype=np.uint8)
    out[..., 0] = (s_bf & 0xFF).astype(np.uint8)
    out[..., 1] = (s_bf >> 8).astype(np.uint8)
    out[..., 2:] = nib
    return out.tobytes()

def quant_pack2(w2d):
    """XBF2：rows×K fp32 → (rows, K/256, 66) bytes。朴素对称 absmax/2。"""
    rows, K = w2d.shape
    b = w2d.reshape(rows, K // BLK, BLK)
    s = (np.maximum(np.abs(b).max(axis=-1), 1e-12) / 2.0).astype(np.float32)
    q = np.clip(np.round(b / s[..., None]), -2, 1).astype(np.int16) + 2    # field 0..3
    u = q.reshape(rows, K // BLK, 64, 4).astype(np.uint8)
    pk = u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)
    s_bf = f32_to_bf16_u16(s)
    out = np.empty((rows, K // BLK, 66), dtype=np.uint8)
    out[..., 0] = (s_bf & 0xFF).astype(np.uint8)
    out[..., 1] = (s_bf >> 8).astype(np.uint8)
    out[..., 2:] = pk
    return out.tobytes()

def norm_name(n):
    p = "model.language_model."
    if n.startswith(p):
        n = n[len(p):]
        n = {"embed_tokens.weight": "embed_tokens", "norm.weight": "norm"}.get(n, n)
    return n

def scheme_of(n, layer):
    """按 v1 配比定 scheme。n = 规范化名。"""
    if ".experts." in n:
        if n.startswith("mtp."): return SCHEME_XBF4
        return SCHEME_XBF2 if (layer is not None and 10 <= layer <= 29) else SCHEME_XBF4
    base = n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n
    leaf = n.split(".")[-1]
    stem = n.rsplit(".weight", 1)[0].split(".")[-1]
    if leaf != "weight":      # A_log / dt_bias 等无 .weight 后缀
        return SCHEME_F32
    if stem in ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm",
                "norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        return SCHEME_F32
    if stem in ("A_log", "dt_bias", "conv1d"):
        return SCHEME_F32
    if stem in ("gate", "shared_expert_gate"):   # 路由器
        return SCHEME_BF16 if n.startswith("mtp.") else SCHEME_F32
    return SCHEME_XBF4

def pack4_pretiled(w2d):
    """XBF4 + TileFuse 预平铺（gemv_w4a16/test.cpp 布局）：大瓦片 128(K)x64(N)
    = 4096B 权重 + 128B bf16 scale + 128B zp（zp=8 常数，对称量化）；
    大瓦片 col_mod(=lt_col%8) → lt_col → ltr 主序；小瓦片 8x8 行主序 lo|hi<<4。
    w2d: [N, K] fp32（行=输出）。返回 bytes。"""
    N, K = w2d.shape
    LT_K, LT_N, NCOL = 128, 64, 8
    # 量化（与 quant_pack4 同参：absmax/8，clip [-8,7]，field=q+8）
    b = w2d.reshape(N, K // 256, 256)
    s = (np.maximum(np.abs(b).max(axis=-1), 1e-12) / 8.0).astype(np.float32)   # [N, K/256]
    q = np.clip(np.round(b / s[..., None]), -8, 7).astype(np.int16) + 8
    Wf = q.reshape(N, K).astype(np.uint8)                                     # field 0..15
    s_bf = f32_to_bf16_u16(s)                                                 # [N, K/256] u16
    nlt_col = N // LT_N
    nlt = (K // LT_K) * nlt_col
    B = np.zeros((nlt, 4352), dtype=np.uint8)
    idx = 0
    for cm in range(NCOL):
        for lt_col in range(cm, nlt_col, NCOL):
            for ltr in range(K // LT_K):
                blk = Wf[lt_col*LT_N:(lt_col+1)*LT_N, ltr*LT_K:(ltr+1)*LT_K].T.copy()  # [k,n]
                blk = blk.reshape(16, 8, 8, 8).transpose(0, 2, 1, 3)          # [str,stc,i,j]
                pairs = blk.reshape(16, 8, 8, 4, 2)
                B[idx, :4096] = (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1)
                # scale：该大瓦片 k 范围 = 我们 256-sb 的 kt/2
                sb = (ltr * LT_K) // 256
                B[idx, 4096:4224] = np.ascontiguousarray(s_bf[lt_col*LT_N:(lt_col+1)*LT_N, sb]).view(np.uint8)
                B[idx, 4224:4352] = 8   # zp = 8（对称量化 field−8）
                idx += 1
    return B.reshape(-1).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\xdna_llm\models\Ornith-1.5-35B-A3B")
    ap.add_argument("--out", default="models/ornith35_xbf24.baa")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    mr = ModelReader(a.src)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    names = [n for n in sorted(idx["weight_map"].keys()) if not n.startswith("model.visual.")]
    if a.only:
        names = [n for n in names if a.only in n]

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    meta = {
        "hyperparameters": cfg["text_config"],
        "rope_sections": cfg["text_config"].get("rope_parameters", {}).get("mrope_section"),
        "chat_template_ref": "chat_template.jinja",
        "quant": {"scheme": "XBF24 v1 mix", "path": "naive symmetric absmax (no imatrix)",
                  "xbf4": "absmax/8, nibble lo|hi<<4", "xbf2": "absmax/2, 4 vals/byte",
                  "mix": "experts L10-29 XBF2 else XBF4; dense XBF4; router/SSM/norm F32"},
    }

    # 展开为目录项列表（专家 3D 拆专家粒度）
    entries = []   # [name, scheme, shape, orig, expert_idx or None]
    for n in names:
        nn = norm_name(n)
        dtype, shape, _ = mr.meta(n)
        assert dtype == "BF16", (n, dtype)
        layer = None
        if nn.startswith("layers."):
            layer = int(nn.split(".")[1])
        sch = scheme_of(nn, layer)
        if ".experts." in nn and len(shape) == 3:      # 打包 3D 专家张量 → 拆
            E = shape[0]
            for e in range(E):
                entries.append([f"{nn}.{e}", sch, (shape[1], shape[2]), n, e])
        elif ".experts." in nn and len(shape) == 2:     # mtp 单专家 2D（名字自带 .N）
            entries.append([nn, sch, shape, n, None])
        else:
            entries.append([nn, sch, shape, n, None])
    entries.sort(key=lambda e: e[0])   # 目录按 name 排序（不变量）
    print(f"{len(entries)} dir entries (from {len(names)} tensors)")

    # 布局：仅 (N,K)=(8192,2048) 密集张量（in_proj_qkv / q_proj，每 token 40 次调用
    # 同形状零切换）→ TileFuse 预平铺 + NPU；其余 XBF4/XBF2 行主序（CPU vnni4/vnni2
    # 快速路径不变）。flags bit0=预平铺。（保留字节 offset7 复用，已向主线报告）
    # 取证：xclbin 切换 2.6ms/次（bench_switch.exe），多形状交替把 decode 拖到 600ms/token。
    data_off = 0
    for e in entries:
        sch = e[1]; shape = e[2]
        nel = int(np.prod(shape))
        is_expert = ".experts." in e[0]
        pretiled = (sch == SCHEME_XBF4 and not is_expert and len(shape) == 2
                    and tuple(shape) == (8192, 2048))
        if sch == SCHEME_BF16: psz = nel * 2
        elif sch == SCHEME_F32: psz = nel * 4
        elif sch == SCHEME_XBF4 and pretiled: psz = (shape[1] // 128) * (shape[0] // 64) * 4352
        elif sch == SCHEME_XBF4: psz = (nel // BLK) * 130
        else: psz = (nel // BLK) * 66
        data_off = (data_off + 255) & ~255
        e += [psz, data_off, 1 if pretiled else 0]
        data_off += psz

    pool = bytearray()
    for e in entries:
        e.append(len(pool)); pool += e[0].encode() + b"\0"
    dir_off = 128
    meta_json = json.dumps(meta).encode()
    meta_off = dir_off + 96 * len(entries) + len(pool)
    data_start = (meta_off + len(meta_json) + 255) & ~255

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t0 = time.time()
    with open(a.out, "wb") as f:
        f.write(b"BAA1"); f.write(struct.pack("<I", 1))
        f.write(b"qwen35moe\0".ljust(32, b"\0"))
        f.write(struct.pack("<QQQQQ", len(entries), dir_off, meta_off, len(meta_json), data_start))
        f.write(struct.pack("<Q", 1))
        f.write(b"\0" * 40)
        for (nn, sch, shape, orig, ei, psz, doff, flags, noff) in entries:
            sh = list(shape) + [1] * (4 - len(shape))
            f.write(struct.pack("<IBBBB", noff, sch, 1, len(shape), flags))
            f.write(struct.pack("<4Q", *sh))
            f.write(struct.pack("<QQQ", doff, psz, 0))
            f.write(b"\0" * 32)
        f.write(bytes(pool))
        f.write(meta_json)
        f.write(b"\0" * (data_start - f.tell()))
        for i, (nn, sch, shape, orig, ei, psz, doff, flags, noff) in enumerate(entries):
            f.write(b"\0" * (data_start + doff - f.tell()))
            w = mr.read(orig) if ei is None else mr.read(orig, (ei,))
            if sch == SCHEME_BF16:
                f.write(f32_to_bf16_u16(w).tobytes())
            elif sch == SCHEME_F32:
                f.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())
            elif sch == SCHEME_XBF4 and flags:
                try:
                    f.write(pack4_pretiled(w.astype(np.float32)))
                except Exception as ex:
                    print("PRETILE FAIL on", nn, "shape", w.shape, ":", ex, flush=True)
                    raise
            else:
                f.write((quant_pack4 if sch == SCHEME_XBF4 else quant_pack2)(
                    w.reshape(-1, shape[-1]).astype(np.float32)))
            if i % 2000 == 0:
                print(f"[{i+1}/{len(entries)}] {nn} elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"done: {a.out} = {os.path.getsize(a.out)/1e9:.2f} GB in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
