#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
""".baa 容器写入器 v0（BAA_FORMAT.md v0.1）。

Ornith-1.5-35B-A3B（qwen35moe）→ models/ornith35_xbf3.baa：
- 专家张量（.experts.）→ XBF3（快速 absmax/4 路径，无网格搜索）；
  布局 = vnni2 kernel 消费格式：每行 K/256 个 superblock，sb = [2B bf16 scale][96B pack3(q+4)]
- 其余 BF16 原样；视觉塔（model.visual.*）跳过；MTP 保留（flags bit0）

名字规范化：strip "model.language_model." 前缀（embed_tokens/norm/lm_head/layers.*）。
"""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import ModelReader
import xbf3

BLK = 256  # superblock
SCHEME_BF16, SCHEME_XBF3 = 0, 2

def f32_to_bf16_u16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def quantize_xbf3_fast(w2d):
    """w2d (rows, K) fp32 → packed bytes (rows, K/256, 98)。快速 absmax/4 路径。"""
    rows, K = w2d.shape
    assert K % BLK == 0
    b = w2d.reshape(rows, K // BLK, BLK)
    absmax = np.maximum(np.abs(b).max(axis=-1), 1e-12)
    s = (absmax / 4.0).astype(np.float32)
    q = np.clip(np.round(b / s[..., None]), -4, 3).astype(np.int8)
    packed = xbf3.pack3(q + 4).reshape(rows, K // BLK, 96)   # offset-binary（vnni2）
    s_bf = f32_to_bf16_u16(s)
    out = np.empty((rows, K // BLK, 98), dtype=np.uint8)
    out[..., 0] = (s_bf & 0xFF).astype(np.uint8)
    out[..., 1] = (s_bf >> 8).astype(np.uint8)
    out[..., 2:] = packed
    return out.tobytes()

def norm_name(n):
    p = "model.language_model."
    if n.startswith(p):
        n = n[len(p):]
        n = {"model.embed_tokens.weight": "embed_tokens",
             "embed_tokens.weight": "embed_tokens",
             "norm.weight": "norm"}.get(n, n)
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/d/xdna_llm/models/Ornith-1.5-35B-A3B")
    ap.add_argument("--out", default="models/ornith35_xbf3.baa")
    ap.add_argument("--only", default=None, help="只写名字含该子串的张量（调试用）")
    a = ap.parse_args()

    mr = ModelReader(a.src)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    names_all = sorted(idx["weight_map"].keys())
    names = [n for n in names_all if not n.startswith("model.visual.")]
    if a.only:
        names = [n for n in names if a.only in n]
    print(f"{len(names)} tensors (of {len(names_all)}, vision dropped)")

    # config → meta
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    meta = {
        "hyperparameters": cfg["text_config"],
        "rope_sections": cfg["text_config"].get("rope_parameters", {}).get("mrope_section"),
        "chat_template_ref": "chat_template.jinja",
        "quant": {"scheme": "XBF3", "path": "absmax/4 fast (no grid search)",
                  "block": 256, "packing": "pack3(q+4) offset-binary, vnni2 layout"},
    }

    # 第一遍：算目录/数据布局（按规范化名排序——目录不变量）
    entries = []   # (name, scheme, shape, packed_size, payload_fn)
    data_off = 0
    t0 = time.time()
    for n in sorted(names, key=norm_name):
        dtype, shape, _ = mr.meta(n)
        is_expert = ".experts." in n and dtype == "BF16"
        scheme = SCHEME_XBF3 if is_expert else SCHEME_BF16
        if scheme == SCHEME_BF16:
            assert dtype == "BF16", (n, dtype)
            psz = int(np.prod(shape)) * 2
        else:
            rows = int(np.prod(shape[:-1])); K = shape[-1]
            assert K % BLK == 0, (n, shape)
            psz = rows * (K // BLK) * 98
        data_off = (data_off + 255) & ~255
        entries.append([norm_name(n), scheme, shape, psz, data_off, n])
        data_off += psz

    # 目录 + 名字池
    pool = bytearray()
    for e in entries:
        e.append(len(pool))
        pool += e[0].encode() + b"\0"
    dir_off = 128
    meta_json = json.dumps(meta).encode()
    meta_off = dir_off + 96 * len(entries) + len(pool)
    data_start = (meta_off + len(meta_json) + 255) & ~255

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        # header
        f.write(b"BAA1")
        f.write(struct.pack("<I", 1))
        f.write(b"qwen35moe\0".ljust(32, b"\0"))
        f.write(struct.pack("<QQQQQ", len(entries), dir_off, meta_off, len(meta_json), data_start))
        f.write(struct.pack("<Q", 1))  # flags: bit0 = MTP
        f.write(b"\0" * 40)
        assert f.tell() == 128
        # dir
        for (nn, scheme, shape, psz, doff, _, noff) in entries:
            sh = list(shape) + [1] * (4 - len(shape))
            f.write(struct.pack("<IBBBB", noff, scheme, 1, len(shape), 0))
            f.write(struct.pack("<4Q", *sh))
            f.write(struct.pack("<QQQ", doff, psz, 0))
            f.write(b"\0" * 32)
        f.write(bytes(pool))
        assert f.tell() == meta_off
        f.write(meta_json)
        f.write(b"\0" * (data_start - f.tell()))
        # data
        for i, (nn, scheme, shape, psz, doff, orig, noff) in enumerate(entries):
            assert f.tell() <= data_start + doff
            f.write(b"\0" * (data_start + doff - f.tell()))
            w = mr.read(orig)  # fp32
            if scheme == SCHEME_BF16:
                f.write(f32_to_bf16_u16(w).tobytes())
            else:
                w2d = w.reshape(-1, shape[-1])
                f.write(quantize_xbf3_fast(w2d))
            if i % 40 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(entries)}] {nn} ({psz/1e6:.0f}MB) elapsed {el:.0f}s", flush=True)
    print(f"done: {a.out} = {os.path.getsize(a.out)/1e9:.2f} GB in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
