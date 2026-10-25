#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""阶段 2/3 参考：单层（GDN + MoE）numpy 参考，支持多 token 递归（--ntok）。

输出：x_in.bin（ntok×2048）、y_gdn.bin / y_full.bin（ntok×2048，fp32）。
零初始状态；GDN 数学照 llama.cpp qwen35moe/delta-net decode（含 q×1/√128）。
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import ModelReader

H = 2048
KH, VH, HD = 16, 32, 128
KEY_DIM, VAL_DIM, QKV = 2048, 4096, 8192
NE, TOPK, MI = 256, 8, 512
EPS = 1e-6

def rmsnorm(x, w, eps=EPS):
    return x / np.sqrt((x ** 2).mean() + eps) * (1 + w)   # qwen3_5 零中心 RMSNorm

def sigmoid(x): return 1 / (1 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def softplus(x): return np.logaddexp(0, x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\xdna_llm\models\Ornith-1.5-35B-A3B")
    ap.add_argument("--out", default="testvec_layer0")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ntok", type=int, default=1)
    ap.add_argument("--x-embed", type=int, default=None, help="用 embed[token] 作输入（真实输入对拍 HF）")
    ap.add_argument("--skip-moe", action="store_true")
    ap.add_argument("--sim-xbf4", action="store_true", help="XBF4 档张量先量化再反解（模拟引擎噪声）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mr = ModelReader(a.src)
    L = a.layer
    P = f"model.language_model.layers.{L}."

    rng = np.random.default_rng(a.seed)
    xs = rng.standard_normal((a.ntok, H)).astype(np.float32)
    if a.x_embed is not None:
        xs[0] = mr.read("model.language_model.embed_tokens.weight", (a.x_embed,)).astype(np.float32)

    def T(n):
        w = mr.read(P + n)
        is_q4 = (n.endswith(".weight") and "norm" not in n and "conv1d" not in n
                 and (n.split(".")[0] in ("linear_attn", "self_attn")
                      or n.startswith("mlp.shared_expert.") or ".experts." in n))
        if a.sim_xbf4 and is_q4:
            from baa_writer_v1 import quant_pack4
            pk = np.frombuffer(quant_pack4(w.reshape(-1, w.shape[-1]).astype(np.float32)), dtype=np.uint8)
            rows, K = int(np.prod(w.shape[:-1])), w.shape[-1]
            pk = pk.reshape(rows, K//256, 130)
            sc = (pk[...,0].astype(np.uint16) | (pk[...,1].astype(np.uint16)<<8))
            sf = (sc.astype(np.uint32)<<16).view(np.float32)
            nib = pk[...,2:]
            q = (np.stack([nib & 0xF, nib >> 4], axis=-1).reshape(rows, K//256, 256)).astype(np.int16) - 8
            w = (q * sf[...,None]).reshape(w.shape)
        return w

    Wqkv = T("linear_attn.in_proj_qkv.weight")
    Wz = T("linear_attn.in_proj_z.weight")
    Wb = T("linear_attn.in_proj_b.weight")
    Wa = T("linear_attn.in_proj_a.weight")
    dt_b = T("linear_attn.dt_bias")
    A_log = T("linear_attn.A_log")
    cw = T("linear_attn.conv1d.weight")[:, 0, :]   # [8192,4]
    ln1 = T("input_layernorm.weight")
    nw = T("linear_attn.norm.weight")
    Wo = T("linear_attn.out_proj.weight")

    conv = np.zeros((3, QKV), dtype=np.float32)    # 最近 3 个 token 的 qkv（旧→新）
    ssm = np.zeros((VH, HD, HD), dtype=np.float32) # [j][i(v), m(k)]

    y_gdn_all = np.zeros((a.ntok, H), dtype=np.float32)
    for t in range(a.ntok):
        x = xs[t]
        xn = rmsnorm(x, ln1)
        qkv = Wqkv @ xn
        z = Wz @ xn
        beta = sigmoid(Wb @ xn)
        alpha = Wa @ xn
        g = -softplus(alpha + dt_b) * np.exp(A_log)
        win = np.vstack([conv, qkv[None, :]])       # [4, 8192]
        qc = silu((win * cw.T).sum(axis=0))           # depthwise 因果（t 对应窗口偏移 3）
        conv = win[1:]
        q, k, v = qc[:KEY_DIM], qc[KEY_DIM:KEY_DIM*2], qc[KEY_DIM*2:]

        def l2n(vec):
            vec = vec.reshape(-1, HD)
            return (vec / np.sqrt((vec ** 2).mean(axis=-1, keepdims=True) + EPS / HD) / np.sqrt(HD)).reshape(-1)
        q, k = l2n(q), l2n(k)
        q = q / np.sqrt(HD)

        o = np.zeros(VAL_DIM, dtype=np.float32)
        for j in range(VH):
            kj = k[(j // (VH // KH)) * HD:(j // (VH // KH) + 1) * HD]
            qj = q[(j // (VH // KH)) * HD:(j // (VH // KH) + 1) * HD]
            vj = v[j * HD:(j + 1) * HD]
            s = ssm[j]
            s *= np.exp(g[j])
            sk = s @ kj                       # [i] = Σ_m s[i,m]k[m]
            d = (vj - sk) * beta[j]
            s += np.outer(d, kj)              # s[i,m] += d[i]*k[m]
            o[j * HD:(j + 1) * HD] = s @ qj
        og = o.reshape(VH, HD)
        og = og / np.sqrt((og ** 2).mean(axis=-1, keepdims=True) + EPS) * nw
        og = (og * silu(z.reshape(VH, HD))).reshape(-1)
        y_gdn_all[t] = x + Wo @ og
        if os.environ.get("RL_DBG"):
            np.savez("rl_dbg.npz", xn=xn, qkv=qkv, qc=qc, o=o, og=og, y=x + Wo @ og, z=z, beta=beta, g=g)

    # MoE（对最后一个 token）
    x2 = rmsnorm(y_gdn_all[-1], T("post_attention_layernorm.weight"))
    logits = T("mlp.gate.weight") @ x2
    wexp = np.exp(logits - logits.max()); wexp /= wexp.sum()
    top = np.argsort(-wexp)[:TOPK]
    tw = wexp[top]; tw /= tw.sum()
    gu_all = T("mlp.experts.gate_up_proj")
    dn_all = T("mlp.experts.down_proj")
    acc = np.zeros(H, dtype=np.float32)
    for tt, e in enumerate(top):
        gu = gu_all[e] @ x2
        h = silu(gu[:MI]) * gu[MI:]
        acc += tw[tt] * (dn_all[e] @ h)
    sg = T("mlp.shared_expert.gate_proj.weight") @ x2
    su = T("mlp.shared_expert.up_proj.weight") @ x2
    sh = T("mlp.shared_expert.down_proj.weight") @ (silu(sg) * su)
    g0 = sigmoid(T("mlp.shared_expert_gate.weight") @ x2)
    y_full = y_gdn_all[-1] + acc + g0[0] * sh

    xs.astype(np.float32).tofile(os.path.join(a.out, "x_in.bin"))
    y_gdn_all.astype(np.float32).tofile(os.path.join(a.out, "y_gdn.bin"))
    y_full.astype(np.float32).tofile(os.path.join(a.out, "y_full.bin"))
    np.save(os.path.join(a.out, "topk.npy"), top)
    print(f"layer {L} ntok={a.ntok}: top8={top.tolist()} |y_gdn[-1]|max={np.abs(y_gdn_all[-1]).max():.4f}")

if __name__ == "__main__":
    main()
