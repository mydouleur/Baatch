#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""全注意力层（layer 3 型）多 token 参考：attn + MoE。

输出：x_in.bin（ntok×2048）、y_attn.bin / y_full.bin（fp32）。
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import ModelReader

H = 2048
QH, KVH, AHD = 16, 2, 256
ROPE_DIM, THETA = 64, 1e7
NE, TOPK, MI = 256, 8, 512
EPS = 1e-6

def rmsnorm(x, w, eps=EPS):
    return x / np.sqrt((x ** 2).mean() + eps) * (1 + w)   # qwen3_5 零中心 RMSNorm

def sigmoid(x): return 1 / (1 + np.exp(-x))
def silu(x): return x * sigmoid(x)

def rope(vec, pos, nheads):
    v = vec.reshape(nheads, AHD).copy()
    for h in range(nheads):
        for i in range(ROPE_DIM // 2):
            ang = pos / (THETA ** (2.0 * i / ROPE_DIM))
            c, s = np.cos(ang), np.sin(ang)
            x0, x1 = v[h, 2 * i], v[h, 2 * i + 1]
            v[h, 2 * i] = x0 * c - x1 * s
            v[h, 2 * i + 1] = x0 * s + x1 * c
    return v.reshape(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\xdna_llm\models\Ornith-1.5-35B-A3B")
    ap.add_argument("--out", default="testvec_layer3")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ntok", type=int, default=4)
    ap.add_argument("--sim-xbf4", action="store_true", help="XBF4 档张量先量化再反解")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mr = ModelReader(a.src)
    L = a.layer
    P = f"model.language_model.layers.{L}."

    rng = np.random.default_rng(a.seed)
    xs = rng.standard_normal((a.ntok, H)).astype(np.float32)

    def T(n):
        w = mr.read(P + n)
        is_q4 = (n.endswith(".weight") and
                 (n.startswith("self_attn.") or n.startswith("mlp.shared_expert.")
                  or n.startswith("mlp.experts.")))
        if a.sim_xbf4 and is_q4:
            from baa_writer_v1 import quant_pack4
            pk = np.frombuffer(quant_pack4(w.reshape(-1, w.shape[-1]).astype(np.float32)), dtype=np.uint8)
            rows, K = int(np.prod(w.shape[:-1])), w.shape[-1]
            pk = pk.reshape(rows, K//256, 130)
            sc = pk[..., 0].astype(np.uint16) | (pk[..., 1].astype(np.uint16) << 8)
            sf = (sc.astype(np.uint32) << 16).view(np.float32)
            nib = pk[..., 2:]
            q = np.stack([nib & 0xF, nib >> 4], axis=-1).reshape(rows, K//256, 256).astype(np.int16) - 8
            w = (q * sf[..., None]).reshape(w.shape)
        return w

    Wq, Wk, Wv, Wo = T("self_attn.q_proj.weight"), T("self_attn.k_proj.weight"), T("self_attn.v_proj.weight"), T("self_attn.o_proj.weight")
    ln1 = T("input_layernorm.weight")
    qnw, knw = T("self_attn.q_norm.weight"), T("self_attn.k_norm.weight")

    Ks = np.zeros((a.ntok, KVH * AHD), dtype=np.float32)
    Vs = np.zeros((a.ntok, KVH * AHD), dtype=np.float32)
    y_attn = np.zeros((a.ntok, H), dtype=np.float32)
    for t in range(a.ntok):
        x = xs[t]
        xn = rmsnorm(x, ln1)
        qg = Wq @ xn                      # [16, 512]：前 256 q，后 256 gate
        kk = Wk @ xn
        vv = Wv @ xn
        qg = qg.reshape(QH, 2 * AHD)
        q = qg[:, :AHD].copy()
        gate = qg[:, AHD:]
        q = q.reshape(QH, AHD) / np.sqrt((q.reshape(QH, AHD) ** 2).mean(axis=-1, keepdims=True) + EPS) * (1 + qnw)
        kk = kk.reshape(KVH, AHD) / np.sqrt((kk.reshape(KVH, AHD) ** 2).mean(axis=-1, keepdims=True) + EPS) * (1 + knw)
        q = rope(q.reshape(-1), t, QH).reshape(QH, AHD)
        kk = rope(kk.reshape(-1), t, KVH).reshape(KVH, AHD)
        Ks[t], Vs[t] = kk.reshape(-1), vv
        out = np.zeros((QH, AHD), dtype=np.float32)
        for h in range(QH):
            kvh = h // (QH // KVH)
            sc = (Ks[:t + 1, kvh * AHD:(kvh + 1) * AHD] @ q[h]) / np.sqrt(AHD)
            w = np.exp(sc - sc.max()); w /= w.sum()
            out[h] = w @ Vs[:t + 1, kvh * AHD:(kvh + 1) * AHD]
            out[h] *= sigmoid(gate[h])
        y_attn[t] = x + Wo @ out.reshape(-1)

    x2 = rmsnorm(y_attn[-1], T("post_attention_layernorm.weight"))
    logits = T("mlp.gate.weight") @ x2
    wexp = np.exp(logits - logits.max()); wexp /= wexp.sum()
    top = np.argsort(-wexp)[:TOPK]
    tw = wexp[top]; tw /= tw.sum()
    gu_all = T("mlp.experts.gate_up_proj")
    dn_all = T("mlp.experts.down_proj")
    acc = np.zeros(H, dtype=np.float32)
    for tt, e in enumerate(top):
        gu = gu_all[e] @ x2
        acc += tw[tt] * (dn_all[e] @ (silu(gu[:MI]) * gu[MI:]))
    sg = T("mlp.shared_expert.gate_proj.weight") @ x2
    su = T("mlp.shared_expert.up_proj.weight") @ x2
    sh = T("mlp.shared_expert.down_proj.weight") @ (silu(sg) * su)
    g0 = sigmoid(T("mlp.shared_expert_gate.weight") @ x2)
    y_full = y_attn[-1] + acc + g0[0] * sh

    xs.astype(np.float32).tofile(os.path.join(a.out, "x_in.bin"))
    y_attn.astype(np.float32).tofile(os.path.join(a.out, "y_attn.bin"))
    y_full.astype(np.float32).tofile(os.path.join(a.out, "y_full.bin"))
    print(f"attn layer {L} ntok={a.ntok} done")

if __name__ == "__main__":
    main()
