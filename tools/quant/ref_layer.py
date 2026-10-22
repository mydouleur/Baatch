#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""阶段 2 参考：单层（GDN + MoE）numpy 参考，直接读 BF16 safetensors（未量化）。

输出到 --out 目录：x_in.bin / y_gdn.bin / y_full.bin（fp32）。
与 C++ 引擎（读 .baa，专家 XBF3）对拍；差异 = XBF3 量化噪声 + fp32 噪声。
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
    return x / np.sqrt((x ** 2).mean() + eps) * w

def sigmoid(x): return 1 / (1 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def softplus(x): return np.logaddexp(0, x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\xdna_llm\models\Ornith-1.5-35B-A3B")
    ap.add_argument("--out", default="testvec_layer0")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mr = ModelReader(a.src)
    L = a.layer
    P = f"model.language_model.layers.{L}."

    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal(H).astype(np.float32)

    def T(n): return mr.read(P + n)

    xn = rmsnorm(x, T("input_layernorm.weight"))
    qkv = T("linear_attn.in_proj_qkv.weight") @ xn          # 8192
    z = T("linear_attn.in_proj_z.weight") @ xn              # 4096
    beta = sigmoid(T("linear_attn.in_proj_b.weight") @ xn)  # 32
    alpha = T("linear_attn.in_proj_a.weight") @ xn          # 32
    g = -softplus(alpha + T("linear_attn.dt_bias")) * np.exp(T("linear_attn.A_log"))

    # conv（state=0）：y[c] = silu(qkv[c] * w[c,3])。注意 conv 核索引序：w[c,0..3] 对 window[0..3]
    cw = T("linear_attn.conv1d.weight")[:, 0, :]            # [8192,4]
    qc = silu(qkv * cw[:, 3])                               # state 全零 → 只有 t=3 项
    q, k, v = qc[:KEY_DIM], qc[KEY_DIM:KEY_DIM*2], qc[KEY_DIM*2:]

    def l2n(vec):  # per-head L2 norm（rms(x, eps/n)/sqrt(n) 等价形）
        vec = vec.reshape(-1, HD)
        return (vec / np.sqrt((vec ** 2).mean(axis=-1, keepdims=True) + EPS / HD) / np.sqrt(HD)).reshape(-1)
    q, k = l2n(q), l2n(k)

    # delta rule（零初始状态 → sk=0, d=v*beta, s=k⊗d, o=s@q）
    o = np.zeros(VAL_DIM, dtype=np.float32)
    for j in range(VH):
        kj = k[(j % KH) * HD:(j % KH + 1) * HD]
        qj = q[(j % KH) * HD:(j % KH + 1) * HD]
        vj = v[j * HD:(j + 1) * HD]
        s = np.outer(kj, vj * beta[j]).T           # s[i,m] = k[m]*v[i]*beta（decay 乘 0 状态=0）
        o[j * HD:(j + 1) * HD] = s @ qj
    nw = T("linear_attn.norm.weight")
    og = o.reshape(VH, HD)
    og = og / np.sqrt((og ** 2).mean(axis=-1, keepdims=True) + EPS) * nw
    og = (og * silu(z.reshape(VH, HD))).reshape(-1)
    y_gdn = x + T("linear_attn.out_proj.weight") @ og

    # MoE
    x2 = rmsnorm(y_gdn, T("post_attention_layernorm.weight"))
    logits = T("mlp.gate.weight") @ x2
    wexp = np.exp(logits - logits.max()); wexp /= wexp.sum()
    top = np.argsort(-wexp)[:TOPK]
    tw = wexp[top]; tw /= tw.sum()
    gu_all = T("mlp.experts.gate_up_proj")     # [256,1024,2048]
    dn_all = T("mlp.experts.down_proj")        # [256,2048,512]
    acc = np.zeros(H, dtype=np.float32)
    for t, e in enumerate(top):
        gu = gu_all[e] @ x2
        h = silu(gu[:MI]) * gu[MI:]
        acc += tw[t] * (dn_all[e] @ h)
    sg = T("mlp.shared_expert.gate_proj.weight") @ x2
    su = T("mlp.shared_expert.up_proj.weight") @ x2
    sh = T("mlp.shared_expert.down_proj.weight") @ (silu(sg) * su)
    g0 = sigmoid(T("mlp.shared_expert_gate.weight") @ x2)
    y_full = y_gdn + acc + g0[0] * sh

    x.astype(np.float32).tofile(os.path.join(a.out, "x_in.bin"))
    y_gdn.astype(np.float32).tofile(os.path.join(a.out, "y_gdn.bin"))
    y_full.astype(np.float32).tofile(os.path.join(a.out, "y_full.bin"))
    np.save(os.path.join(a.out, "topk.npy"), top)
    print(f"layer {L}: x→y_full done. top8={top.tolist()} |y_gdn|max={np.abs(y_gdn).max():.4f} |y_full|max={np.abs(y_full).max():.4f}")

if __name__ == "__main__":
    main()
