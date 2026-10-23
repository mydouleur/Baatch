#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""全模型单 token 参考（numpy，直读 BF16 safetensors）——定位逐层偏差。

用法：python ref_full.py --ids 760 --out probe_full/
写出：每层的 hidden（probe_full/ylayer_NN.bin fp32）+ logits_top10.txt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import ModelReader

H = 2048
KH, VH, HD = 16, 32, 128
KEY_DIM, VAL_DIM, QKV = 2048, 4096, 8192
QH, KVH, AHD = 16, 2, 256
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
    ap.add_argument("--out", default="probe_full")
    ap.add_argument("--ids", default="760")
    ap.add_argument("--flip-gu", action="store_true", help="gate/up 两半对调（约定探测）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mr = ModelReader(a.src)

    tok = int(a.ids.split(",")[0])
    emb = mr.read("model.language_model.embed_tokens.weight")
    x = emb[tok].astype(np.float32)

    for L in range(40):
        P = f"model.language_model.layers.{L}."
        def T(n): return mr.read(P + n)
        is_gdn = (L + 1) % 4 != 0
        xn = rmsnorm(x, T("input_layernorm.weight"))
        if is_gdn:
            qkv = T("linear_attn.in_proj_qkv.weight") @ xn
            z = T("linear_attn.in_proj_z.weight") @ xn
            beta = sigmoid(T("linear_attn.in_proj_b.weight") @ xn)
            alpha = T("linear_attn.in_proj_a.weight") @ xn
            g = -softplus(alpha + T("linear_attn.dt_bias")) * np.exp(T("linear_attn.A_log"))
            cw = T("linear_attn.conv1d.weight")[:, 0, :]
            qc = silu(qkv * cw[:, 3])          # 零 conv 状态
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
                s = np.outer(vj * beta[j], kj)     # 零状态：s = d ⊗ k
                o[j * HD:(j + 1) * HD] = s @ qj
            og = o.reshape(VH, HD)
            og = og / np.sqrt((og ** 2).mean(axis=-1, keepdims=True) + EPS) * T("linear_attn.norm.weight")
            og = (og * silu(z.reshape(VH, HD))).reshape(-1)
            x = x + T("linear_attn.out_proj.weight") @ og
        else:
            qg = (T("self_attn.q_proj.weight") @ xn).reshape(QH, 2 * AHD)
            q = qg[:, :AHD]
            gate = qg[:, AHD:]
            kk = T("self_attn.k_proj.weight") @ xn
            vv = T("self_attn.v_proj.weight") @ xn
            q = q / np.sqrt((q ** 2).mean(axis=-1, keepdims=True) + EPS) * (1 + T("self_attn.q_norm.weight"))
            kk = kk.reshape(KVH, AHD)
            kk = kk / np.sqrt((kk ** 2).mean(axis=-1, keepdims=True) + EPS) * (1 + T("self_attn.k_norm.weight"))
            # pos=0 → rope 恒等；KV cache 只有本 token → 注意力输出 = v（按头）
            out = np.zeros((QH, AHD), dtype=np.float32)
            for h in range(QH):
                kvh = h // (QH // KVH)
                out[h] = vv[kvh * AHD:(kvh + 1) * AHD]
                out[h] *= sigmoid(gate[h])
            x = x + T("self_attn.o_proj.weight") @ out.reshape(-1)

        # MoE
        x2 = rmsnorm(x, T("post_attention_layernorm.weight"))
        logits_e = T("mlp.gate.weight") @ x2
        wexp = np.exp(logits_e - logits_e.max()); wexp /= wexp.sum()
        top = np.argsort(-wexp)[:TOPK]
        tw = wexp[top]; tw /= tw.sum()
        gu_all = T("mlp.experts.gate_up_proj")
        dn_all = T("mlp.experts.down_proj")
        acc = np.zeros(H, dtype=np.float32)
        for tt, e in enumerate(top):
            gu = gu_all[e] @ x2
            ggate, uup = (gu[MI:], gu[:MI]) if a.flip_gu else (gu[:MI], gu[MI:])
            acc += tw[tt] * (dn_all[e] @ (silu(ggate) * uup))
        sg = T("mlp.shared_expert.gate_proj.weight") @ x2
        su = T("mlp.shared_expert.up_proj.weight") @ x2
        sh = T("mlp.shared_expert.down_proj.weight") @ (silu(sg) * su)
        g0 = sigmoid(T("mlp.shared_expert_gate.weight") @ x2)
        x = x + acc + g0[0] * sh

        x.astype(np.float32).tofile(os.path.join(a.out, f"ylayer_{L:02d}.bin"))
        print(f"L{L:02d} {'gdn' if is_gdn else 'att'} |x|max={np.abs(x).max():.4f} rms={np.sqrt((x**2).mean()):.4f}", flush=True)

    x = rmsnorm(x, mr.read("model.language_model.norm.weight"))
    logits = mr.read("lm_head.weight") @ x
    top10 = np.argsort(-logits)[:10]
    with open(os.path.join(a.out, "logits_top10.txt"), "w") as f:
        for t in top10:
            f.write(f"{t} {logits[t]:.4f}\n")
    print("top10:", top10.tolist())

if __name__ == "__main__":
    main()
