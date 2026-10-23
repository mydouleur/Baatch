#!/usr/bin/env python3
"""一次性对拍：内联（验证过 ≈HF）vs ref_layer.py 输出的逐中间量比较。"""
import numpy as np, sys
sys.path.insert(0, "tools/quant")
from safetensors_reader import ModelReader
mr = ModelReader(r"D:\xdna_llm\models\Ornith-1.5-35B-A3B")
H=2048; KH,VH,HD=16,32,128; KEY_DIM,VAL_DIM=2048,4096; EPS=1e-6
NE,TOPK,MI=256,8,512
def T(n): return mr.read("model.language_model.layers.0."+n)
def sigmoid(x): return 1/(1+np.exp(-x))
def silu(x): return x*sigmoid(x)
def softplus(x): return np.logaddexp(0,x)
emb = mr.read("model.language_model.embed_tokens.weight", (248045,))
x = emb.astype(np.float32)
xn = x/np.sqrt((x**2).mean()+EPS)*(1+T("input_layernorm.weight"))
qkv = T("linear_attn.in_proj_qkv.weight") @ xn
z = T("linear_attn.in_proj_z.weight") @ xn
beta = sigmoid(T("linear_attn.in_proj_b.weight") @ xn)
alpha = T("linear_attn.in_proj_a.weight") @ xn
g = -softplus(alpha + T("linear_attn.dt_bias")) * np.exp(T("linear_attn.A_log"))
cw = T("linear_attn.conv1d.weight")[:,0,:]
qc = silu(qkv * cw[:,3])
q,k,v = qc[:KEY_DIM], qc[KEY_DIM:KEY_DIM*2], qc[KEY_DIM*2:]
def l2n(vec):
    vec = vec.reshape(-1,HD)
    return (vec/np.sqrt((vec**2).mean(axis=-1,keepdims=True)+EPS/HD)/np.sqrt(HD)).reshape(-1)
q,k = l2n(q), l2n(k)
q = q/np.sqrt(HD)
o = np.zeros(VAL_DIM, dtype=np.float32)
for j in range(VH):
    kj = k[(j//2)*HD:(j//2+1)*HD]; qj = q[(j//2)*HD:(j//2+1)*HD]; vj = v[j*HD:(j+1)*HD]
    s = np.outer(vj*beta[j], kj)
    o[j*HD:(j+1)*HD] = s @ qj
og = o.reshape(VH,HD)
og = og/np.sqrt((og**2).mean(axis=-1,keepdims=True)+EPS)*T("linear_attn.norm.weight")
og = (og*silu(z.reshape(VH,HD))).reshape(-1)
x1 = x + T("linear_attn.out_proj.weight") @ og
np.save("inline_ygdn.npy", x1)

yg = np.fromfile("src/engine/testvec_embed/y_gdn.bin", dtype=np.float32)
d = x1 - yg
print("inline y_gdn vs ref_layer y_gdn: rel=%.5f  max|d|=%.4f" % (
    np.sqrt((d**2).sum()/((yg**2).sum())), np.abs(d).max()))
hf = np.fromfile("probe_hf0/ylayer_00.bin", dtype=np.float32)
print("inline y_gdn vs HF(pre-MoE 不可比分)... 只看 y_full 层")
# 检查 ref_layer 的 x_in 是否真的是 embed[248045]
xin = np.fromfile("src/engine/testvec_embed/x_in.bin", dtype=np.float32)
print("ref_layer x_in == embed[248045]:", np.abs(xin - x).max())
EOF_MARK = None
