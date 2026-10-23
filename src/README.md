# src/ — Baatch 引擎本体（M2' 起）

规划结构（见 docs/M2_ARCH.md §3）：

```
src/
├─ ir/        # 厂商中立计算图 IR（tensor + 算子图）
├─ backends/  # xdna2/（主力）、cpu/（AVX-512 配角 + turbo 冷专家流）
├─ sched/     # decode 循环、MTP 投机、turbo 分流、KV/SSM 状态管理
└─ serve/     # OpenAI 兼容 API（最后做）
```

纪律：IR 层禁止出现 XDNA2 专属参数；后端经 `baa_backend` 接口注册（M2_ARCH.md §3.1）。

## M3'-a 现状（2026-10-23，阶段 1–3 全通）

`src/engine/`（plain C++17 + AVX-512 单二进制，未按上面目录拆分——v0 骨架）：

- `baa.h` — .baa 容器 mmap loader（目录二分 + 线性兜底）
- `kernels.h` — bf16 GEMV（2 大核池 LP0/LP2）+ vnni2 XBF3 GEMV（M3'-1 实测版）+
  激活 int8 量化 + 小 ops
- `model.h` — qwen35moe 全部层数学（GDN 递归 / 全注意力 RoPE+gate / MoE+共享专家）
- `main.cpp` — `--test-layer/--test-attn`（单层对拍）、`--probe`（逐层 hidden dump）、
  `--ids/--n`（全模型贪心 decode）

**关键约定（全部经 HF transformers 真值核实，qwen3_5_moe ≠ qwen3next/llama 笔记的若干处）**：

1. **RMSNorm 两种**：层 norm（input/post/q/k/final）= x/rms·**(1+w)**（零中心权重，
   Gemma 式）；GDN 输出门控 norm = x/rms·**w**（ones 初始化）。搞反 = 全模型漂移。
   （notes/qwen3next_impl.md 未覆盖此点——以 transformers/modeling_qwen3_5_moe.py 为准）
2. **k→v 头广播 = repeat_interleave（v_j ← k_{j/2}）**——不是笔记 §6.3 说的交错
   j%16（那是 GGUF/llama 张量序；HF safetensors 直读用 j//2）。
3. GDN：q 先乘 1/√128（llama delta-net decode 内的 ggml_scale）；gate g =
   −softplus(α+dt_bias)·exp(A_log)；conv tap w[3]×当前（F.conv1d 互相关不翻转）。
4. RoPE：partial 0.25×256=64 维，theta=1e7，interleaved 配对；文本路径各段同位
   （mrope sections 退化等价普通 rope）。

**验证**：layer0（GDN+MoE）与 layer3（attn+MoE）单层对拍 PASS（C++ vs numpy，
rel_l2≤3e-4）；numpy 参考 vs HF transformers 逐层 rel_l2 = 3e-3（XBF3 量化噪声量级）；
全模型贪心 decode 首 token = `<think>`（与 HF 完全一致），后续连贯英文。

**decode 速度现状**：~8.5 tok/s（2 大核）——瓶颈不是专家 XBF3（vnni2 39 GB/s 达标），
而是**其余全部 bf16 权重**（qkv/z/out/shared/router/lm_head ≈ 4.4GB/token 纯 DRAM 流，
39 GB/s → ~113ms/token 下限）。要 ≥20 tok/s 必须按 v1 定案把非专家权重也压成
XBF2/XBF4（这就是混合精度配比的动机）。

### v1 定案顺手遵守项（来自主线 docs）

专家 dtype 粒度：**当前 .baa v0 的 scheme 是按张量的**；v1 要求按专家独立 dtype
（核心层 XBF2/边缘层 XBF4 混合）——引擎 `Model::mv_xbf3` 已按张量 scheme 分发，
writer 侧改成"专家 3D 张量拆成按专家独立目录项（或加 per-expert scheme 数组）"
即可原生支持，待 v1 容器格式落地时做。

## 工具（tools/quant/）

- `baa_writer.py` — safetensors → .baa v0.1（专家 XBF3 快速 absmax 路径，其余 BF16；
  17.6GB / ~8min）
- `ref_layer.py` / `ref_attn.py` — 单层 numpy 参考（多 token 递归，--x-embed 真输入）
- `ref_full.py` / `hf_probe.py` / `hf_hooks.py` / `hf_gdn_hooks.py` — 全模型参考 /
  HF 逐层与逐子模块 hook 抓取（torch CPU 版 transformers 已装在本机）
