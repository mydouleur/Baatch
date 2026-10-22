# llama.cpp Qwen3-Next / Qwen3.5-MoE 实现笔记

参照版本：llama.cpp @ 7775ffb7（2026-10-22）。本地路径 `ref/llama.cpp/`。

**首要结论**：我们的 Ornith-1.5-35B-A3B（40 层）对应的不是 `qwen3next`，而是较新的 **`qwen35moe`** 架构（GGUF arch 名 `qwen3_5_moe`）——`src/models/qwen35moe.cpp:28-33` 中 `n_layer==40 → LLM_TYPE_35B_A3B` 精确匹配。`qwen3next.cpp`（48 层 80B-A3B）是上一代，两者结构几乎相同但张量布局有若干关键差异（见 §6）。

## 1. 文件地图

- `src/models/qwen35moe.cpp`（735 行）：我们模型的完整实现（权重加载 + 主干 graph + MTP graph）
- `src/models/qwen3next.cpp`（812 行）：上一代 Qwen3-Next-80B
- `src/models/delta-net-base.cpp`：Gated DeltaNet 三条路径（chunked prefill / autoregressive decode / fused 单算子），被 qwen35moe/qwen3next/qwen35/kimi-linear 等共用
- `src/models/models.h:14-18`：`build_gdn_l2_norm`（GDN 的 q/k L2 归一化）
- `src/llama-graph.cpp:2186-2577`：`build_moe_ffn`（MoE 通用构建器）
- `src/llama-memory-hybrid.cpp` / `src/llama-memory-recurrent.cpp`：混合 KV+recurrent 状态缓存
- `common/speculative.cpp:1475-1850`：MTP 投机解码驱动（`draft-mtp`）

## 2. 层类型调度与混合内存

层类型由 `full_attention_interval=4` 推出：`(i+1)%4 != 0 → 线性注意力`，即第 3、7、11、…、39 层是全注意力（3:1 混合），`qwen35moe.cpp:20-26`。MTP 层（索引 = n_layer）强制标记为非 recurrent（同处注释）。

主循环逐层分发（`qwen35moe.cpp:178-226`）：

```
if (hparams.is_recr(il)) cur = build_layer_attn_linear(inp->get_recr(), ...);
else                     cur = build_layer_attn(inp->get_attn(), ...);
```

混合内存 `llama_memory_hybrid`（`llama-memory-hybrid.cpp:34-65`）= 一个 `llama_kv_cache` + 一个 `llama_memory_recurrent`，用过滤器划分：**KV cache 只给 `!is_recr` 层分配**（默认 filter_attn，`:48-50`），recurrent 状态只给 `is_recr` 层（`:62-64`）。即 KV cache 只有 10 层（40 层模型的 1/4），每层线性注意力只有固定大小的状态。

## 3. 线性注意力（Gated DeltaNet）

### 3.1 权重张量（qwen35moe.cpp:83-91）

| 张量 | 形状 | 说明 |
|---|---|---|
| `attn_qkv.weight` (wqkv) | {n_embd, 2·key_dim + value_dim} | q,k,v 融合投影 |
| `attn_gate.weight` (wqkv_gate) | {n_embd, value_dim} | 输出门 z |
| `ssm_conv1d.weight` | {d_conv=4, conv_dim} | depthwise 因果卷积 |
| `ssm_dt.bias` / `ssm_a` | {n_v_heads} | dt 偏置 / -A_log |
| `ssm_beta.weight` / `ssm_alpha.weight` | {n_embd, n_v_heads} | beta / alpha 投影 |
| `ssm_norm.weight` | {head_v_dim} | 输出 RMSNorm |
| `ssm_out.weight` | {value_dim, n_embd} | 输出投影 |

对我们的模型：n_k_heads=16（ssm_n_group）、n_v_heads=32（ssm_dt_rank）、head_dim=128（ssm_d_state）、d_inner=4096、key_dim=2048、value_dim=4096、conv_dim=8192。

### 3.2 层内数据流（build_layer_attn_linear，qwen35moe.cpp:357-486）

1. `qkv_mixed = x @ wqkv`，`z = x @ wqkv_gate`（`build_qkvz`，:249-263，两个 mul_mat）
2. `beta = sigmoid(x @ ssm_beta)`；`alpha = x @ ssm_alpha`；`gate = softplus(alpha + ssm_dt) * ssm_a`（对数域衰减，:380-398）
3. **conv 状态**：缓存的最后 d_conv-1=3 个 token 与 qkv_mixed 拼接 → `ggml_ssm_conv`（ggml.c:5683，depthwise 因果 conv）→ SiLU（:407-417）。状态写回用 `ggml_cpy` 写 cache 视图（`delta-net-base.cpp:446-522`）
4. 从 conv 输出切出 q/k/v 视图，q、k 过 `build_gdn_l2_norm` = `rms_norm(x, eps/n) * 1/sqrt(n)`（models.h:14-18）
5. **k→v 头广播**：num_k_heads(16) ≠ num_v_heads(32) 时把 q,k repeat 到 32 头（:460-464；fused GDN 支持广播时可跳过）。**注意 qwen3.5 与 qwen3next 的头映射顺序不同**（见 §6）
6. `build_recurrent_attn` → `build_delta_net`（delta-net-base.cpp:422-444）：`n_seq_tokens==1` 走 decode 路径，否则走 chunk 路径（或 fused 算子）
7. 输出过 gated norm：`RMSNorm(o) * SiLU(z)`（`build_norm_gated`，:265-274），再 `x @ ssm_out`

### 3.3 Decode 路径（build_delta_net_autoregressive，delta-net-base.cpp:286-368）

状态 `s` 形状 **[head_v_dim=128, 128, n_v_heads=32, n_seqs]**，fp32。单 token 更新全是元素级/归约 op：

```
s = s * exp(g)                    # 衰减
sk = sum_rows(s * k)              # 读取旧值 (S_k 维归约)
d = (v - sk^T) * b                # delta rule
s = s + repeat(k) * d^T           # 外积写回
o = sum_rows(s * q)               # 输出
```

op 清单：`exp, mul, sum_rows, sub, transpose, repeat, add, scale`——**decode 完全没有 matmul**，非常适合一个融合 kernel；llama.cpp 也正是提供了 fused 版。

### 3.4 Prefill chunk 路径（build_delta_net_chunking，:16-284）

chunk 大小 CS=64（GDN；KDA 是 16，:61）。要点：

- gate 沿 chunk 内做 `ggml_cumsum`，构造下三角 decay mask（`ggml_tri + ggml_exp`，:89-145）
- WY 表示用 `ggml_solve_tri` 解 `(I + tril(kb))` 三角方程（:155-168）
- 逐 chunk 循环：块内注意力 `mul_mat` + 跨 chunk 状态递推 `s = s*g_last + kg^T@v_new`（:232-271）
- 涉及的重 op：`cumsum, tri, exp, solve_tri, pad, 多次 mul_mat`——prefill 的 NPU 移植工作量集中在这一坨

### 3.5 Fused 算子 ggml_gated_delta_net

`GGML_OP_GATED_DELTA_NET`（ggml.h:583，构建于 ggml.c:6434），单算子输出 output+new_state（`build_delta_net_fused`，:370-420；带快照版 :524-603）。CPU 后端有实现（ggml-cpu.c:2088），Metal/Hexagon 也有。默认开启（`cparams.fused_gdn_ar/ch = true`，llama-context.cpp:325-326），auto 模式会探测后端能力后回退（llama-context.cpp:664-669）。

### 3.6 状态缓存的形状与精度

- `cache_r_l`（conv 状态）：2D `{n_embd_r, n_rows}`，n_embd_r = (d_conv-1)·conv_dim = 3·8192 = 24576（llama-hparams.cpp:229）
- `cache_s_l`（SSM/delta 状态）：2D `{n_embd_s, n_rows}`，n_embd_s = ssm_d_state·ssm_d_inner = 128·4096 = 524288 ≈ **每层每序列 2 MiB**（llama-hparams.cpp:255）
- **两者一律 fp32**：`recurrent_type_r/s = GGML_TYPE_F32`（llama-model.cpp:2914-2915、3165-3166；分配于 llama-memory-recurrent.cpp:101-107）
- `n_rows = mem_size * (1 + n_rs_seq)`：n_rs_seq 是投机解码用的回滚快照槽（llama.h:371）

## 4. 全注意力层（build_layer_attn，qwen35moe.cpp:276-355）

1. **Q 与 gate 融合在一个投影里**：`create_tensor_qkv` 的 q 输出宽度是 `n_embd_head * n_head * 2`（:74），前半是 Q、后半是 gate，用 view 切开（:297-315）
2. Q/K 逐头 RMSNorm（attn_q_norm / attn_k_norm，:303-309）
3. RoPE 用 **`ggml_rope_multi` + 4 段 rope_sections**（:321-331；元数据 `LLM_KV_ROPE_DIMENSION_SECTIONS`，:9）——qwen3next 用的是普通 `ggml_rope_ext`。⚠️ 需要对照我们 GGUF 的 rope sections 元数据确认纯文本路径下各段配置
4. `build_attn`（标准 KV-cache 注意力）
5. **attn_output_gate**：`out = attn_out * sigmoid(gate)`（:345-349），再乘 wo
6. kq_scale 默认 `1/sqrt(head_dim)`，可被 `f_attention_scale` 覆盖（:338）

## 5. MoE（build_moe_ffn，llama-graph.cpp:2186-2577）

qwen35moe 调用参数（:492-506）：softmax 门控、`norm_w=true`（top-k 权重归一化）、`w_scale=expert_weights_scale`。

- 路由：`logits = x @ ffn_gate_inp`（小 GEMM，n_embd×256）→ softmax → `ggml_argsort_top_k`（:2304）→ `ggml_get_rows` 取权重 → sum 归一化（clamp 到 fp16 最小值防除零，:2329-2343）
- 专家计算：专家权重是 3D 张量 `{n_ff_exp, n_embd, n_expert}`；gate+up 融合为 `{2*n_ff_exp, n_embd, n_expert}`，一次 `ggml_mul_mat_id`（经 `build_lora_mm_id`，:1712-1713）再按 view 切 gate/up（:2367-2385）→ `ggml_swiglu_split`（:2444）→ down 再一次 `mul_mat_id`（:2513）
- 归约：专家输出乘权重后对 n_expert_used 个 view 做 ggml_add 链（:2538-2567）
- 共享专家：普通 dense SwiGLU + 独立 sigmoid 门（`ffn_gate_inp_shexp` 输出 1 维），与 MoE 结果相加（qwen35moe.cpp:510-536）

## 6. qwen35moe 与 qwen3next 的差异（移植时最容易踩的坑）

1. **beta/alpha**：qwen3next 用融合的 `ssm_ba.weight`（{n_embd, 2·n_v_heads}，qwen3next.cpp:92），qwen35moe 拆成 `ssm_beta`/`ssm_alpha` 两张（:88-89）
2. **qkv 投影**：qwen3next 保留 legacy `ssm_in` 融合 qkvz 路径（qwen3next.cpp:311-390），qwen35moe 只有 wqkv + wqkv_gate
3. **k→v 头映射顺序不同**！llama-model.cpp:654-657 明确注释：qwen3next 是 `[k0_v0, k0_v1, k1_v2, k1_v3]`，qwen3.5 是 `[k0_v0, k1_v1, k0_v2, k1_v3]`（repeat-interleave 模式不同，qwen3next.cpp:517-530 vs qwen35moe.cpp:462-463）。我们模型是 qwen3.5 模式
4. RoPE：qwen3next `ggml_rope_ext`，qwen35moe `ggml_rope_multi`（4 段）

## 7. MTP / NextN 支持现状

**已完整支持单层 MTP 投机解码**（`GGML_ASSERT(n_layer_nextn == 1)`，qwen35moe.cpp:548），架构：

- **权重加载**：MTP block 作为第 n_layer 层加载到同一模型里（`load_block_mtp`，:106-139）：一个全注意力 Qwen3.5 block + MoE FFN + nextn 专属张量 `eh_proj {2·n_embd, n_embd}`、`enorm`、`hnorm`，可选独立 `embed_tokens`/`shared_head`；`nextn_flags`（llama-model.cpp:3898）支持 GGUF 只含主干/只含 MTP/两者皆有
- **图选择**：`build_arch_graph` 对 `LLM_GRAPH_TYPE_DECODER_MTP` 返回 `graph_mtp`（:149-154）
- **draft 前向**（graph_mtp，:545-735）：输入 = token + 主干最后一层 norm 后的 hidden `h_nextn`；`concat(RMSNorm(e,enorm), RMSNorm(h,hnorm))` → `eh_proj` → 单个全注意力 block（有自己的 KV cache 条目，il=n_layer）→ MoE FFN → shared head
- **驱动端**（common/speculative.cpp `draft-mtp`，:1475-1850）：draft ctx 用 `ctx_type = LLAMA_CONTEXT_TYPE_MTP` 创建（:2660-2662）且与 target **共享内存**（`ctx_other = ctx_tgt`，:2670）；target 通过 `llama_set_embeddings_nextn(ctx_tgt, true, masked=false)` 暴露 hidden（:1520），draft 输出自己的 pre-head hidden 供下一步链接（:1775）；n_layer_nextn>1 时用 `llama_set_nextn_layer_offset` 链多个头（:1653）
- **回滚**：混合模型的投机解码需要 recurrent 状态回滚 → target ctx 设 `n_rs_seq > 0`（common.h:401-405 的 `need_n_rs_seq()` 对 DRAFT_MTP 为真；common.cpp:1720 注入），GDN 算子每步写 min(n_tokens, n_rs_seq+1) 份状态快照（delta-net-base.cpp:560-600）

## 8. 对我们 NPU 引擎的可移植结论

**必须自研（或重点优化）的 NPU kernel**：

1. **GDN decode 融合 kernel**：§3.3 的 6 个元素级 op（state 128×128×32 fp32，每 token 读改写 ~2 MiB/layer）——访存密集型，融合后一次 state 往返；直接照抄 `build_delta_net_autoregressive` 的数学即可，无 matmul
2. **GDN prefill chunked kernel**：§3.4，含 cumsum/tri/solve_tri/块内 matmul/跨 chunk 状态递推——工作量最大的一块；可先 CPU 兜底（llama.cpp 非融合路径即是纯 ggml 组合，有 CPU 实现）再逐步 NPU 化
3. **depthwise 因果 conv1d（k=4）+ 状态拼接/写回**：`ggml_ssm_conv` 语义简单（ggml.c:5683），kernel 4 很小，NPU 上好写
4. **mul_mat_id 专家 GEMM**（256 专家 top-8，intermediate 512，量化权重）：decode 时 batch=1，本质是 8 次专家 gather + GEMV + 加权求和；归一化/softmax/top-k（argsort）可在 CPU 或小 kernel
5. **融合 Q+gate 投影的切分 + sigmoid 门乘**（attn_output_gate）：只需在 QKV GEMM 输出布局上注意 gate 位于每头后半段

**可以 CPU 兜底的**：beta/alpha/dt 小投影（n_embd→32）、L2 norm、gated RMSNorm、状态 cache 的 cpy/view 管理、MoE 路由（256 维 logits 的 softmax+top-k）、MTP 的 enorm/hnorm/eh_proj（单层，每步一次）。

**状态内存预算**（每序列）：线性注意力 30 层 × (conv 96 KiB + ssm 2 MiB) ≈ 63 MiB fp32；KV cache 只有 10 个全注意力层（16 头 × 256d）。投机解码回滚按 (1+n_rs_seq) 倍增 recurrent 状态。

**不确定/未深挖的点**：
- qwen35moe 的 4 段 rope_sections 在我们 GGUF 里的具体取值（纯文本模型是否退化为单段）——需要 dump Ornith GGUF 元数据验证
- fused GDN 的 CPU 数值路径与非融合 chunk 路径的逐位一致性未验证（llama.cpp 测试套件 test-backend-ops 有覆盖，但我们自研 kernel 需自己对齐参考输出）
- `n_rs_seq` 快照机制在 MTP 拒绝 token 后的回放正确性依赖"ubatch 尾部 n_rs_seq+1 个 token 在同一 ubatch"（delta-net-base.cpp:495-497 注释），我们自研调度器要复制这个不变量
- MTP 的 draft KV cache 与主干共享 memory 对象时的 seq_rm 交互细节没有逐行追完（common/speculative.cpp:1648-1654 区域）
