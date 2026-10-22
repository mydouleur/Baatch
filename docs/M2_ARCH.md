# Baatch 推理引擎架构设计（M2 方案）

> 目标读者：项目决策者。本文是引擎的框架设计方案与实现细节，不含引擎代码。
> 版本：v0.1（2026-10-22，M0 探针数据已并入）

## 0. 一页纸摘要

为 Tiel Coder 35B-A3B（Qwen3.5-MoE 混合线性注意力架构）在 AMD XDNA2 NPU 上构建低功耗推理引擎。

- **目标硬件**：Ryzen AI 9 H 365（XDNA2 50 TOPS，LPDDR5x-8000，32 GB）
- **量化**：自研 BAA 容器 + XBF3 方案（3.0625 bpw，~14 GB），NPU 原生消费、零反量化
- **实测锚点**：NPU 路径 DRAM 带宽上限 ~60–65 GB/s（本机探针实测 ~60 持续，社区参照 65）
- **性能预期**：decode ~45 tok/s（裸）→ ×MTP(~1.8) ≈ **80 tok/s 有效**；turbo 混部（+CPU 冷专家流）可再 +30–60%；整机功耗 decode ~12–18 W / turbo ~22 W
- **可移植性**：backend 接缝 + 方案 tag（XBF3/IBF3/…），第二平台只加后端与打包器

## 1. 模型架构事实（从权重 config 实证）

| 项 | 值 | 引擎含义 |
|---|---|---|
| 层数 | 40 = 30 线性注意力 + 10 全注意力（3:1 间隔） | KV cache 仅挂 10 层 |
| 线性注意力 | Gated DeltaNet：conv k=4，16K/32V 头 ×128d，SSM state fp32 | decode = 状态更新小 op；prefill = chunk-wise matmul |
| 全注意力 | head_dim 256，16 头，attn_output_gate | 标准 GQA 路径，规模小 |
| MoE | 256 专家，top-8，intermediate 512 | 权重读取 90% 在专家 |
| 总/激活参数 | 35.3B / ~3B | decode bytes/token ≈ 1.34 GB（XBF3） |
| MTP | 1 层（含 shared expert），已训练 | 投机解码主引擎 |
| 上下文 | 262144；KV（10 层）16-bit 下 262k < 5 GB | 内存宽裕 |

## 2. .baa 容器与 XBF3 方案

### 2.1 容器分层

```
.baa 文件 = [头部元数据][张量目录][数据区]
头部：magic / 版本 / 架构标识(qwen3_5_moe) / 超参数 / chat 模板 / tokenizer 引用
张量目录：名称 / 形状 / 方案 tag(XBF3) / 数据偏移
数据区：按方案编码的张量流
```

- 容器厂商中立；方案 tag 决定后端 kernel 选择（XBF3→XDNA2 kernel，未来 IBF3→Intel）
- 参考 GGUF 的目录式布局（随机访问友好、mmap 友好），简化掉其历史包袱

### 2.2 XBF3 张量布局（3.0625 bpw）

```
per 张量（按行主序存储）：
  每 256 权重一个 superblock：
    [2B BF16 scale][96B packed 权重]
    packed：3-bit 补码，8 权重 = 3 字节，小端位流
attention QKV/O、lm_head：同构 4-bit profile（5.0625 bpw 含 scale）
router / embedding / norms：BF16 或 INT8 原样
```

- 设计原则：**格式即 MAC 输入**。DRAM 中以压缩态流入 NPU，片上 unpack 到 int8/int16 向量车道做 MAC，全程无 host 侧反量化
- 对齐约束：superblock 256 = AIE 向量 lane 的整数倍；张量行起始地址 64B 对齐（DMA burst 友好）
- 量化管线：Hadamard 旋转（离线，融合进相邻 RMSNorm/投影矩阵，推理零开销）+ 逐 superblock 最优 clip + 可选 AWQ 校准

## 3. 引擎分层架构

```
┌─ serve 层：OpenAI 兼容 API / CLI（最后做）
├─ 前端：tokenizer（BPE，复用 tokenizers 库 ABI）/ Sharp chat 模板（Jinja 子集）
├─ 调度器（核心）：
│    ├─ decode 循环 + MTP 投机验证（draft 1 层 → verify k tokens → accept）
│    ├─ turbo 混部：top-8 专家按带宽比分流 NPU/CPU，MoE reduce 汇合
│    ├─ KV/SSM state 管理（分页 KV 仅 10 层；SSM state 常驻 pinned 内存）
│    └─ 采样器（温度/top-p/top-k，CPU）
├─ 计算图 IR：厂商中立 tensor + 算子图；MoE scatter/gather、线性注意力状态更新为一级算子
└─ Backend 注册表（可移植缝）：
     ├─ XDNA2：专家 FFN GEMV/GEMM、attn 投影、MTP head、RMSNorm —— cardano/peano 编译的 xclbin，XRT 提交
     ├─ CPU（AVX-512）：embedding、router、采样、turbo 模式的冷专家流、fallback
     └─ （未来）Intel/Qualcomm 后端：同接口
```

### 3.1 Backend 接口（可移植性核心）

```c
// 伪接口，语义示意
struct baa_backend {
    caps      -> {支持的方案 tag 列表, 最大对齐, 是否共享内存}
    alloc(sz, alignment) -> buffer
    pack(tensor_meta, src) -> buffer      // 布局转换在此发生
    submit(graph_segment, inputs) -> event // 异步
    wait(event)
    bandwidth_hint() -> GB/s              // 调度器分流依据
};
```

纪律：IR 层不得出现任何 XDNA2 专属参数（tile 数、block 尺寸）；这些只能出现在 backend 与 .baa 打包器里。

## 4. 流水线细节

### 4.1 Decode（带宽受限，主战场）

每 token 数据流：

```
embed 查表(CPU) → for 每层:
  线性注意力层: qkv 投影(NPU) → conv4 状态更新(CPU/NPU 小 op) → DeltaNet 状态读写(NPU 小 GEMV) → o 投影(NPU)
  全注意力层: qkv 投影(NPU) → RoPE → KV append → attn(短序列 CPU/长序列 NPU) → gate → o 投影(NPU)
  MoE: router top-8(CPU) → 8 专家 FFN(NPU 流式读 XBF3；turbo 时冷专家分给 CPU AVX-512) → reduce
→ MTP head(NPU) → 采样(CPU)
```

关键设计：
- **专家权重流式读取**是唯一的大头：kernel 按"读一行算一行"组织，L2 双缓冲，所有 8 列 ShimTile 并发——探针证明此形状可到 ~60 GB/s
- NPU↔CPU 同步点每 token 仅 2 处（router 后、reduce 前），turbo 汇合成本待实测（已挂账）
- SSM state（每层 ~0.8 MB × 30 层 fp32 ≈ 25 MB）常驻内存，decode 时读写都在片上/内存，不进 DRAM 权重路径

### 4.2 Prefill（算力受限，NPU 甜区）

- 大 GEMM（batch=prompt 长度）走 BFP16/INT8 满阵列；实测参照 38 TOPS INT8（学术）/ 50 TOPS（标称）
- 线性注意力 chunk-wise 形式：chunk 内并行为 matmul，chunk 间顺序——天然 pipeline
- chunked prefill（默认 2048 token/块）防止长 prompt 阻塞 decode 交互
- 预期：2K prompt TTFT ~1–2 s（对标 FastFlowLM 0.6B 的 1356 tps prefill 按比例外推 + 我们 50 TOPS 修正）

### 4.3 MTP 投机循环

```
draft: MTP 层(1 层) 自回归出 k=4 个候选 token（轻量：1/40 的层成本）
verify: 主模型一次前向验证 4 个位置（decode 的小 GEMM 化，NPU 利用率甜区）
accept: 最长前缀接受，期望接受长度 ~1.8–2.5 → 有效吞吐 ×1.8+
无损：验证用主模型本身，输出分布不变
```

## 5. 内存预算（32 GB 机型）

| 项 | 大小 |
|---|---|
| 权重（XBF3 方案 A） | ~14 GB |
| KV cache（32k ctx，INT8，仅 10 层） | ~0.3 GB（262k 也就 <3 GB） |
| SSM+conv state（fp32，仅 30 层） | ~63 MB/序列（投机动机快照另倍增） |
| 激活/中间缓冲 + NPU workspace | ~2 GB |
| Windows + 余量 | ~8 GB |
| **合计** | **~25 GB，宽松** |

## 6. XDNA2 kernel 清单（M2' 要写的）

llama.cpp `qwen35moe` 参照实现已克隆至 `ref/llama.cpp`（@7775ffb7），调研笔记见 `notes/qwen3next_impl.md`。**注意参照 `qwen35moe.cpp` 而非 `qwen3next.cpp`**（k→v 头映射顺序、beta/alpha 张量拆分、RoPE 调用均不同，照错会静默出错）。

按优先级：

1. **XBF3 GEMV 流式 kernel**（专家 FFN，decode 生命线）——探针已验证数据流形状；专家为 3D 张量上的 `mul_mat_id`（gate_up 融合一次 + down 一次）+ swiglu_split + 8 路加权求和
2. **GDN decode 融合 kernel**——调研重要发现：**decode 递推完全是元素级 op（exp/mul/外积加/归约），无 matmul**；state fp32 [128,128,32]/层 ≈ 2 MiB，每 token 读改写一次，访存密集但形状规整，适合单融合 kernel（数学照抄 `build_delta_net_autoregressive`）
3. XBF3 小 batch GEMM（MTP verify 步 + 短 prefill）
4. depthwise 因果 conv1d（k=4）+ 状态拼接（小 kernel）
5. BFP16/INT8 大 GEMM（prefill，可先用 AIE API 参考设计改）
6. RMSNorm / RoPE(rope_multi 4 段) / SiLU / gate（小 kernel，或 CPU 兜底起步）
7. **GDN prefill chunked kernel**（cumsum/tri/solve_tri/块内 matmul/跨 chunk 递推）——移植工作量最大，**先 CPU 兜底**（llama.cpp 非融合路径有纯 ggml CPU 实现）再 NPU 化
8. 全注意力 decode kernel（QK^T/softmax/AV，仅 10 层，含 attn_output_gate 的 sigmoid 门乘，可后置）

CPU 兜底：beta/alpha/dt 小投影、L2 norm、MoE 路由（256 维 softmax+top-k）、MTP 的 enorm/hnorm/eh_proj。

**MTP 关键机制**：draft 与主干共享内存、主干经 `h_nextn` 导出 hidden；投机回滚依赖 **n_rs_seq 状态快照**（GDN 算子每步写 min(n_tokens, n_rs_seq+1) 份快照）——自研调度器必须复制这个快照/回滚不变量（ubatch 尾部 n_rs_seq+1 个 token 同批）。

## 7. 构建与仓库布局

```
xdna_llm/
├─ docs/           # DESIGN.md(总纲) M2_ARCH.md(本文)
├─ m0/probe/       # 带宽探针（已出数 ~60 GB/s）
├─ tools/quant/    # M1' 量化实验室（Python，safetensors → .baa）
├─ models/         # Ornith BF16 源 + sharp_chat_template.jinja
├─ ref/            # llama.cpp 参考（调研用，不链入）
├─ kernels/        # AIE MLIR + host wrapper（cardano/peano 编译）
├─ src/
│   ├─ ir/         # 计算图 IR
│   ├─ backends/   # xdna2/ cpu/ ...
│   ├─ sched/      # decode 循环、MTP、turbo 分流
│   └─ serve/      # API
└─ third_party/    # XRT 头文件（Apache-2.0）等
```

工具链：host = MSVC/Clang + CMake；kernel = cardano(aiecompiler) + peano(clang)，Windows 原生闭环（M0 已验证编译链可用）。运行时依赖仅 XRT（xrt_coreutil.dll，驱动自带）。

## 8. 风险与未决项（按优先级）

1. **XBF3 GEMV kernel 实测带宽**——DMA 形状已实测 ~60 GB/s（读写合并）；**纯读上限未分离（30 vs 60 的生死问题，挂账）**；带计算的 GEMV 能否守住 ~55+ GB/s 待 M2' 验证（最大技术风险）
2. **XBF3 质量**——M1' v0 已出：SNR 达 3-bit 理论极限带（14.05 dB），谨慎乐观；端到端 PPL 与真实激活校准是 v1 门槛。不达标退路：4-bit profile（~17.7 GB）
3. **turbo 汇合开销**——M3' 实测；不达标则 turbo 降级为实验模式
4. **线性注意力 kernel 碎片化**——小 op 多的调度开销，可能需要 op 融合或 CPU 承担
5. **MTP 接受率**——~1.8 是文献经验值，Tiel 实测待 M4'

## 9. 里程碑映射

- M0 ✅ 工具链 + 带宽探针（读写合并 ~60–61 GB/s 实测；纯读上限挂账；xrt-smi 基线 GEMM 51.3 TOPS）
- M1' ✅ 量化管线 v0 + 层级质量报告（SNR 14.05 dB 达 3-bit 理论带；v1 = 真实校准语料 + AWQ + 端到端 PPL）
- M2' XBF3 GEMV kernel 打满带宽 → 单专家 decode 路径（纯读探针建议在此阶段以 MemTile sink ctrlcode 合成补上）
- M3' 全模型 decode（GDN decode 融合 kernel + CPU 兜底小 op）+ turbo 分流
- M4' MTP 投机（含 n_rs_seq 快照回滚）+ prefill 优化（GDN chunk kernel 可 CPU 兜底起步）+ serve 层
