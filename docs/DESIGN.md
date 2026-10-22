# XDNA-LLM 设计文档

Tiel Coder 35B-A3B（Qwen3.6 系 MoE）在 AMD XDNA2 NPU 上的低功耗推理引擎。

## 1. 目标与约束

- **目标机型**：AMD Ryzen AI 9 H 365（Strix Point），XDNA2 NPU（50 TOPS，硬件 BFP16/MX 支持），32 GB **LPDDR5x-8000 实测跑满**（Micron 4×8GB，聚合理论 128 GB/s），**NPU 路径上限 ~65 GB/s**（NPU 侧 NoC 配额限制，非内存限制，见 §5）；>65 GB/s 只能靠 NPU+CPU/iGPU 异构聚合（plan B，与低功耗目标冲突）
- **核心动机：低功耗**。纯 NPU 路线，decode 时 CPU 大部分空闲，整机目标 15–25W（CPU 推理约 45–55W）
- **首发模型**：Tiel-Coder-35B-A3B（权重源：ornith-ai/Ornith-1.5-35B-A3B BF16，MIT；两者权重逐比特一致，Tiel 仅换 Sharp 模板）
- **首发范围（2026-10-22 用户定）**：**不做视觉塔**（mmproj 路径留接口不实现）；**发布不着急**（质量优先于时间表）；**首发只支持 Strix Point**（XDNA2 4 列 32 tile；Strix Halo/Krackan 后续再验）
- **真实架构（2026-10-21 从 config.json 实证，比此前假设更复杂）**：`qwen3_5_moe` = **Qwen3-Next 式混合架构**：
  - 40 层，**3:1 线性注意力（Gated DeltaNet 式：conv4 + 16K/32V heads × 128d，SSM 态 fp32）: 全注意力**（full_attention_interval=4）→ **KV cache 只存在于 10 个全注意力层**（这就是 262k 上下文 KV <5 GB 的原因）
  - MoE：**256 专家 / top-8 / moe_intermediate 512**；hidden 2048；head_dim 256，16 attn heads；attn_output_gate=true
  - MTP：1 层（mtp.layers.0.*，含 shared_expert），权重在 BF16 仓库内
  - 多模态：含视觉塔（mmproj 903 MB，可选路径）
  - 引擎含义：decode 的线性注意力是状态更新（小 GEMV+elementwise），全注意力只占 1/4 层；prefill 线性注意力走 chunk-wise matmul。llama.cpp 的 qwen3-next 支持是直接参照

## 2. 核心设计判断

| 阶段 | 计算形态 | 瓶颈 | 结论 |
|---|---|---|---|
| Prefill | 大 GEMM | 算力 | NPU 50 TOPS 友好 |
| Decode（无 MTP） | GEMV batch=1 | DRAM 带宽 | 90 GB/s ÷ bytes/token 决定上限 |
| Decode（有 MTP） | 验证步 = 小 GEMM | 半算力半带宽 | NPU 利用率甜区，吞吐主引擎 |

- **MTP 投机解码是一等公民**：Tiel Coder 有官方 MTP 变体；验证步把"一遍权重 1 token"变成"一遍权重验证 k 个、接受 ~1.8 个"
- 反直觉洞察：**lm_head 每 token 全量读**，8-bit 下占 decode 带宽近 30%；专家占总参数 90% 但只占 decode 带宽约一半 → 专家为存储负责（往死里压），稠密部分为带宽负责（3–4 bit 即可）

## 3. BAA 容器格式族（替代 IQ3_XXS 的自有格式）

**命名约定（2026-10-21 用户定）**：模型文件后缀 **`.baa`**（）= 容器格式，厂商中立；**量化方案按平台命名**，XDNA 上为 **XBF3**（3-bit profile，类比 llama.cpp 的 Q3/IQ3 之于 GGUF），将来 Intel 平台为 IBF3 等。即：同一个 .baa 容器，内部 tag 标识量化方案（XBF3/IBF3/…），后端按 tag 选 kernel。

**为什么弃用 IQ3_XXS**：非均匀网格码本 + LUT 反量化为 CPU 向量指令设计；AIE-ML 无高效 gather/LUT，上进 NPU 必须软件反量化，每 token 多付一遍 CPU 读写（~0.5 GB/token 隐形带宽税），抵消格式体积优势。

**BAA 设计原则**：格式即 NPU MAC 输入，解码路径零反量化；位宽和 scale 层级是可调旋钮，kernel 按位宽参数化。

### XBF3 profile（3-bit，方案 A 采用）

```
superblock(256 权重): [BF16 scale]            ← 0.0625 bpw
  权重本体: 3-bit 补码，8 权重打包 3 字节        ← 3.0 bpw
──────────────────────────────────
专家位宽合计 3.0625 bpw（与 IQ3_XXS 同体积）
```

- 无 sub-block micro-scale：Hadamard 旋转将 block 内权重摊成近似 i.i.d. 高斯，局部 scale 的边际收益被旋转替代（如实测不足，可加回 4-bit/32 权重的 micro-scale，+0.125 bpw）
- attention / lm_head：旋转后 4-bit 同构格式；router / embed：8-bit
- 量化管线：Hadamard 旋转（融合进 RMSNorm/相邻矩阵，推理期零开销）+ AWQ/GPTQ 式激活校准
- **与 IQ3_XXS 的关系**：信息论上非均匀码本对高斯源有 0.2–0.3 bit 效率优势，我们放弃码本（NPU 跑不了 LUT），差距由旋转 + 校准补回；体积持平，带宽路径反超

### 档位（同族，仅元素位宽不同）

| 档 | 专家 | 全模型体积 | 质量预期 | 状态 |
|---|---|---|---|---|
| **A（先跑）** | 均匀 3.0625 | ~14 GB | ≥ IQ3_XXS | 默认目标 |
| B | 热专家 3.19 / 冷专家 2.19（频率分级），均 ~2.6 | ~12 GB | 介于 IQ2–IQ3，需实测 | A 达标后试验 |
| C | 均匀 2.19 + LoRA 纠错 | ~10.1 GB | 对标 QuIP# 2-bit 外推，风险大 | 仅实验 |

- B 档依据：专家路由频率分布不均（PowerInfer 热/冷专家思想），压缩建立在 MoE 统计特性上；不剪枝，每专家设质量下限
- 已拒绝路线：格码本/向量量化（QuIP# E8、AQLM —— LUT 不友好）；熵编码（ANS 解压吃 CPU，违背低功耗目标）；全模型三值化（PTQ 掉点不可控，仅冷专家可考虑）

## 4. 架构（借鉴 llama.cpp 骨架，重写实现）

**硬约束：可移植性（2026-10-21 用户要求）**——架构必须便于后续支持其他 NPU（Intel AI Boost / Qualcomm Hexagon / Apple ANE / 未来 XDNA 代际）。落法三条缝：

1. **Backend 接口是唯一的可移植缝**：ggml 式后端注册（llama.cpp 已证明：CPU/CUDA/Metal/Vulkan 同构）。后端只实现：buffer 分配、布局转换、算子 kernel、异步执行、能力查询（支持的位宽/block 大小/对齐）。计算图 IR 保持厂商中立，XDNA2 的 tile 数、block 尺寸等假设禁止渗入 IR
2. **权重格式分层**：设备中立的 `.baa` 容器（块量化张量 + 元数据 + 方案 tag）+ 每平台量化方案（XBF3 = XDNA2，将来 IBF3 = Intel）。方案决定 kernel 选择，容器决定加载路径——类比 GGUF 元数据 vs llama.cpp 各后端的 buffer 转换
3. **必然留在厂商侧的**：kernel 工具链（XDNA 用 cardano/MLIR-AIE，别家各用各的 SDK）、xclbin 类产物、内存模型（共享 DRAM vs PCIe 设备）

纪律：M2' 只实现 XDNA2 后端，但接口按"会有第二个后端"来设计；接缝的正确性在接入第二个后端时才真正验证。

```
前端：tokenizer / chat 模板 / OpenAI 兼容 API
调度：采样、MTP 投机循环、分页 KV cache（int8）
计算图 IR：ggml 式 tensor + 算子图；router→scatter→expert→gather 在图层表达
Backend：
  XDNA2（主力）：expert FFN、attn 投影、MTP head、norm；cardano/MLIR-AIE kernel + XRT
  CPU（配角）：embed 查表、router top-k、采样、fallback
权重常驻内存，NPU/CPU 零拷贝映射
```

## 5. 基准方法（无 CPU 基线路线）

- 基准 = **带宽 roofline**：有效带宽 ÷ bytes/token = tok/s 上限
- **带宽实测（2026-10-22 本机探针，m0/probe/RESULTS.md）**：读写合并 sustained **~60–61 GB/s**（turbo/performance 模式；default 低 ~10%）。4 列 ShimTile 即压满共享内存路径（8 列无收益）；≥32 MB 长连续块是必要条件（16 MB 带宽近腰斩）；1000 次长跑无热衰减。对照 xrt-smi validate：GEMM 51.3 TOPS
- **关键悬念（decode 的生死数字）**：探针是读写回环，**纯读带宽未分离**——若读写配额对半则 decode 纯读仅 ~30–33 GB/s，若读可独占总线则 ~60。需 MemTile sink 设计的纯读探针（ctrlcode txn 手工合成或 WSL2 mlir-aie）
- 工具链定案（2026-10-22）：**CI 编译 + 本机实测**——GitHub Actions（公开仓库免费无分钟限制）ubuntu + mlir-aie（Apache-2.0）编 kernel，本机 Windows XRT 上机实测；kernel 100% 自有源码、零 AMD 专有 blob（license 最干净）。**已放弃** Windows 本地 ctrlcode 逆向路线：能通 shim DMA 但计算 tile 写入导致过一次系统宕机，时间/稳定性成本不可接受（证据归档 m0/poc_archive/）
- 硬件事实：XDNA2 BFP16 = 每 8 元素共享指数；MemTile 带宽合计 240 GB/s 读（非瓶颈），瓶颈在 ShimTile→NoC→DRAM；片上 SRAM ~800 GB/s → XBF 数据流 = DRAM 压缩态流入 → 片上 unpack 至 int8/MX6 车道 → 原生 MAC
- 推论：B 档压缩暂搁置（质量优先）；MTP 为必需品；hybrid decode 定为**全量 turbo（含离电，22 W 用户判定可接受），eco 暂缓**，真实功耗待 turbo 拉满后实测（2026-10-21 用户决策）。待实测假设：NPU↔CPU 每 token 汇合的同步开销（M3' 实测）、turbo 边际功耗 +5–10 W 为估算值（引擎落地后实测）
- **功耗模型（2026-10-21 修正）**：decode 带宽受限 → NPU 动态功耗仅 1–2 W（destevez xrt-smi 实测 1.146 W @ 65 GB/s 流式）。整机 decode 墙插 ~12–18 W，大头是 DRAM（4–6 W）+ 平台底噪 + uncore 的**固定成本**。推论：turbo 混部（CPU 专家流）的边际功耗仅 +5–10 W（CPU 核心动态功耗），能效损失 ~10% 换 ~65% 速度；eco 省的是 CPU 唤醒的边际功耗而非 NPU 本体
- 带宽探针（纯 DMA 流式）归属 **M0** 并作为 baseline：先于 M1' 完成，直测理论上限而非经模型间接推算
- FastFlowLM 实测锚点：GPT-OSS 20B @ 19 tok/s（Ryzen AI 7 350），能效比 iGPU 低 67× / CPU 低 223×——低功耗路线被独立验证
- 外部锚点：Tiel Coder 已发表成绩（Mac MLX oQ4 ~144 tok/s）做 sanity check

## 6. 质量验收门（每档实测，不达标拧位宽旋钮）

- PPL（WikiText-2 + 真实代码语料）：相对 BF16 锚点劣化 < 3%
- HumanEval / MBPP：下降 < 1 pp（PPL 对代码能力不敏感，必须看任务指标）
- 路由一致性：量化前后 top-8 专家重合率 > 95%（MoE 特有探针，暴露 router 漂移）

**v0 实测证据（2026-10-22，reports/xbf3_v0_report.md）**：XBF3 专家权重 SNR 13.78→14.05 dB（旋转后），收敛到 3-bit 高斯源理论极限窄带；旋转的价值在尾部（最坏张量 +2.09 dB，block max 误差 0.57→0.30），"旋转替代 micro-scale"假设获权重域支持。对照公开经验（Q3_K 档 PPL 劣化通常 <2%）谨慎乐观。**未覆盖风险**：真实激活离群通道（v1 AWQ 校准）、端到端 PPL、线性注意力层敏感性。附带发现：存在近死专家（L0.e167 超半数 block 全零），利好 B 档冷热分级。全模型离线量化外推 4–6 小时（CPU，通宵量级）。

## 7. 许可证

- 引擎本体：**Apache-2.0**（自研，建议）
- Tiel Coder 权重：MIT（继承 Ornith-1.5），XBF 转换件可直接再分发
- MLIR-AIE/IRON、XRT：Apache-2.0；XRT 运行时 DLL 可随包分发
- llama.cpp：MIT，可借鉴/引用（保留版权声明）
- Ryzen AI SW / Vitis AI EP：AMD 专有 EULA，仅开发机，不进分发包
- 量化校准语料：eaddario/imatrix-calibration（MIT）

## 8. 路线图

- **M0**：工具链（Ryzen AI SW / XRT / cardano+peano）+ **带宽探针（纯 DMA 流式，直测理论上限，baseline）** → 标定本机 NPU 带宽
- **M1' 量化实验室**：旋转 + 校准 + XBF 打包工具链，PPL/HumanEval harness，出档位-质量曲线
- **M2'**：XBF 的 NPU kernel（单 expert GEMV 打满带宽）——✅ 收官结论：管线 37.6 GB/s（8 列）；**3-bit 计算墙判定结构性**（车道错配 + 无 LUT + 无逐车道移位，v7 掩码选择路线实测否定）；**prefill 不受影响**（unpack 被 batch 摊薄 64×）
- **M3'**：全模型 decode + KV cache + turbo 分流
- **M4'**：MTP 投机解码
- 风险排序：M1' 量化质量 > M2' kernel 学习曲线 > 驱动/工具链成熟度

## 9. 架构级修正（2026-10-22，M2' 实证驱动）：按阶段分工

- **prefill → NPU**：3-bit unpack 成本被 prompt batch 摊薄（每权重 unpack 一次复用 M 次），50 TOPS 满阵列，预期 1500–2500 tok/s
- **decode → CPU AVX-512（主力）**：3-bit LUT 反量化是 vpermb 主场（IQ3_XXS 的设计原点），Zen5 无车道错配问题；预期 ~50 GB/s → 裸 ~35 tok/s ×MTP ~63
- NPU decode 仅可选增强：4-bit 部件（attn 投影/lm_head 车道对齐，NPU 算得动）+ CPU 3-bit 专家的混合 decode，理论 ~47 tok/s，M3' 后期实测决定
- 功耗预期修正：decode ~25–30 W（4 大核 + DRAM），仍远低于纯 CPU 45–55 W；纯 NPU decode 的 12–18 W 目标放弃
- turbo 含义更新：不再是"NPU+CPU 分流专家"（NPU 算不动 3-bit 专家），而是 decode 期 4-bit/3-bit 部件的 NPU/CPU 分工
