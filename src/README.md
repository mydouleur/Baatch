# src/ — Baatch 引擎本体（M2' 起）

## 当前分工（2026-10-25 实测修订）

- XBF2/XBF4 routed experts 默认走 CPU；`--npu-xbf4-experts` 仅作实验对照。
- 共享专家与 routed experts 默认走 CPU；attention/GDN 大投影 decode 与 MTP 头走 NPU；lm_head 在 decode 默认走两核 CPU，在 prefill 末尾走 NPU。
- GDN 状态递推、embedding、router/top-k、采样和 fallback → CPU；prefill 大投影与 out_proj 也默认两核 CPU，`NPU_PROJ=1`/`NPU_OUT_PROJ=1` 保留对照。65-token CPU projection 3321ms、NPU 3824ms；第 0 层 projection rel_l2=0.00438。
- GDN qkv+z、attention q+k+v 默认共同复用 N12288；decode 暖态约 75–100ms/token，65-token prefill 约 6–6.3s。

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

**decode 速度现状**（v1 XBF24 全量打包后，2 大核暖态实测）：**~70–130ms/token ≈
8–14 tok/s**（min 69.5ms ≈ 14.4 tok/s；方差来自页缓存与未向量化的标量段）。
较 v0（专家 XBF3+其余 BF16）的 8.7 tok/s 提升 ~1.3–1.5×，未到 20 tok/s：
每 token 字节 ~1.3GB（专家 XBF2/4 ~0.39GB + 密集 XBF4 ~0.6GB + lm_head 0.26GB + F32 小件）
→ DRAM 地板 ~33ms/token；差额的 2-3× 来自标量段（router softmax/norm/conv/门控）与
激活量化循环的未向量化。delta rule 已 AVX-512 化（~1.3× 总提速）。

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

## M3'-b 现状（2026-10-24，进行中）：NPU W4A8 进密集路径

- **NPU 后端**（`src/engine/npu.h`）：XRT 直调，形状库 `models/npu/k{K}n{N}.xclbin/.insts.bin`
  （本机全链 15s/个：1.4.4 python 出 MLIR → RAI 编好的 i8 kernel .o → aiecc.exe + xclbinutil）。
  密集预平铺张量（flags bit0）自动分发 NPU；flake（整块 0）校验+重试≤3+计数；
  run 超时（ERT timeout）= 挂起语义，不重试直接 CPU 兜底。
- **布局**：v1.1 容器 = 密集 XBF4 TileFuse 预平铺（zp=8 常数 ≡ 对称量化）+
  专家行主序 XBF2/XBF4；目录 offset7 为 layout flags（bit0=预平铺），规范见 BAA_FORMAT.md。
- **原形状库**（4 个，全部上机逐位 PASS）：
  k2048n8192（qkv/q_proj）、k2048n4096（z）、k4096n2048（out/o_proj）、k512n2048（shexp down）。
  每调用地板 ~0.1ms（小形状）/~0.26ms（8.5MB）。
- **v1.1 留 CPU**：k/v_proj、in_proj_a/b、router、shexp gate/up、norm、SSM、embed、
  lm_head、全部专家（XBF2/XBF4 行主序）。
- CPU 兜底：`cpu_mv_pretiled`（标量反解，仅 fallback/对拍用）。
- **混合 decode 已端到端跑通**（per-shape xclbin 版）：3450 次 NPU 调用、**flake=0**、
  首 token `<think>` 与 HF 一致；但 **600ms/token 不达标**，归因：
  (a) xclbin 切换 2.6ms/次 × 每 token 160 次形状交替（bench_switch.exe 实测：
  同形状连跑 0.43ms/call vs 交替 5.8ms/对）；
  (b) 预平铺密集张量的 CPU 兜底当时是标量慢路径。
- **事故记录**："单 xclbin + insts 热换"优化（4 形状 design.pdi 逐位一致所以理论可行）
  实测**挂死设备**（NPU 驱动僵尸态，进程不可杀，需重启恢复）——已回退，禁再用。
- **v1.2**：writer 默认只把 (8192,2048) 的 in_proj_qkv/q_proj 预平铺
  （每 token 40 次同形状调用、**零切换**）；其余密集张量回行主序走 CPU vnni4 快路径。
- 残余：flake 率实测（本进程 3450 运行 0 次，与 rai_explore 的 10-13% 口径不同——
  疑似与其 i8 kernel 快时序相关，我们引擎调用间隔长得多）。

## M3 密集主路径（2026-10-24，功能闭环，带宽调优中）

- 现有 writer `--npu-layout dense` 覆盖 qkv/q_proj、z、out/o_proj、k/v、共享专家投影和
  lm_head；embedding 保持行主序查表，router/小投影和 routed experts 留 CPU。
  XBF4 routed experts 的 NPU 路径已验收，但总成本高于 CPU，2026-10-25 起默认留在 CPU。
- 预平铺权重的物理 N 补齐到 2048，目录 shape 保持逻辑 N，补齐行权重为零。
  引擎选择补齐后的 NPU shape，只消费逻辑 N 个输出；CPU fallback 使用同一磁盘格式。
- k2048n2048 与 k2048n249856 上机逐位 PASS；lm_head 大矩形物理形状已闭合。
- NPU 超时后全局关闭本进程 NPU 后端；hw_context 由 Shape 持有，不再泄漏。
- 密集容器 16.08GB、259 个 NPU 张量，目录审计 PASS。GDN/attention/MoE 使用同
  XBF24 参考对拍 PASS；CPU 完整生成 OUT_IDS 保持 220,17。修复预平铺 CPU fallback
  奇数输出误用相邻 scale 的问题。
- NPU submit/wait 已拆分：GDN qkv 与 a/b、qkv 模式下 q 与 k/v、共享专家 down
  与标量门控可重叠。后端限制单未完成任务，不多 context 并发提交。
- 三次独立 32-token 混合 decode 输出完全一致；每次 8283 次调用均 flake=0，
  无 fallback/超时。当前多形状版约 0.8–1.0s/token，瓶颈是每 token 251 次调用及切换。
- 多 kernel xclbin 合并可运行，但同形状约 0.6ms、交替约 8ms/对，无收益；
  --expand-load-pdis 无效。full ELF 静态 sequence 可构建，但 XRT 2.19 创建
  context 时以 invalid stoul argument 拒绝，未进入执行；两路线暂不扩展。
- `--npu-layout uniform` 是过渡对照：除 lm_head 外统一物理 K4096/N8192（flags bit1），
  有效量化值不变、补齐区置零，目标先验证少量 xclbin 切换的收益；额外 padding
  流量不是最终格式决策。uniform 32-token 输出与 dense 一致、flake=0；performance
  暖态约 171–220ms/token，kernel 约 38GB/s。物理流量 4.73GB/token（dense 1.185GB），
  因此只作过渡，不作为低功耗最终格式。

## Prefill 与当前优化

- M=1 dense prefill 已实现：GDN/attention 64-token 分块、prompt lm_head 只算末 token、
  prefill→decode 状态衔接。MoE 按专家 gather→多 token GEMM→scatter，XBF4 合并两个
  128-half 后只做一次整数归约；T=1/2/8/65 的 XBF4/XBF2 direct 对拍全部 PASS。
  layer0 65-token MoE 21.71→19.05ms，变化 prompt 全模型 prefill 4936→4718ms。
- M>1 采用打包 A/C FIFO + kernel 内权重 tile 复用，A FIFO 深度 8 修复跨组覆盖；
  M2 两行共享一次 W4 解包和 B 加载。K2048/N12288 随机与独立行 PASS，1000 次均值
  0.678ms vs M1×2 1.012ms；N10240 0.619ms vs 0.760ms。K4096/N2048 仍确定性
  ERT timeout；通用 M4 数值 FAIL。M2 成对引擎实验在跨权重/激活时序下出现成对块
  损坏，最终 hidden rel_l2=0.528；新建 run 不能修复，接入已移除，默认保持 M1。
- M=1 K4096/N2048 已删除无效 ping-pong 并接默认资产：暖态 avg 0.332→0.221ms；
  65-token prefill 49.63→46.21s，输出一致、38193 次调用 0 flake。
- 默认使用两核单份匿名模型驻留（`BAA_MMAP=1` 回退）、SMT host（`MOE_HOST_SIBLING=0`
  回退）、66B 行缓存、XBF2 合并归约与 CPU lm_head。decode 约 53–70ms/token（14–19 tok/s），
  65-token prefill 5.87s，NPU 0 flake。常驻并发 XBF4 约 78GB/s、XBF2 约 61GB/s。
  prefill 后台预热 decode out/lm_head 行缓存，GDN 32 头两核并行；真实 65+64 的 decode
  中位 57.90ms、平均 58.71ms、p95 61.9ms，TTFT 5.88s；匿名模型启动读写约 12s。
  router 已两核化，NPU 每 Shape 复用 `xrt::run`。prefill `PROF=1` 覆盖 proj/state/out/MoE；
  attention 16 头两核化使 256-token state 总计减少 0.461s，65+16 IDs 一致、0 flake。
  `bench_moe` 另有 direct/scheduled、固定/随机 top8
  与 GEMV/同步/量化分项；4/8 routed 线程无稳定收益，保持两线程。
- CPU 激活量化、整数求和、偶奇/模4拆分已 AVX-512 化，单测与层对拍 PASS。
  GDN 状态更新与输出已融合，少读约 63MB/token；prefill 后 decode 实测最低约 67ms。
  当前约 1.78GB/token，CPU4+NPU 并发探针约 77GB/s；18ms 需约 99GB/s 且受逐层依赖限制。
- `--npu-layout fast` 过渡实验已停止，不再缩小既定 NPU 分工。balanced 也没有实质收益。
- `bench_moe` 支持 current/planned/batch/parallel 单层对照。planned 数值一致但暖态无显著收益；
  E4×2 batch 数值通过但 layer0 约 12.0ms；列并行 p4 每列持完整专家，数值通过、
  约 10.8ms，优于串行 batch 但仍慢于 current 约 9.1ms，均不接默认引擎。
  E6、长序列 p8、独立 BD＋去除专家间等待均挂起，禁止直接重试。
