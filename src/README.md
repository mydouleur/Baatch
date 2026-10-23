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
  专家行主序 XBF2/XBF4；目录 offset7 保留字节复用为 layout flags（bit0=预平铺）
  ——**规范出入已报主线**（BAA_FORMAT.md 未定义该字节的此用途）。
- **形状库**（4 个，全部上机逐位 PASS）：
  k2048n8192（qkv/q_proj）、k2048n4096（z）、k4096n2048（out/o_proj）、k512n2048（shexp down）。
  每调用地板 ~0.1ms（小形状）/~0.26ms（8.5MB）。
- **留 CPU**：k/v_proj、in_proj_a/b、router、shexp gate/up、norm、SSM、embed、
  lm_head（N=248320 不被 32 整除，不适配设计）、全部专家（XBF2/XBF4 行主序）。
- CPU 兜底：`cpu_mv_pretiled`（标量反解，仅 fallback/对拍用）。
- **混合 decode 已端到端跑通**（per-shape xclbin 版）：3450 次 NPU 调用、**flake=0**、
  首 token `<think>` 与 HF 一致；但 **600ms/token 不达标**，归因：
  (a) xclbin 切换 2.6ms/次 × 每 token 160 次形状交替（bench_switch.exe 实测：
  同形状连跑 0.43ms/call vs 交替 5.8ms/对）；
  (b) 预平铺密集张量的 CPU 兜底当时是标量慢路径。
- **事故记录**："单 xclbin + insts 热换"优化（4 形状 design.pdi 逐位一致所以理论可行）
  实测**挂死设备**（NPU 驱动僵尸态，进程不可杀，需重启恢复）——已回退，禁再用。
- **当前定案**（v1.2，待设备恢复后复测）：writer 只把 (8192,2048) 的
  in_proj_qkv/q_proj 预平铺（每 token 40 次同形状调用、**零切换**）；其余密集
  张量回行主序走 CPU vnni4 快路径。models/ornith35_xbf24.baa 需按 v1.2 重打包
  （写时被僵尸进程锁文件阻塞——重启后重跑 baa_writer_v1.py）。
- 残余：flake 率实测（本进程 3450 运行 0 次，与 rai_explore 的 10-13% 口径不同——
  疑似与其 i8 kernel 快时序相关，我们引擎调用间隔长得多）。
