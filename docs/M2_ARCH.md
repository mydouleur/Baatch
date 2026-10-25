# Baatch 推理引擎架构设计（当前修订）

> 本文替代 2026-10-22 的 XBF3/MIE 初期方案。量化与后端分工以 `DESIGN.md` §10 为准。

## 1. 目标与模型事实

- Ryzen AI 9 H 365，XDNA2，32GB LPDDR5x-8000。
- 40 层：30 GDN + 10 全注意力；256 专家 top-8；hidden 2048；一层 MTP。
- GDN 使用 conv/SSM 状态，只有全注意力层需要 KV cache。
- 核心目标是低功耗，性能与功耗必须实测，不沿用早期 45/80 tok/s 预算。

## 2. 容器与量化

- `.baa` 为厂商中立容器，目录保存逻辑 shape，flags 描述物理布局。
- **XBF4 routed experts → NPU**。
- **XBF2 routed experts + imatrix → CPU**。
- 共享专家、attention/GDN 投影、MTP 头和 lm_head → NPU。
- embedding 查表、router/top-k、采样和 fallback → CPU；其他部件只有在 CPU 综合性价比明显更好时才切到 CPU。
- 当前 absmax 是开发版，XBF2 imatrix 和 PPL 质量门尚未完成。

## 3. 后端与调度

```
serve/CLI（后续）
  → prefill / decode / MTP 调度
    → 模型数学与状态管理
      → XDNA2：XBF4 experts、共享专家、密集投影、lm_head
      → CPU：XBF2 experts、embedding、router、采样、fallback
```

- 后端负责 buffer、布局转换、异步 submit/wait、能力查询和缓存上限。
- 分发同时看 scheme、带宽、同步成本与功耗，不按“专家/非专家”机械分配。
- CPU/NPU 同一张量应共享磁盘格式；布局转换不得造成无界重复常驻。
- IR 不绑定 tile 数或 block 尺寸，物理形状只留在后端与打包器。

## 4. Decode

- XBF4 routed experts 在 NPU 执行，XBF2 routed experts 在 CPU 执行，MoE reduce 保持原语义。
- embedding 查表留在 CPU；router/top-k 和采样留在 CPU。
- 多形状切换、无效 padding、权重 BO 缓存和同步成本需要分别计量，不能只看 kernel GB/s。
- uniform 是减少切换的过渡实验，不是最终低功耗格式。

## 5. Prefill

- M=1 已闭环：GDN/attention 64-token 分块，prompt lm_head 只计算末 token。
- prefill 与 decode 共享 conv/SSM/KV 状态，65-token 全模型对拍一致。
- M>1 旨在复用权重 tile，但随机激活尚未闭合，未接入引擎。

## 6. MTP（后续）

- draft/verify/accept 必须复用同一分工和输出语义。
- GDN/KV 状态快照与回滚复用同一套设施，并可用于前缀缓存。
- 实测接受率与净加速，不引用文献值作为本机结果。

## 7. 当前差距与路线

1. XBF4 routed experts 的 NPU 数值与全模型验收。
2. 多形状切换、无效 padding 和缓存预算优化。
3. prefill M>1 与 decode/prefill 性能功耗基线。
4. MTP 与接受率验证。
5. XBF2 imatrix、PPL、代码任务和路由一致性。

历史 M0/M1/M2 探针数据仍有效，但早期 XBF3 专家分工与里程碑状态不再作为当前设计依据。
