# .baa 容器格式规范 v0.1

Baatch 引擎的模型容器。厂商中立；量化方案由方案 tag 标识（XBF3 = XDNA2/CPU 3-bit，XBF4 = 4-bit，未来 IBF3 等）。

## 总体布局

```
[文件头][张量目录][数据区]
全部小端。偏移量 64 位。
```

## 文件头（128 B 固定）

| 偏移 | 字段 | 类型 | 说明 |
|---|---|---|---|
| 0 | magic | 4B | `BAA1` |
| 4 | version | u32 | 容器版本（1） |
| 8 | arch | 32B strz | 架构标识，如 `qwen35moe` |
| 40 | tensor_count | u64 | 张量数 |
| 48 | dir_offset | u64 | 张量目录偏移 |
| 56 | meta_offset | u64 | 扩展元数据区（JSON：超参数、rope sections、模板引用等） |
| 64 | meta_len | u64 | 元数据长度 |
| 72 | data_offset | u64 | 数据区起始（256B 对齐） |
| 80 | flags | u64 | bit0: 含 MTP 头；bit1: 含视觉塔；其余保留 |
| 88 | 保留 | 40B | 全零 |

## 张量目录项（每条 96 B）

| 偏移 | 字段 | 说明 |
|---|---|---|
| 0 | name_offset | u32，名字符串池偏移 |
| 4 | scheme | u8：0=BF16 原样，1=INT8，2=XBF3，3=XBF4，4=XBF2，5=F32 原样 |
| 5 | dtype_hint | u8：原始逻辑类型 |
| 6 | ndim | u8 |
| 7 | 保留 | u8 |
| 8 | shape | u64×4（最多 4 维，不足补 1） |
| 40 | data_offset | u64，相对数据区 |
| 48 | packed_size | u64 |
| 56 | aux_offset | u64（旋转矩阵/校准参数引用，无则 0） |
| 64 | 保留 | 32B |

## XBF3 张量数据（3.0625 bpw）

```
per 行（K 维连续）：K/256 个 superblock
superblock(256 权重) = [2B BF16 scale][96B 位组数据]
位组：连续 pack3 跨字节打包（vnni2 kernel 的 vpmultishiftqb 布局；M3'-1 实测位平面版 ~4 GB/s/核，弃用）
行起始 64B 对齐（行间可含 padding）
```

## XBF4 张量数据（4.0625 bpw）

```
superblock(256) = [2B BF16 scale][128B nibble 数据]（lo|hi<<4 字节内打包）
布局兼容 TileFuse 预平铺（NPU 路径）；CPU 路径行主序
```

## XBF2 张量数据（2.0625 bpw）

```
superblock(256) = [2B BF16 scale][64B 位组数据]
位组：连续 pack2（4 权重/字节，vnni LUT kernel 的 vpmultishiftqb 布局）
行起始 64B 对齐
量化工艺：imatrix 加权舍入（组内 scale 小网格搜索最小化加权误差）+ Hadamard 预旋转（旋转矩阵经 aux_offset 引用）
```

## 专家粒度位宽（v1+ 不变量）

- 路由专家张量（`ffn_*_exps`）在 .baa 内**按专家子块存储**，每个专家子块带独立 scheme 字节；引擎分发禁止按层假设统一位宽
- 冷/热配比是打包期决策，容器与引擎均无感知；改配比 = 重打包 + 改 NPU/CPU 归属表，引擎代码不动

## 不变量

- 目录按 name 排序，查找可二分
- 数据区可 mmap，所有张量零拷贝映射
- 元数据 JSON 必含：`hyperparameters`、`rope_sections`、`chat_template_ref`、量化校准信息（语料、clip 策略、旋转矩阵引用）
- MTP 头张量命名前缀 `mtp.`；视觉塔前缀 `vision.`
