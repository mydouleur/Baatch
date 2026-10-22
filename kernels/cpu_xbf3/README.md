# cpu_xbf3 — CPU 后端 XBF3 GEMV（AVX-512，Zen5）（M3'-1）

decode 主力 kernel（阶段分工：prefill→NPU，decode→CPU）。XBF3 3-bit 打包权重流式 GEMV。

## 变体

- **lut**（`gemv_rows_lut4`）：vpermb 把 24-bit 组聚进 32-bit 车道 → vpsrld 提取 3-bit
  → vpermps 16 项 LUT（8 项×2 复制，免 vpandd）→ fmadd。x 按相位预排（xp_lut.bin）。4 行交错 ILP。
- **vnni**（`gemv_rows_vnni`）：W3A8 offset-binary（field=q+4），vpermb + srli/and×4 + or/slli
  打包 4 相位进 32-bit 车道 → vpdpbusd ×4 独立累加器。x 预量化 int8 相位重排（xq_vnni.bin）。
- **vnni2**（`gemv_rows_vnni2`）★推荐：vpermb 把 8 组 3B 聚进 8 个 qword 车道 →
  **vpmultishiftqb** 按位偏移 [0,3,…,21] 一次提出 8 字段/组 → and 7 → 字节即 field 且
  **值序天然连续**（qword j 的 8 字段 = 组 j 的 8 值 → dword 2j/2j+1 连续覆盖）→
  xq 免重排（xq_plain.bin）→ vpdpbusd ×4。每 64 值仅 6 条向量指令（vnni 的 1/4）。
- **bp**（已弃）：位平面 vpand + vptestmb 掩码 → 掩码 vaddps。结构性慢（~4 GB/s/核）。

数据流：128 个独立随机专家拼接流（49MB > 24MB L3，防缓存假象），x 常驻 L1/L2。

## 实测（Ryzen AI 9 H 365，E=128 49MB 真 DRAM 流，iters=50，agg=聚合 wall）

| 变体 | 1 大核 | 2 大核 agg | 4 大核 agg | per-thread-best |
|---|---|---|---|---|
| lut | 9.5 GB/s | 19.6 GB/s | 30.8-34.7 GB/s | ~10-11 GB/s |
| vnni | 9.6 GB/s | 18.5 GB/s | 31.4 GB/s | ~9.8 GB/s |
| **vnni2** | **21.1 GB/s** | **39.0 GB/s** | **49.0 GB/s** | **~21.7 GB/s** |

- 目标线：fp16 杠杆 2 大核 ≥20（lut 已达）→ 跳过；VNNI 杠杆 2 大核 ≥28 → **vnni2 39 达标（+39%）**
- 每行周期数（vnni2，21.7 GB/s/核）：196 B/行 → 111M 行/s ≈ **~46 cyc/行 @5.05GHz**，
  即每 64 值窗口（6 指令）~5.7 cyc——仍非端口吞吐极限，是 load→vpermb→multishift→dpbusd 链延迟墙
- E=4（1.5MB，L2 常驻）对照：vnni2 单核 21.4 ≈ DRAM 流 21.1 → **vnni2 是纯算力墙**；
  lut 同理（11.2 L2 ≈ 11.1 DRAM）
- 大核 = 逻辑核 0/2/4/6（LP 扫描实证：偶数 LP0-7 ~大核，LP8+ Zen5c ~5 GB/s）

## 试过但 slower 的方向（勿重复）

- **合并 load**（2×48B + 双 vpermb 高/低 8 组）：独立链 4→2，−16%（16.2 vs 19.2 GB/s）
- **车道级 fp 修正项**（corr=4·Σxq 逐车道，免每 sb 水平归约）：多 12 指令/sb，−7%（19.7 vs 21.4 L2）
- **2 行交错**（配 corr 方案）：`__m512*` 指数组迫使累加器落栈，−20%；要做须全显式寄存器写法
- **vnni（srli/and/or 打包链）**：提取链 16+ 指令/pg 主导，dpbusd 的 4×MAC 密度吃不上，仅与 lut 持平

## 数值校验

lut / vnni / vnni2 均 PASS（对照 tools/quant/xbf3.py 参考反量化；容差带 rel>1e-3 且
abs>1e-3 双门）。vnni/vnni2 max_rel=2.4e-3（xq int8 量化残差；y_ref_vnni 用同一 xq 计算）。
lut max_rel ~8e-2 只出现在 |ref|<1e-3 的近零元（fp32 归约序差异），bad=0。
注意：gen_data 用快速量化路径（absmax/4，跳网格搜索）——带宽测试不关心量化质量；
参考 y_ref 用 bf16 舍入后的 scale 计算（与 kernel 消费一致）。

## 构建/运行

```
# 数据（E=128 防 L3 假象；E=4 可做纯算力对照）
python gen_data.py --out data_cpu128 --experts 128
# 编译（MSVC；vsenv.sh 见 m0/probe/；Git Bash 下用 -arch:AVX512 防路径转换）
cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus -arch:AVX512 -utf-8 gemv_avx512.cpp -Fe:gemv_avx512.exe
# 跑
gemv_avx512.exe --variant vnni2 --threads 2 --iters 50 --dir data_cpu128 --experts 128
gemv_avx512.exe --variant vnni2 --threads 1 --lp 8 ...   # 单线程绑指定逻辑核
```

## 结论（M3'-1 0b 深挖收官）

- **推荐进引擎：vnni2**（vpmultishiftqb 提取 + vpdpbusd，xq 免重排）。2 大核 39 GB/s、
  4 大核 49 GB/s；数值 PASS。
- 剩余空间：单核 ~21.7 GB/s 对应 ~5.7 cyc/窗口，理论上 shuffle/FMA 端口混合应能到
  ~2×；需全显式寄存器的 2-4 行交错打破链延迟墙（估 +30-50%），但 decode 带宽已主要
  受 DRAM 聚合限制（4 核 49 GB/s 时 per-core 降至 18.5），继续抠单核意义递减。
- decode bytes/token ≈ 1.34 GB（XBF3）：2 大核 39 GB/s → **~29 tok/s**；4 大核 49 GB/s
  → **~36 tok/s**（vs 任务启动时 lut 基线 2 大核 ~20 GB/s → 15 tok/s，提升 ~2×）。
