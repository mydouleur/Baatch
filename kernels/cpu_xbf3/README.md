# cpu_xbf3 — CPU 后端 XBF3 GEMV（AVX-512，Zen5）（M3'-1）

decode 主力 kernel（阶段分工：prefill→NPU，decode→CPU）。XBF3 3-bit 打包权重流式 GEMV。

## 变体

- **lut**（`gemv_lut_sb`）：vpermb 把 24-bit 组聚进 32-bit 车道 → vpsrld+vpandd 提取 3-bit
  → vpermps 8 项 LUT（fp32 {0,1,2,3,-4,-3,-2,-1}）→ fmadd。x 按相位预排（xp_lut.bin）。
- **bp**（`gemv_bp_sb`）：pack3_bp 位平面 → vpand + vptestmb 提掩码 → 掩码 vaddps 选择性累加
  （平面权重 1/2/−4 预折进 xp_bp.bin）。

数据流：64/128 个独立随机专家拼接流（防 L3 假象），x 常驻 L1，y 每行一次 reduce_add。

## 实测（Ryzen AI 9 H 365，E=128（49MB > 24MB L3，真 DRAM 流），iters=40，per-thread-best）

| 变体 | 1 大核 | 2 大核 | 4 大核（聚合 wall） |
|---|---|---|---|
| **lut** | **8.0 GB/s** | **8.2/核（agg 13.8）** | **7.5/核（agg 23.9）** |
| bp | 3.9 GB/s | 4.0/核（agg 7.7） | 4.0/核（agg 13.7） |

- 单核 E=4（1.5MB，全 L2）= 7.3 GB/s ≈ E=128 DRAM 流（8.0）→ **纯算力墙，非内存墙**
- 每行周期数（lut）：~8 GB/s ÷ 196 B/行 ≈ 41M 行/s ≈ **~105 cyc/行**（4.4GHz 估）
- 大核 = 逻辑核 0/2/4/6（LP 扫描实证：0-7 偶数位 ~7.8 GB/s，LP8+ Zen5c ~5 GB/s）
- bp 变体慢的根因：掩码路径每 (平面,相位) 要 and+test+2×masked-add，且掩码寄存器链是串行的

## 数值校验

两变体均 PASS（对照 tools/quant/xbf3.py 参考反量化，max_rel ~8e-3 内 bad=0；容差带
rel>1e-3 且 abs>1e-3 双门）。注意：gen_data 用快速量化路径（absmax/4，跳网格搜索）——
带宽测试不关心量化质量；参考 y_ref 用 bf16 舍入后的 scale 计算（与 kernel 消费一致）。

## 构建/运行

```
# 数据（E=128 防 L3 假象；E=4 可做纯算力对照）
python gen_data.py --out data_cpu128 --experts 128
# 编译（MSVC；vsenv.sh 见 probes/bandwidth/）
cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus -arch:AVX512 -utf-8 gemv_avx512.cpp -Fe:gemv_avx512.exe
# 跑
gemv_avx512.exe --variant lut --threads 2 --iters 40 --dir data_cpu128 --experts 128
gemv_avx512.exe --variant lut --threads 1 --lp 8 ...   # 单线程绑指定逻辑核
```

## 瓶颈分析与目标差距

- 目标 2 大核 ≥24 GB/s **未达**（实测 ~14-16 GB/s）；4 大核 ~24-30 GB/s 达标。
- 每行 ~170 条向量指令 @ IPC ~1.75 → ~96 cyc/行；shuffle 端口（vpermb/permps）与 FMA
  各占约 1/3，提限需每条指令覆盖更多值——3-bit 对 32-bit 车道的错配在 x86 同样存在，
  但 vpermb/vpermps LUT 比 NPU 上的处境好一个量级。
- 下一步（若继续）：fp16 x + vpermw LUT + vpdp… 半精度双车道点积理论上再 ~1.5×；
  或接受现状，把带宽缺口放到"减少 decode 字节数"（更好量化）一侧解决。

## 对 decode tok/s 预期的修正

decode bytes/token ≈ 1.34 GB（XBF3）。2 大核 16 GB/s → ~12 tok/s；4 大核 24-30 GB/s
→ ~18-22 tok/s。比"NPU 60 GB/s → 45 tok/s"的原预期低——CPU 路线需靠混合精度 MTP
或更高核数补偿，或在架构评审里重新评估。
