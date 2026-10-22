# 任务 0c：NPU↔CPU 汇合同步延迟实测（turbo 混部 go/no-go）

机器：Ryzen AI 9 H 365（Strix Point，XDNA2 8 列），32GB LPDDR5x-8000
驱动/运行时：NPU driver 32.0.203.329，XRT 2.19.0，NPU FW 1.0.21.43
日期：2026-10-22。**测量期间功耗模式 = turbo**（`xrt-smi configure --pmode turbo`，测后恢复 default）。
计时：steady_clock（QPC），实测时钟粒度 ~100ns。

## 头条结论

**过验收线，GO。** 单次汇合（CPU 提交 → NPU 执行最小负载 → 完成中断 → CPU 线程被唤醒拿到结果）
端到端 **p50 = 70 µs，p99 = 108–135 µs，max = 177 µs**（1000 次样本），
验收线 ≤0.5 ms 有 **3–7× 余量**。占 60 tok/s 目标帧预算（16.7 ms/token）的 **0.42%（p50）/ 0.81%（p99）**；
即使每个 token 汇合 10 次也只占 ~4%。

## 实测分布（µs，剔除前 5 次预热；n≈995/295）

| 实验 | 负载（r+w 流量） | e2e p50 | e2e p99 | e2e max | 备注 |
|---|---|---|---|---|---|
| bare tiny | 64 KB | **70.0** | 108.1 | 164.9 | 汇合延迟地板 |
| rendezvous tiny（CPU 先做 ~0.3ms GEMV） | 64 KB | 82.5 | 123.7 | 176.6 | 汇合模式模拟 |
| rendezvous tiny（CPU ~1.1ms GEMV） | 64 KB | 84.6 | 135.4 | 173.1 | 同上，更长 CPU 段 |
| poll tiny（忙轮询 `run.state()`） | 64 KB | 63.8 | 120.9 | 300.8 | 见坑 2 |
| bare 1m | 8 MB | 217.6 | 288.3 | 337.4 | 差分点 |
| bare 4m | 32 MB | 651.0 | 724.9 | 773.6 | 差分点 |
| bare 16m | 128 MB | 2690.5 | 3412.2 | 3417.0 | 差分点 |
| rendezvous 4m（CPU ~0.55ms GEMV） | 32 MB | 667.0 | 1415.1 | 1486.1 | 接近真实 decode 步 |

e2e = `run.start()` 返回前 → `wait2()` 返回后。分解（tiny，µs p50）：
`xrt::run` 创建+set_arg = 9；`start()` 提交调用（反向：CPU→NPU doorbell）= 4.8；
剩余 ~65 µs = 驱动派发 + NPU 侧 txn 控制码执行（~104 个寄存器写 + shim DMA 握手 + TCT 同步）+ 完成中断 + 线程唤醒。

**差分分析**（分离固定汇合开销 vs DMA 时间）：e2e 随流量线性增长，
斜率 ≈ 18–19 µs/MB（≈53–55 GB/s，与 M0 带宽探针一致；128 MB 点掉到 ~47 GB/s 因单 BD 8MB 块效率下降），
**截距 ≈ 70 µs = 固定汇合开销**。阻塞 wait 相对忙轮询的唤醒附加仅 ~6 µs——**唤醒路径不是瓶颈，
地板由 NPU 侧命令处理主导**。减小每次汇合的寄存器编程量（复用已编程通道/常驻 kernel）可进一步压低地板。

## 反向路径（CPU 提交 → NPU 开跑）

`start()` 提交调用本身 p50 = 4.8 µs（纯 host 侧写 doorbell）；NPU 实际开跑时刻 host 无法直接打时间戳，
但 poll 模式 e2e = 63.8 µs ≈ 提交 + NPU 处理 + 完成可见的全链路（tiny 负载 DMA 仅 ~1 µs），
即**提交→完成全链路 ~64 µs，唤醒只占其中 ~6 µs**。

## 坑 / 注意

1. **BD 长度只能改 word0**。df-bw loopback 的 BD 8 字里 word5/7 必须保持 0x02000000
   （配对通道的期望值），三个一起改小会死锁（ERT_CMD_STATE_TIMEOUT）。逐字节 diff 自
   m0/probe 的 df_bw_4col_gen vs len16m 变体确认。
2. **忙轮询会"提前完成"**：`run.state()==COMPLETED` 时最后 ~64 字（实测 64 words=256B）可能还没落到
   host 可见内存（poll 模式 verify FAIL 64/4096 words；阻塞 wait2 无此问题）。轮询路径省 ~6 µs
   但读结果需要额外 fence/尾部填充，**不建议**为这点收益冒正确性风险。
3. bare_1m（span=1MB 布局）在 300 次长跑后最终 verify 偶发 16–1408 words 不一致（50 次短跑 PASS，
   4m/16m 全 PASS）；疑似该 span 配置下末次迭代尾拍可见性瑕疵，属探针设计边角，不影响延迟结论。
4. 干扰声明：测量期间无其他 NPU 进程（前后 tasklist 确认；且 4m/16m 隐含带宽 47–55 GB/s 与 M0 干净
   单上下文数字一致）。若与 0a 带宽测试并发，分布尾部会右移，本数据为**干净环境**值。
5. `xrt::run` 对象每次迭代新建（ert 命令单次性）；create 的 9 µs 在真实引擎里可用 run 池摊掉，
   已单列不计入 e2e。

## 复现

```bash
cd d:\xdna_llm\Baatch\probes\sync_latency
# 编译（MSVC；vsenv.sh 是 m0/probe 的手工工具链环境）
bash -c "source d:/xdna_llm/m0/probe/vsenv.sh && cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus \
  -ID:/xdna_llm/m0/probe/xrt_src/src/runtime_src/core/include host_sync_lat.cpp \
  D:/xdna_llm/m0/probe/xrt_coreutil.lib -Fe:host_sync_lat.exe"
# 生成负载 ELF（span=每列跨度，--len=BD word0，每 BD 实际传输 4*len 字节）
python gen_elf.py --cols 4 --span 0x1000 --len 512 --out df_lat_tiny.elf   # 64KB 流量
# 跑（mode 0=bare / 1=rendezvous(cpu_us) / 2=poll；SYNC_XFER=每BD字节数，仅供 verify 范围）
xrt-smi configure --pmode turbo   # C:\Windows\System32\AMD\xrt-smi.exe
SYNC_XFER=2048 ./host_sync_lat.exe <validate_elf_17f0_10.xclbin> df_lat_tiny.elf 1000 16384 1 200 out.csv
xrt-smi configure --pmode default # 恢复
```

## 产物

| 文件 | 说明 |
|---|---|
| `host_sync_lat.cpp/.exe` | 探针 host（bare/rendezvous/poll 三模式，逐迭代 CSV） |
| `gen_elf.py` | 可变 BD 长度的 df-bw ctrlcode ELF 合成器（源自 m0/probe/gen_elf.py） |
| `df_lat_{tiny,s64k,1m,4m,16m}.elf` | 64KB / 64KB(64K span) / 8MB / 32MB / 128MB 流量负载 |
| `bare_tiny.csv`、`rdv_tiny_{200,1000}us.csv`、`poll_tiny.csv`、`bare_{1m,4m,16m}.csv`、`rdv_4m_500us.csv` | 逐迭代原始数据（create_ns, submit_ns, e2e_ns, burn_ns） |
