# `split_qkvgate_gemma_rmsnorm_rope_kernel` 性能数据

- 测试脚本: `split_qkvgate_gemma_rmsnorm_rope_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `N8_Q2048_KV512_D128_R64` | 4.00 | 3.60 | 4.31 | 10 | AI_VECTOR_CORE |
| 2 | `N8_Q2048_KV512_D256_R128` | 4.20 | 3.62 | 4.64 | 10 | AI_VECTOR_CORE |
| | **sum** | **8.20** | | | | |
| | **mean** | **4.10** | | | | |

## 调优记录

### J1_gg_rows2

- 时间: 2026-10-12 22:48:43
- 方案: row-programs n_rows = 2x
- 结论: 无效(变慢); sum 8 -> 10 (+24.4%)

| # | shape | base avg(us) | 本次 avg(us) | Δ(us) | Δ% |
|---|-------|--------------|--------------|-------|-----|
| | `N8_Q2048_KV512_D128_R64` | 4.00 | 5.01 | +1.02 | +25.4% |
| | `N8_Q2048_KV512_D256_R128` | 4.20 | 5.18 | +0.98 | +23.3% |
| | **sum** | **8.20** | **10.20** | **+2.00** | **+24.4%** |

- 本次 min/median: `N8_Q2048_KV512_D128_R64`=4.38/5.03, `N8_Q2048_KV512_D256_R128`=4.63/5.21

