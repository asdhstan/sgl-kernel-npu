# `_chain_rejection_block_sum_kernel` 性能数据

- 测试脚本: `chain_rejection_block_sum_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `N128_Dt5_Dr4_V157184_B2048_NB77_acc1` | 47.20 | 46.62 | 48.08 | 10 | AI_VECTOR_CORE |
| 2 | `N1_Dt5_Dr4_V157184_B2048_NB77_acc1` | 4.86 | 4.36 | 5.32 | 10 | AI_VECTOR_CORE |
| 3 | `N1_Dt5_Dr4_V2048_B2048_NB1_acc1` | 1.20 | 1.15 | 1.30 | 10 | AI_VECTOR_CORE |
| 4 | `N256_Dt5_Dr4_V157184_B2048_NB77_acc1` | 122.49 | 121.69 | 124.06 | 10 | AI_VECTOR_CORE |
| 5 | `N32_Dt5_Dr4_V157184_B2048_NB77_acc0` | 18.72 | 18.35 | 19.17 | 10 | AI_VECTOR_CORE |
| 6 | `N32_Dt5_Dr4_V157184_B2048_NB77_acc1` | 15.98 | 15.65 | 16.27 | 10 | AI_VECTOR_CORE |
| 7 | `N32_Dt5_Dr4_V256000_B2048_NB125_acc1` | 21.47 | 20.61 | 21.94 | 10 | AI_VECTOR_CORE |
| 8 | `N32_Dt5_Dr4_V32000_B2048_NB16_acc1` | 7.98 | 7.54 | 8.39 | 10 | AI_VECTOR_CORE |
| 9 | `N32_Dt8_Dr7_V157184_B2048_NB77_acc1` | 15.75 | 15.23 | 16.30 | 10 | AI_VECTOR_CORE |
| 10 | `N8_Dt5_Dr4_V157184_B2048_NB77_acc1` | 8.33 | 7.95 | 8.83 | 10 | AI_VECTOR_CORE |
| | **sum** | **263.98** | | | | |
| | **mean** | **26.40** | | | | |
