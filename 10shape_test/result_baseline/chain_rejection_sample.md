# `_chain_rejection_sample_kernel` 性能数据

- 测试脚本: `chain_rejection_sample_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `N128_Dt5_Dr4_V157184_B2048_NB77_pad128_acc1` | 13.98 | 13.42 | 14.56 | 10 | AI_VECTOR_CORE |
| 2 | `N1_Dt5_Dr4_V157184_B2048_NB77_pad128_acc1` | 4.85 | 4.74 | 4.98 | 10 | AI_VECTOR_CORE |
| 3 | `N256_Dt5_Dr4_V157184_B2048_NB77_pad128_acc1` | 22.40 | 21.86 | 23.25 | 10 | AI_VECTOR_CORE |
| 4 | `N32_Dt5_Dr4_V157184_B2048_NB77_pad128_acc0` | 6.24 | 5.99 | 6.44 | 10 | AI_VECTOR_CORE |
| 5 | `N32_Dt5_Dr4_V157184_B2048_NB77_pad128_acc1` | 5.99 | 5.86 | 6.37 | 10 | AI_VECTOR_CORE |
| 6 | `N32_Dt5_Dr4_V256000_B2048_NB125_pad128_acc1` | 6.06 | 5.81 | 6.29 | 10 | AI_VECTOR_CORE |
| 7 | `N32_Dt5_Dr4_V32000_B2048_NB16_pad16_acc1` | 5.73 | 5.62 | 5.91 | 10 | AI_VECTOR_CORE |
| 8 | `N32_Dt5_Dr4_V4096_B2048_NB2_pad2_acc1` | 5.65 | 5.44 | 5.84 | 10 | AI_VECTOR_CORE |
| 9 | `N32_Dt8_Dr7_V157184_B2048_NB77_pad128_acc1` | 5.96 | 5.67 | 6.16 | 10 | AI_VECTOR_CORE |
| 10 | `N8_Dt5_Dr4_V157184_B2048_NB77_pad128_acc1` | 5.63 | 5.43 | 5.81 | 10 | AI_VECTOR_CORE |
| | **sum** | **82.49** | | | | |
| | **mean** | **8.25** | | | | |
