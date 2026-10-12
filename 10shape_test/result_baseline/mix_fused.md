# `_mix_fused_kernel` 性能数据

- 测试脚本: `test_mix_fused_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `N1024_B1_H128` | 20.60 | 19.88 | 20.98 | 10 | AI_VECTOR_CORE |
| 2 | `N1_B1_H128` | 2.89 | 2.15 | 3.56 | 10 | AI_VECTOR_CORE |
| 3 | `N256_B32_H2048` | 109.07 | 108.30 | 109.65 | 10 | AI_VECTOR_CORE |
| 4 | `N32_B1_H1024` | 4.16 | 3.74 | 4.85 | 10 | AI_VECTOR_CORE |
| 5 | `N32_B1_H128` | 3.74 | 3.00 | 4.12 | 10 | AI_VECTOR_CORE |
| 6 | `N32_B1_H256` | 3.76 | 3.07 | 4.47 | 10 | AI_VECTOR_CORE |
| 7 | `N32_B1_H8192` | 6.66 | 6.18 | 7.20 | 10 | AI_VECTOR_CORE |
| 8 | `N32_B2_H512` | 4.36 | 4.05 | 5.02 | 10 | AI_VECTOR_CORE |
| 9 | `N32_B64_H128` | 30.39 | 29.69 | 30.83 | 10 | AI_VECTOR_CORE |
| 10 | `N32_B8_H128` | 6.59 | 5.80 | 7.23 | 10 | AI_VECTOR_CORE |
| | **sum** | **192.23** | | | | |
| | **mean** | **19.22** | | | | |
