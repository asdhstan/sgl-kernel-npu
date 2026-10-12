# `add_rmsnorm_bias_kernel` 性能数据

- 测试脚本: `add_rmsnorm_bias_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `B128_H1024_scale0` | 4.48 | 3.90 | 5.21 | 10 | AI_VECTOR_CORE |
| 2 | `B1_H1024_scale0` | 1.49 | 1.41 | 1.62 | 10 | AI_VECTOR_CORE |
| 3 | `B32_H1024_scale0` | 2.61 | 2.29 | 2.85 | 10 | AI_VECTOR_CORE |
| 4 | `B32_H16384_scale1` | 8.42 | 8.18 | 8.81 | 10 | AI_VECTOR_CORE |
| 5 | `B32_H256_scale0` | 2.52 | 2.33 | 2.71 | 10 | AI_VECTOR_CORE |
| 6 | `B32_H4096_scale1` | 3.75 | 3.23 | 4.09 | 10 | AI_VECTOR_CORE |
| 7 | `B32_H8192_scale0` | 3.70 | 3.47 | 3.87 | 10 | AI_VECTOR_CORE |
| 8 | `B32_H8192_scale1` | 5.10 | 4.79 | 5.39 | 10 | AI_VECTOR_CORE |
| 9 | `B512_H1024_scale0` | 10.84 | 10.34 | 11.55 | 10 | AI_VECTOR_CORE |
| 10 | `B8_H1024_scale0` | 1.78 | 1.70 | 1.87 | 10 | AI_VECTOR_CORE |
| | **sum** | **44.71** | | | | |
| | **mean** | **4.47** | | | | |
