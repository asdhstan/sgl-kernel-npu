# `split_qkv_rmsnorm_rope_kernel` 性能数据

- 测试脚本: `split_qkv_rmsnorm_rope_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `N128_Q2048_KV512_D128_R128` | 9.36 | 8.69 | 9.79 | 10 | AI_VECTOR_CORE |
| 2 | `N1_Q2048_KV512_D128_R128` | 3.37 | 2.90 | 4.07 | 10 | AI_VECTOR_CORE |
| 3 | `N32_Q2048_KV512_D128_R128` | 4.91 | 4.59 | 5.20 | 10 | AI_VECTOR_CORE |
| 4 | `N8_Q1024_KV256_D64_R64` | 3.64 | 3.14 | 4.19 | 10 | AI_VECTOR_CORE |
| 5 | `N8_Q2048_KV512_D128_R128` | 3.69 | 3.27 | 3.99 | 10 | AI_VECTOR_CORE |
| 6 | `N8_Q2048_KV512_D128_R64` | 4.08 | 3.61 | 4.55 | 10 | AI_VECTOR_CORE |
| 7 | `N8_Q2048_KV512_D256_R128` | 3.73 | 3.08 | 4.36 | 10 | AI_VECTOR_CORE |
| 8 | `N8_Q4096_KV1024_D128_R128` | 4.19 | 3.49 | 5.05 | 10 | AI_VECTOR_CORE |
| 9 | `N8_Q4096_KV512_D128_R128` | 3.87 | 3.44 | 4.29 | 10 | AI_VECTOR_CORE |
| 10 | `N8_Q8192_KV1024_D128_R128` | 4.35 | 3.49 | 4.92 | 10 | AI_VECTOR_CORE |
| | **sum** | **45.19** | | | | |
| | **mean** | **4.52** | | | | |
