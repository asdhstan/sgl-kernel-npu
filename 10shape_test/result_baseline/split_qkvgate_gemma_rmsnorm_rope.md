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
