# `fused_rope_qk_mqa_kernel_opt` 性能数据

- 测试脚本: `fused_rope_qk_mqa_kernel_opt.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `T1024_Hq8_Hk1_D128_R128_neox1` | 7.61 | 7.18 | 8.40 | 10 | AI_VECTOR_CORE |
| 2 | `T128_Hq8_Hk1_D128_R128_neox1` | 3.63 | 3.14 | 4.18 | 10 | AI_VECTOR_CORE |
| 3 | `T1_Hq8_Hk1_D128_R128_neox1` | 1.38 | 1.31 | 1.47 | 10 | AI_VECTOR_CORE |
| 4 | `T32_Hq32_Hk8_D128_R128_neox1` | 2.80 | 2.16 | 3.21 | 10 | AI_VECTOR_CORE |
| 5 | `T32_Hq64_Hk1_D128_R128_neox1` | 2.77 | 2.38 | 3.14 | 10 | AI_VECTOR_CORE |
| 6 | `T32_Hq8_Hk1_D128_R128_neox0` | 2.57 | 2.19 | 3.00 | 10 | AI_VECTOR_CORE |
| 7 | `T32_Hq8_Hk1_D128_R128_neox1` | 2.54 | 1.91 | 2.89 | 10 | AI_VECTOR_CORE |
| 8 | `T32_Hq8_Hk1_D128_R64_neox1` | 2.72 | 2.00 | 3.22 | 10 | AI_VECTOR_CORE |
| 9 | `T32_Hq8_Hk1_D256_R128_neox1` | 2.66 | 2.21 | 3.18 | 10 | AI_VECTOR_CORE |
| 10 | `T8_Hq8_Hk1_D128_R128_neox1` | 1.75 | 1.64 | 1.85 | 10 | AI_VECTOR_CORE |
| | **sum** | **30.43** | | | | |
| | **mean** | **3.04** | | | | |
