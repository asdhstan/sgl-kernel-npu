# `_store_kv_cache_prefix_valid_npu_kernel` 性能数据

- 测试脚本: `store_kv_cache_prefix_valid_npu_kernel.py`
- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 `Duration(us)` 的均值（profiler active 步）
- 测量协议: baseline 取 best-of-2 次运行的 `avg_time`（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）
- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；跨次比较时同时看 `avg_time` 与 `min`

## Baseline

| # | shape | avg_time(us) | min(us) | max(us) | n | core |
|---|-------|--------------|---------|---------|---|------|
| 1 | `B1_W16_H1_D128_C16` | 2.20 | 1.71 | 2.64 | 10 | AI_VECTOR_CORE |
| 2 | `B1_W16_H32_D128_C16` | 4.71 | 4.13 | 5.06 | 10 | AI_VECTOR_CORE |
| 3 | `B1_W16_H8_D128_C16` | 3.55 | 2.55 | 4.32 | 10 | AI_VECTOR_CORE |
| 4 | `B1_W16_H8_D64_C16` | 3.20 | 2.91 | 3.66 | 10 | AI_VECTOR_CORE |
| 5 | `B1_W1_H8_D128_C1` | 1.64 | 1.44 | 1.79 | 10 | AI_VECTOR_CORE |
| 6 | `B1_W32_H8_D128_C32` | 4.10 | 3.54 | 5.06 | 10 | AI_VECTOR_CORE |
| 7 | `B1_W4_H8_D128_C4` | 3.22 | 2.68 | 3.82 | 10 | AI_VECTOR_CORE |
| 8 | `B1_W64_H8_D128_C64` | 5.10 | 4.52 | 5.90 | 10 | AI_VECTOR_CORE |
| 9 | `B1_W8_H8_D128_C8` | 3.12 | 2.92 | 3.26 | 10 | AI_VECTOR_CORE |
| 10 | `B4_W16_H8_D128_C8` | 5.24 | 4.55 | 5.96 | 10 | AI_VECTOR_CORE |
| | **sum** | **36.07** | | | | |
| | **mean** | **3.61** | | | | |
