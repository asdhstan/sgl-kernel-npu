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

## 调优记录

### I1_fr_autotune

- 时间: 2026-10-12 22:48:06
- 方案: autotune num_warps{1,2,4} x num_stages{1,2}
- 结论: 无效(变慢): 该 kernel 每 token 一个 program，num_warps/num_stages 无可调空间; sum 30 -> 33 (+7.8%)

| # | shape | base avg(us) | 本次 avg(us) | Δ(us) | Δ% |
|---|-------|--------------|--------------|-------|-----|
| | `T1024_Hq8_Hk1_D128_R128_neox1` | 7.61 | 9.98 | +2.37 | +31.1% |
| | `T128_Hq8_Hk1_D128_R128_neox1` | 3.63 | 3.88 | +0.26 | +7.0% |
| | `T1_Hq8_Hk1_D128_R128_neox1` | 1.38 | 1.37 | -0.01 | -0.9% |
| | `T32_Hq32_Hk8_D128_R128_neox1` | 2.80 | 2.78 | -0.02 | -0.8% |
| | `T32_Hq64_Hk1_D128_R128_neox1` | 2.77 | 2.78 | +0.02 | +0.6% |
| | `T32_Hq8_Hk1_D128_R128_neox0` | 2.57 | 2.51 | -0.06 | -2.5% |
| | `T32_Hq8_Hk1_D128_R128_neox1` | 2.54 | 2.47 | -0.06 | -2.5% |
| | `T32_Hq8_Hk1_D128_R64_neox1` | 2.72 | 2.75 | +0.04 | +1.3% |
| | `T32_Hq8_Hk1_D256_R128_neox1` | 2.66 | 2.60 | -0.07 | -2.5% |
| | `T8_Hq8_Hk1_D128_R128_neox1` | 1.75 | 1.68 | -0.07 | -3.8% |
| | **sum** | **30.43** | **32.81** | **+2.38** | **+7.8%** |

- 本次 min/median: `T1024_Hq8_Hk1_D128_R128_neox1`=9.65/9.92, `T128_Hq8_Hk1_D128_R128_neox1`=3.25/3.90, `T1_Hq8_Hk1_D128_R128_neox1`=1.30/1.40, `T32_Hq32_Hk8_D128_R128_neox1`=2.17/2.78, `T32_Hq64_Hk1_D128_R128_neox1`=2.43/2.76, `T32_Hq8_Hk1_D128_R128_neox0`=2.30/2.43, `T32_Hq8_Hk1_D128_R128_neox1`=1.94/2.50, `T32_Hq8_Hk1_D128_R64_neox1`=1.97/2.92, `T32_Hq8_Hk1_D256_R128_neox1`=2.29/2.54, `T8_Hq8_Hk1_D128_R128_neox1`=1.62/1.68

### Z1_current_state_full_run

- 时间: 2026-10-12 23:22:09
- 方案: 全部调优默认固化后整套 full suite 复测（单次运行，非 best-of-2），用作当前工作区状态的统一口径
- 结论: 当前工作区状态（全部调优默认已固化）统一口径复测: add_rmsnorm_bias 45->43(-2.8%); argmax_prob 1173->1174(+0.1%); causal_conv1d_update 94->64(-32.4%); chain_rejection_block_sum 264->264(+0.2%); chain_rejection_sample 82->83(+0.1%); fused_rope_qk_mqa 30->31(+0.3%); mix_fused 192->194(+0.8%); split_qkv_rmsnorm_rope 45->46(+2.2%); split_qkvgate_gemma_rmsnorm_rope 8->8(+0.7%); store_kv_cache_prefix_valid 36->24(-34.8%)

| # | shape | base avg(us) | 本次 avg(us) | Δ(us) | Δ% |
|---|-------|--------------|--------------|-------|-----|
| | `T1024_Hq8_Hk1_D128_R128_neox1` | 7.61 | 7.67 | +0.06 | +0.7% |
| | `T128_Hq8_Hk1_D128_R128_neox1` | 3.63 | 3.74 | +0.12 | +3.2% |
| | `T1_Hq8_Hk1_D128_R128_neox1` | 1.38 | 1.40 | +0.01 | +0.9% |
| | `T32_Hq32_Hk8_D128_R128_neox1` | 2.80 | 2.73 | -0.07 | -2.5% |
| | `T32_Hq64_Hk1_D128_R128_neox1` | 2.77 | 2.78 | +0.01 | +0.4% |
| | `T32_Hq8_Hk1_D128_R128_neox0` | 2.57 | 2.43 | -0.14 | -5.4% |
| | `T32_Hq8_Hk1_D128_R128_neox1` | 2.54 | 2.69 | +0.16 | +6.1% |
| | `T32_Hq8_Hk1_D128_R64_neox1` | 2.72 | 2.75 | +0.04 | +1.3% |
| | `T32_Hq8_Hk1_D256_R128_neox1` | 2.66 | 2.58 | -0.08 | -3.0% |
| | `T8_Hq8_Hk1_D128_R128_neox1` | 1.75 | 1.75 | +0.01 | +0.4% |
| | **sum** | **30.43** | **30.53** | **+0.10** | **+0.3%** |

- 本次 min/median: `T1024_Hq8_Hk1_D128_R128_neox1`=6.48/7.59, `T128_Hq8_Hk1_D128_R128_neox1`=3.08/3.71, `T1_Hq8_Hk1_D128_R128_neox1`=1.30/1.42, `T32_Hq32_Hk8_D128_R128_neox1`=2.47/2.70, `T32_Hq64_Hk1_D128_R128_neox1`=2.59/2.72, `T32_Hq8_Hk1_D128_R128_neox0`=1.81/2.52, `T32_Hq8_Hk1_D128_R128_neox1`=2.45/2.62, `T32_Hq8_Hk1_D128_R64_neox1`=2.46/2.70, `T32_Hq8_Hk1_D256_R128_neox1`=1.99/2.68, `T8_Hq8_Hk1_D128_R128_neox1`=1.62/1.74

