# 10shape 基准性能汇总

- 生成时间: 2026-10-12 22:29:30
- git HEAD: `8aadd55`
- 指标: 每个 shape 的 `avg_time(us)`，取自 profiler `kernel_details.csv` 的 `Duration(us)` 均值
- 测量协议: baseline 为 best-of-2 次运行的最小 `avg_time`

## Baseline

| kernel | shapes | sum(us) | mean(us) | 明细 |
|--------|--------|---------|----------|------|
| `_mix_fused_kernel` | 10 | 192.23 | 19.22 | [mix_fused.md](mix_fused.md) |
| `_causal_conv1d_update_kernel_npu_tiled` | 10 | 94.30 | 9.43 | [causal_conv1d_update.md](causal_conv1d_update.md) |
| `_store_kv_cache_prefix_valid_npu_kernel` | 10 | 36.07 | 3.61 | [store_kv_cache_prefix_valid.md](store_kv_cache_prefix_valid.md) |
| `add_rmsnorm_bias_kernel` | 10 | 44.71 | 4.47 | [add_rmsnorm_bias.md](add_rmsnorm_bias.md) |
| `fused_rope_qk_mqa_kernel_opt` | 10 | 30.43 | 3.04 | [fused_rope_qk_mqa.md](fused_rope_qk_mqa.md) |
| `split_qkv_rmsnorm_rope_kernel` | 10 | 45.19 | 4.52 | [split_qkv_rmsnorm_rope.md](split_qkv_rmsnorm_rope.md) |
| `split_qkvgate_gemma_rmsnorm_rope_kernel` | 2 | 8.20 | 4.10 | [split_qkvgate_gemma_rmsnorm_rope.md](split_qkvgate_gemma_rmsnorm_rope.md) |
| `_argmax_prob_kernel` | 10 | 1173.43 | 117.34 | [argmax_prob.md](argmax_prob.md) |
| `_chain_rejection_block_sum_kernel` | 10 | 263.98 | 26.40 | [chain_rejection_block_sum.md](chain_rejection_block_sum.md) |
| `_chain_rejection_sample_kernel` | 10 | 82.49 | 8.25 | [chain_rejection_sample.md](chain_rejection_sample.md) |
| **合计** | | **1971.03** | | |

## 调优尝试索引

| 尝试 | 时间 | 算子 | 方案 | 结论 |
|------|------|------|------|------|
| A1_autotune | 2026-10-12 21:58:35 | `argmax_prob` | ARGMAX_TUNE=autotune: BLOCK_V in {2048,4096,8192,16384,32768} x num_stages {1,2}, num_warps=1 | 基本无效: sum 1173 -> 1165 (-0.7%); 逐 shape -2.2%(B32_V32000) .. +0.3%(B32_V1024) |
| A2_simt_only | 2026-10-12 21:59:20 | `argmax_prob` | compile_mode=simt_only, BLOCK_V=default(32KB/elem) | 无效(变慢 ~2x): sum 1173 -> 2423 (+106.5%) |
| A3_simt_only+autotune | 2026-10-12 21:46:52 | `argmax_prob` | compile_mode=simt_only + ARGMAX_TUNE=autotune (BLOCK_V 2048..32768 x num_stages 1/2) | 无效(变慢 ~7.8x): sum 1173 -> 9109 (+676.3%) |
| A4_grid2 | 2026-10-12 22:00:34 | `argmax_prob` | grid = min(2*vector_cores, B) (default is min(cores, B)) | 基本无效: sum 1173 -> 1183 (+0.8%)，说明瓶颈不在程序数/负载均衡 |
| A5_grid4 | 2026-10-12 22:01:17 | `argmax_prob` | grid = min(4*vector_cores, B) | 基本无效: sum 1173 -> 1179 (+0.5%)，说明瓶颈不在程序数/负载均衡 |
| B1_tchunk8 | 2026-10-12 22:03:09 | `causal_conv1d_update` | T_CHUNK=8 (repo: 1 when BLOCK_N==512, else 48) | 有效: sum 94 -> 79 (-16.1%); 最好 -31.1%(N64_D4096_T8_W4) 最差 +13.5%(N1_D512_T1_W2) |
| B2_tchunk16 | 2026-10-12 22:04:26 | `causal_conv1d_update` | T_CHUNK=16 | 有效: sum 94 -> 78 (-17.4%); 最好 -31.9%(N64_D4096_T8_W4) 最差 +18.6%(N1_D512_T1_W2) |
| B3_tc512_16 | 2026-10-12 22:05:58 | `causal_conv1d_update` | T_CHUNK=16 only when BLOCK_N==512; BLOCK_N==256 keeps repo default 48 | 有效: sum 94 -> 78 (-16.8%); 最好 -32.2%(N64_D4096_T8_W4) 最差 +18.2%(N1_D512_T1_W2) |
| B4_tc512_32 | 2026-10-12 22:07:00 | `causal_conv1d_update` | T_CHUNK=32 when BLOCK_N==512 | 有效: sum 94 -> 82 (-13.2%); 最好 -26.8%(N64_D4096_T8_W4) 最差 +35.8%(N1_D512_T1_W2) |
| B5_tc512_64 | 2026-10-12 22:08:11 | `causal_conv1d_update` | T_CHUNK=64 when BLOCK_N==512 | 有效但弱: sum 94 -> 92 (-2.2%) |
| B6_chunk_seqlen_cap16 | 2026-10-12 22:09:41 | `causal_conv1d_update` | T_CHUNK = min(16, next_pow2(seqlen)) (decode seqlen=1 -> 1) | 有效: sum 94 -> 79 (-16.7%); 最好 -31.7%(N64_D4096_T8_W4) 最差 +28.1%(N1_D128_T1_W6) |
| B7_chunk_seqlen_cap32 | 2026-10-12 22:10:28 | `causal_conv1d_update` | T_CHUNK = min(32, next_pow2(seqlen)) | 最优: sum 94 -> 78 (-17.1%); 最好 -31.1%(N64_D4096_T8_W4) 最差 +25.2%(N1_D128_T1_W6) |
| B8_baked_default | 2026-10-12 22:11:36 | `causal_conv1d_update` | B7 的规则固化为工作区默认配置 (t_chunk = min(32, next_pow2(seqlen)))，重新测量 | 固化确认: sum 94 -> 78 (-16.9%)，与 B7 一致；此后 causal_conv1d 的默认配置即为该调优结果 |
| C1_mix_cores2 | 2026-10-12 22:13:42 | `mix_fused` | grid=NUM_CORES=2*vector_cores (repo: 1x) | 无效(变慢): sum 192 -> 215 (+12.1%) |
| C2_mix_cores4 | 2026-10-12 22:14:43 | `mix_fused` | grid=NUM_CORES=4*vector_cores | 无效(变慢): sum 192 -> 217 (+12.9%) |
| D1_crbs_vb1024 | 2026-10-12 22:15:45 | `chain_rejection_block_sum` | vocab_block_size=1024 (repo: 2048), num_vocab_blocks doubles | 无效(明显变慢): sum 264 -> 424 (+60.7%)，且精度断言未适配（见 D5b 修复） |
| D2_crbs_vb4096 | 2026-10-12 22:16:40 | `chain_rejection_block_sum` | vocab_block_size=4096 | 有效: sum 264 -> 191 (-27.8%)；本次精度用例因 shape 断言未适配而失败，D5b 修复后复测 |
| D3_crbs_grid2 | 2026-10-12 22:17:43 | `chain_rejection_block_sum` | program cap = 2*vector_cores (repo: 1x) | 基本无效: sum 264 -> 258 (-2.4%) |
| C3_mix_nomb | 2026-10-12 22:19:27 | `mix_fused` | multibuffer=False | 无效(中性): sum 192 -> 193 (+0.2%)；multibuffer 开关对本 kernel 无影响 |
| D2b_crbs_vb4096 | 2026-10-12 22:20:30 | `chain_rejection_block_sum` | vocab_block_size=4096 (shape 断言已按 override 修正后复测) | 有效: sum 264 -> 192 (-27.3%)（仍有 1 个 ragged 断言未适配） |
| D5_crbs_vb8192 | 2026-10-12 22:21:32 | `chain_rejection_block_sum` | vocab_block_size=8192 | 有效: sum 264 -> 170 (-35.7%)；本次精度用例因 shape 断言未适配而失败，D5b 修复后复测 |
| D6_crbs_vb4096_grid2 | 2026-10-12 22:22:28 | `chain_rejection_block_sum` | vocab_block_size=4096 + program cap 2*cores | 有效: sum 264 -> 186 (-29.7%)；本次精度用例因 shape 断言未适配而失败，D5b 修复后复测 |
| D5b_crbs_vb8192 | 2026-10-12 22:23:37 | `chain_rejection_block_sum` | vocab_block_size=8192 | 单算子最优(23 passed): sum 264 -> 169 (-35.9%)；但见 E1 的配对结论，未固化默认 |
| D7_crbs_vb16384 | 2026-10-12 22:24:25 |  | vocab_block_size=16384 | 无效(编译失败): bishengir 'ub overflow, requires 2097152 bits while 2031616 bits available'；该次未产生新 profile，故不记录数据 |
| D8_crbs_vb8192_grid2 | 2026-10-12 22:25:54 | `chain_rejection_block_sum` | vocab_block_size=8192 + program cap 2*cores | 单算子最优(23 passed): sum 264 -> 168 (-36.4%)；配对后整体变慢(见 E1)，未固化 |
| E1_crs_vb8192 | 2026-10-12 22:27:41 | `chain_rejection_sample` | sample kernel vocab_block_size=8192 (与 block_sum 配对) | 无效(明显变慢): sum 82 -> 220 (+166.8%)；配对总时 D5b+E1 = 389 vs 基线 346 (+12%)，故 vocab_block_size 不固化 |
| E2_crs_vb4096 | 2026-10-12 22:29:15 | `chain_rejection_sample` | sample kernel vocab_block_size=4096 (与 block_sum 配对) | 无效(变慢): sum 82 -> 123 (+48.6%)；配对总时 D2b+E2 = 315 vs 基线 346 (-9.2%)，净收益但改变算子契约，未固化 |
