# 10shape 基准性能汇总

- 生成时间: 2026-10-12 23:25:02
- git HEAD: `89cf537`
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
| A1_autotune | 2026-10-12 21:58:35 | `argmax_prob` | ARGMAX_TUNE=autotune: BLOCK_V in {2048,4096,8192,16384,32768} x num_stages {1,2}, num_warps=1 | 无效: sum 1173 -> 1165 (-0.7%) |
| A2_simt_only | 2026-10-12 21:59:20 | `argmax_prob` | compile_mode=simt_only, BLOCK_V=default(32KB/elem) | 无效: sum 1173 -> 2423 (+106.5%) |
| A3_simt_only+autotune | 2026-10-12 21:46:52 | `argmax_prob` | compile_mode=simt_only + ARGMAX_TUNE=autotune (BLOCK_V 2048..32768 x num_stages 1/2) | 无效: sum 1173 -> 9109 (+676.3%) |
| A4_grid2 | 2026-10-12 22:00:34 | `argmax_prob` | grid = min(2*vector_cores, B) (default is min(cores, B)) | 无效: sum 1173 -> 1183 (+0.8%) |
| A5_grid4 | 2026-10-12 22:01:17 | `argmax_prob` | grid = min(4*vector_cores, B) | 无效: sum 1173 -> 1179 (+0.5%) |
| B1_tchunk8 | 2026-10-12 22:03:09 | `causal_conv1d_update` | T_CHUNK=8 (repo: 1 when BLOCK_N==512, else 48) | 有效: sum 94 -> 79 (-16.1%); 最好 -31.1%(N64_D4096_T8_W4) 最差 +13.5%(N1_D512_T1_W2) |
| B2_tchunk16 | 2026-10-12 22:04:26 | `causal_conv1d_update` | T_CHUNK=16 | 有效: sum 94 -> 78 (-17.4%); 最好 -31.9%(N64_D4096_T8_W4) 最差 +18.6%(N1_D512_T1_W2) |
| B3_tc512_16 | 2026-10-12 22:05:58 | `causal_conv1d_update` | T_CHUNK=16 only when BLOCK_N==512; BLOCK_N==256 keeps repo default 48 | 有效: sum 94 -> 78 (-16.8%); 最好 -32.2%(N64_D4096_T8_W4) 最差 +18.2%(N1_D512_T1_W2) |
| B4_tc512_32 | 2026-10-12 22:07:00 | `causal_conv1d_update` | T_CHUNK=32 when BLOCK_N==512 | 有效: sum 94 -> 82 (-13.2%); 最好 -26.8%(N64_D4096_T8_W4) 最差 +35.8%(N1_D512_T1_W2) |
| B5_tc512_64 | 2026-10-12 22:08:11 | `causal_conv1d_update` | T_CHUNK=64 when BLOCK_N==512 | 有效: sum 94 -> 92 (-2.2%); 最好 -7.9%(N64_D4096_T8_W4) 最差 +67.4%(N1_D512_T1_W2) |
| B6_chunk_seqlen_cap16 | 2026-10-12 22:09:41 | `causal_conv1d_update` | T_CHUNK = min(16, next_pow2(seqlen)) (decode seqlen=1 -> 1) | 有效: sum 94 -> 79 (-16.7%); 最好 -31.7%(N64_D4096_T8_W4) 最差 +28.1%(N1_D128_T1_W6) |
| B7_chunk_seqlen_cap32 | 2026-10-12 22:10:28 | `causal_conv1d_update` | T_CHUNK = min(32, next_pow2(seqlen)) | 有效: sum 94 -> 78 (-17.1%); 最好 -31.1%(N64_D4096_T8_W4) 最差 +25.2%(N1_D128_T1_W6) |
| B8_baked_default | 2026-10-12 22:11:36 | `causal_conv1d_update` | B7 的规则固化为工作区默认配置 (t_chunk = min(32, next_pow2(seqlen)))，重新测量 | 有效: sum 94 -> 78 (-16.9%); 最好 -31.1%(N64_D4096_T8_W4) 最差 +27.5%(N1_D128_T1_W6) |
| C1_mix_cores2 | 2026-10-12 22:13:42 | `mix_fused` | grid=NUM_CORES=2*vector_cores (repo: 1x) | 无效: sum 192 -> 215 (+12.1%) |
| C2_mix_cores4 | 2026-10-12 22:14:43 | `mix_fused` | grid=NUM_CORES=4*vector_cores | 无效: sum 192 -> 217 (+12.9%) |
| D1_crbs_vb1024 | 2026-10-12 22:15:45 | `chain_rejection_block_sum` | vocab_block_size=1024 (repo: 2048), num_vocab_blocks doubles | 无效: sum 264 -> 424 (+60.7%) |
| D2_crbs_vb4096 | 2026-10-12 22:16:40 | `chain_rejection_block_sum` | vocab_block_size=4096 | 有效: sum 264 -> 191 (-27.8%); 最好 -33.2%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +14.8%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| D3_crbs_grid2 | 2026-10-12 22:17:43 | `chain_rejection_block_sum` | program cap = 2*vector_cores (repo: 1x) | 有效: sum 264 -> 258 (-2.4%); 最好 -7.6%(N32_Dt5_Dr4_V157184_B2048_NB77_acc0) 最差 +6.3%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| C3_mix_nomb | 2026-10-12 22:19:27 | `mix_fused` | multibuffer=False | 无效: sum 192 -> 193 (+0.2%) |
| D2b_crbs_vb4096 | 2026-10-12 22:20:30 | `chain_rejection_block_sum` | vocab_block_size=4096 (shape 断言已按 override 修正后复测) | 有效: sum 264 -> 192 (-27.3%); 最好 -31.9%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +17.0%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| D5_crbs_vb8192 | 2026-10-12 22:21:32 | `chain_rejection_block_sum` | vocab_block_size=8192 | 有效: sum 264 -> 170 (-35.7%); 最好 -43.6%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +29.4%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| D6_crbs_vb4096_grid2 | 2026-10-12 22:22:28 | `chain_rejection_block_sum` | vocab_block_size=4096 + program cap 2*cores | 有效: sum 264 -> 186 (-29.7%); 最好 -35.0%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +18.0%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| D5b_crbs_vb8192 | 2026-10-12 22:23:37 | `chain_rejection_block_sum` | vocab_block_size=8192 | 有效: sum 264 -> 169 (-35.9%); 最好 -43.9%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +32.5%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| D7_crbs_vb16384 | 2026-10-12 22:24:25 |  | vocab_block_size=16384 | 无效(编译失败): bishengir 'ub overflow, requires 2097152 bits while 2031616 bits available'；该次未产生新 profile，故不记录数据 |
| D8_crbs_vb8192_grid2 | 2026-10-12 22:25:54 | `chain_rejection_block_sum` | vocab_block_size=8192 + program cap 2*cores | 有效: sum 264 -> 168 (-36.4%); 最好 -43.8%(N128_Dt5_Dr4_V157184_B2048_NB77_acc1) 最差 +31.0%(N1_Dt5_Dr4_V2048_B2048_NB1_acc1) |
| E1_crs_vb8192 | 2026-10-12 22:27:41 | `chain_rejection_sample` | sample kernel vocab_block_size=8192 (与 block_sum 配对) | 无效: sum 82 -> 220 (+166.8%) |
| E2_crs_vb4096 | 2026-10-12 22:29:15 | `chain_rejection_sample` | sample kernel vocab_block_size=4096 (与 block_sum 配对) | 无效: sum 82 -> 123 (+48.6%) |
| A6_simd | 2026-10-12 22:36:39 | `argmax_prob` | compile_mode=simd (backend 默认是 simd_simt_template) | 无效(完全中性): compile_mode=simd 与后端默认 simd_simt_template 生成相同代码; sum 1173 -> 1173 (-0.0%) |
| A7_autotune_warps | 2026-10-12 22:39:14 | `argmax_prob` | autotune BLOCK_V x num_stages x num_warps{1,2,4} | 无效: autotune 扩到 num_warps{1,2,4} x num_stages{1,2} x BLOCK_V{5} 仍无收益，确认 argmax 已到可用旋钮上限; sum 1173 -> 1166 (-0.6%) |
| F1_skv_block256 | 2026-10-12 22:40:56 | `store_kv_cache_prefix_valid` | row channel BLOCK=256 (repo: 128) | 有效: sum 36 -> 32 (-11.3%); 最好 -29.7%(B1_W4_H8_D128_C4) 最差 +2.2%(B1_W8_H8_D128_C8) |
| F2_skv_block512 | 2026-10-12 22:41:57 | `store_kv_cache_prefix_valid` | row channel BLOCK=512 | 有效: sum 36 -> 27 (-24.6%); 最好 -47.8%(B1_W4_H8_D128_C4) 最差 +1.8%(B1_W16_H1_D128_C16) |
| G1_arb_grid2 | 2026-10-12 22:43:03 | `add_rmsnorm_bias` | grid n_rows = 2*vector_cores (repo: 1x) | 无效: sum 45 -> 44 (-0.7%) |
| G2_arb_col8192 | 2026-10-12 22:44:17 | `add_rmsnorm_bias` | COL_BLOCK_SIZE=8192 (repo: 2048) | 有效: sum 45 -> 43 (-4.0%); 最好 -14.5%(B32_H8192_scale1) 最差 +1.7%(B8_H1024_scale0) |
| H1_sq_rows2 | 2026-10-12 22:45:50 | `split_qkv_rmsnorm_rope` | row-programs n_rows = 2x (repo: ceil(cores/n_cols)) | 无效(变慢): 行程序数超过物理核数只是把同一批 row 再切一遍; sum 45 -> 53 (+17.2%) |
| I1_fr_autotune | 2026-10-12 22:48:06 | `fused_rope_qk_mqa` | autotune num_warps{1,2,4} x num_stages{1,2} | 无效(变慢): 该 kernel 每 token 一个 program，num_warps/num_stages 无可调空间; sum 30 -> 33 (+7.8%) |
| J1_gg_rows2 | 2026-10-12 22:48:43 | `split_qkvgate_gemma_rmsnorm_rope` | row-programs n_rows = 2x | 无效(变慢); sum 8 -> 10 (+24.4%) |
| F3_skv_block1024 | 2026-10-12 22:50:04 | `store_kv_cache_prefix_valid` | row channel BLOCK=1024 | 有效: sum 36 -> 24 (-32.2%); 最好 -54.2%(B1_W4_H8_D128_C4) 最差 +1.3%(B1_W16_H1_D128_C16) |
| G3_arb_col16384 | 2026-10-12 22:51:05 | `add_rmsnorm_bias` | COL_BLOCK_SIZE=16384 | 有效: sum 45 -> 44 (-2.6%); 最好 -22.3%(B32_H16384_scale1) 最差 +18.7%(B32_H4096_scale1) |
| F4_skv_block2048 | 2026-10-12 22:52:14 | `store_kv_cache_prefix_valid` | row channel BLOCK=2048 | 有效: sum 36 -> 25 (-30.7%); 最好 -53.5%(B1_W4_H8_D128_C4) 最差 +2.9%(B1_W16_H1_D128_C16) |
| F5_skv_block4096 | 2026-10-12 22:53:11 | `store_kv_cache_prefix_valid` | row channel BLOCK=4096 | 有效: sum 36 -> 24 (-33.6%); 最好 -53.9%(B1_W4_H8_D128_C4) 最差 +4.4%(B1_W16_H1_D128_C16) |
| F6_skv_baked_default | 2026-10-12 22:54:33 | `store_kv_cache_prefix_valid` | F5 的 BLOCK=4096 固化为工作区默认，重新测量 | 有效: sum 36 -> 24 (-32.8%); 最好 -53.3%(B1_W4_H8_D128_C4) 最差 +6.5%(B1_W16_H1_D128_C16)  << 已固化到工作区默认配置（该旋钮只改变发射分区/列块划分，数学与输出完全一致） |
| G4_arb_baked_default | 2026-10-12 22:55:30 | `add_rmsnorm_bias` | G2 的 COL_BLOCK=8192 固化为工作区默认，重新测量 | 有效: sum 45 -> 44 (-2.0%); 最好 -14.6%(B32_H8192_scale1) 最差 +4.4%(B128_H1024_scale0)  << 已固化到工作区默认配置（该旋钮只改变发射分区/列块划分，数学与输出完全一致） |
| K1_causal_blockn256 | 2026-10-12 23:01:37 | `causal_conv1d_update` | BLOCK_N=256 for all shapes (repo: 512 when dim>=512) | 无效(变慢): BLOCK_N 减半后程序数翻倍但每个 program 变窄，收益为负；当前默认(t_chunk 调优)为 78.39us，本次 103.02us (sum 94 -> 103, +9.2%) |
| K2_causal_blockn128 | 2026-10-12 23:03:13 | `causal_conv1d_update` | BLOCK_N=128 for all shapes | 无效(明显变慢): 同 K1 更极端 (sum 94 -> 120, +27.4%) |
| L1_mix_autotune | 2026-10-12 23:05:42 | `mix_fused` | autotune num_warps{1,2,4} x num_stages{1,2} | 无效(中性): num_warps/num_stages 对 mix_fused 无可调空间 (sum 192 -> 194, +0.8%) |
| M1_causal_btile4 | 2026-10-12 23:07:34 | `causal_conv1d_update` | B_TILE=4 (repo heuristic: up to 8) | 有效: 相对当时的默认(t_chunk 调优后 78.4us); sum 94(原始基线) -> 71 (-24.8%) [当时工作区默认 78.4us] |
| M2_causal_btile2 | 2026-10-12 23:08:26 | `causal_conv1d_update` | B_TILE=2 | 最优: B_TILE=2；收益集中在多序列 shape (N64_D4096_T8_W4 35.2 -> 20.7us); sum 94(原始基线) -> 64 (-32.6%) [当时工作区默认 78.4us] |
| M3_causal_btile1 | 2026-10-12 23:09:11 | `causal_conv1d_update` | B_TILE=1 | 有效: 与 B_TILE=2 接近；batch=1 的 shape 略好，说明静态 B_TILE 循环会浪费 lane; sum 94(原始基线) -> 64 (-32.0%) [当时工作区默认 78.4us] |
| N1_argmax_blockv64k | 2026-10-12 23:10:35 | `argmax_prob` | autotune 扩展 BLOCK_V=65536 | 无效(中性): BLOCK_V 扩到 65536 仍无收益; sum 1173(原始基线) -> 1166 (-0.6%) |
| M4_btile_baked_min2batch | 2026-10-12 23:11:45 | `causal_conv1d_update` | B_TILE = max(1, min(2, batch)) 固化为工作区默认，重新测量 | 固化确认: B_TILE = max(1, min(2, batch)); sum 94(原始基线) -> 63 (-32.9%) << 已固化；此后 causal_conv1d 默认 = BLOCK_N(512/256) + T_CHUNK min(32,next_pow2(seqlen)) + B_TILE min(2,batch) |
| K3_blockn256_btile2 | 2026-10-12 23:12:39 | `causal_conv1d_update` | 在新 B_TILE=min(2,batch) 默认下重扫 BLOCK_N=256 | 无效(变慢): 新默认下 BLOCK_N=256 仍不如 512; sum 94(原始基线) -> 74 (-21.0%) |
| K4_blockn1024_btile2 | 2026-10-12 23:13:44 | `causal_conv1d_update` | BLOCK_N=1024 (更少更宽的程序) | 无效(变慢): BLOCK_N=1024; sum 94(原始基线) -> 67 (-28.6%) |
| Z1_current_state_full_run | 2026-10-12 23:22:09 | `mix_fused`, `causal_conv1d_update`, `store_kv_cache_prefix_valid`, `add_rmsnorm_bias`, `fused_rope_qk_mqa`, `split_qkv_rmsnorm_rope`, `split_qkvgate_gemma_rmsnorm_rope`, `argmax_prob`, `chain_rejection_block_sum`, `chain_rejection_sample` | 全部调优默认固化后整套 full suite 复测（单次运行，非 best-of-2），用作当前工作区状态的统一口径 | 当前工作区状态（全部调优默认已固化）统一口径复测: add_rmsnorm_bias 45->43(-2.8%); argmax_prob 1173->1174(+0.1%); causal_conv1d_update 94->64(-32.4%); chain_rejection_block_sum 264->264(+0.2%); chain_rejection_sample 82->83(+0.1%); fused_rope_qk_mqa 30->31(+0.3%); mix_fused 192->194(+0.8%); split_qkv_rmsnorm_rope 45->46(+2.2%); split_qkvgate_gemma_rmsnorm_rope 8->8(+0.7%); store_kv_cache_prefix_valid 36->24(-34.8%) |
| P1_argmax_grid_div2 | 2026-10-12 23:23:18 | `argmax_prob` | grid = vector_cores/2 (更少程序，此前只向上扫过) | 无效(明显变慢): 程序数减半 -> sum 1173 -> 2149 (+83.1%) |
| P2_argmax_grid_div4 | 2026-10-12 23:24:01 | `argmax_prob` | grid = vector_cores/4 | 无效(严重变慢): grid=cores/4 -> +248.1% |
| Q1_mix_cores_div2 | 2026-10-12 23:24:53 | `mix_fused` | NUM_CORES = vector_cores/2 | 无效(明显变慢): NUM_CORES=cores/2 -> +79.9% |
