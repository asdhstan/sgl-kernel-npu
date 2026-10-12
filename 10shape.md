### _mix_fused_kernel
N:1 B:1 H:128	
N:32 B:1 H:128	
N:1024 B:1 H:128	
N:32 B:8 H:128	
N:32 B:64 H:128	
N:32 B:1 H:256	
N:32 B:1 H:1024	
N:32 B:1 H:8192	
N:32 B:2 H:512	
N:256 B:32 H:2048

### _causal_conv1d_update_kernel_npu_tiled
batch:1 dim:128 seqlen:1 width:2	
batch:8 dim:128 seqlen:1 width:2	
batch:64 dim:128 seqlen:1 width:2	
batch:1 dim:128 seqlen:1 width:4	
batch:1 dim:128 seqlen:1 width:6	
batch:1 dim:256 seqlen:1 width:2	
batch:1 dim:512 seqlen:1 width:2	
batch:1 dim:4096 seqlen:1 width:2	
batch:1 dim:128 seqlen:128 width:2	
batch:64 dim:4096 seqlen:8 width:4

### _store_kv_cache_prefix_valid_npu_kernel
batch:1 prefix_width:1 num_heads:8 head_dim:128 commit_lens:1	
batch:1 prefix_width:4 num_heads:8 head_dim:128 commit_lens:4	
batch:1 prefix_width:8 num_heads:8 head_dim:128 commit_lens:8	
batch:1 prefix_width:16 num_heads:8 head_dim:128 commit_lens:16	
batch:1 prefix_width:32 num_heads:8 head_dim:128 commit_lens:32	
batch:1 prefix_width:64 num_heads:8 head_dim:128 commit_lens:64	
batch:1 prefix_width:16 num_heads:1 head_dim:128 commit_lens:16	
batch:1 prefix_width:16 num_heads:32 head_dim:128 commit_lens:16	
batch:4 prefix_width:16 num_heads:8 head_dim:128 commit_lens:8	
batch:1 prefix_width:16 num_heads:8 head_dim:64 commit_lens:16

### add_rmsnorm_bias_kernel
batch_size:1 hidden_size:1024 SCALE:False	
batch_size:8 hidden_size:1024 SCALE:False	
batch_size:32 hidden_size:1024 SCALE:False	
batch_size:128 hidden_size:1024 SCALE:False	
batch_size:512 hidden_size:1024 SCALE:False	
batch_size:32 hidden_size:256 SCALE:False	
batch_size:32 hidden_size:8192 SCALE:False	
batch_size:32 hidden_size:4096 SCALE:True	
batch_size:32 hidden_size:8192 SCALE:True	
batch_size:32 hidden_size:16384 SCALE:True

### fused_rope_qk_mqa_kernel_opt
T:1 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:8 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:32 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:128 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:1024 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:32 Hq:32 Hk:8 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:32 Hq:64 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:True	
T:32 Hq:8 Hk:1 D_HEAD:256 D_ROPE:128 IS_NEOX_STYLE:True	
T:32 Hq:8 Hk:1 D_HEAD:128 D_ROPE:64 IS_NEOX_STYLE:True	
T:32 Hq:8 Hk:1 D_HEAD:128 D_ROPE:128 IS_NEOX_STYLE:False


### split_qkvgate_gemma_rmsnorm_rope_kernel
batch_size:1 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:32 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:128 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:4096 kv_hidden_size:1024 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:8192 kv_hidden_size:1024 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:256 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:64	
batch_size:8 q_hidden_size:1024 kv_hidden_size:256 head_dim:64 rope_dim:64	
batch_size:8 q_hidden_size:4096 kv_hidden_size:512 head_dim:128 rope_dim:128

### split_qkv_rmsnorm_rope_kernel
batch_size:1 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:32 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:128 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:4096 kv_hidden_size:1024 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:8192 kv_hidden_size:1024 head_dim:128 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:256 rope_dim:128	
batch_size:8 q_hidden_size:2048 kv_hidden_size:512 head_dim:128 rope_dim:64	
batch_size:8 q_hidden_size:1024 kv_hidden_size:256 head_dim:64 rope_dim:64	
batch_size:8 q_hidden_size:4096 kv_hidden_size:512 head_dim:128 rope_dim:128

### _argmax_prob_kernel
B:1 V:157184	
B:8 V:157184	
B:32 V:157184	
B:128 V:157184	
B:512 V:157184	
B:32 V:32000	
B:32 V:4096	
B:32 V:256000	
B:32 V:1024	
B:1024 V:128000

### _chain_rejection_sample_kernel
batch_size:1 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:8 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:128 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:256 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:0	
batch_size:32 num_draft_tokens:8 num_draft_prob_rows:7 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 pad_num_vocab_blocks:128 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:32000 vocab_block_size:2048 num_vocab_blocks:16 pad_num_vocab_blocks:16 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:4096 vocab_block_size:2048 num_vocab_blocks:2 pad_num_vocab_blocks:2 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:256000 vocab_block_size:2048 num_vocab_blocks:125 pad_num_vocab_blocks:128 all_accepted:1

### _chain_rejection_block_sum_kernel
batch_size:1 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:1 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:2048 vocab_block_size:2048 num_vocab_blocks:1 all_accepted:1	
batch_size:8 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:128 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:256 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:0	
batch_size:32 num_draft_tokens:8 num_draft_prob_rows:7 vocab_size:157184 vocab_block_size:2048 num_vocab_blocks:77 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:32000 vocab_block_size:2048 num_vocab_blocks:16 all_accepted:1	
batch_size:32 num_draft_tokens:5 num_draft_prob_rows:4 vocab_size:256000 vocab_block_size:2048 num_vocab_blocks:125 all_accepted:1
