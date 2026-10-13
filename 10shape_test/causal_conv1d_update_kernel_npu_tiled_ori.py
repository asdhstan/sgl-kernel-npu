"""the native ``causal_conv1d_update_v2`` tiling heuristic (BLOCK_N 512/256,
``b_tile`` from the ~80-program target, ``t_chunk`` 1/48)"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test
import pytest
import torch
import triton
import triton.language as tl
from test_common import profiling_test


PAD_SLOT_ID = -1


@triton.jit(
    do_not_specialize=[
        "batch",
        "state_len",
        "num_cache_lines",
        "stride_x_seq",
        "stride_x_token",
        "stride_conv_state_seq",
        "stride_state_indices",
        "stride_o_seq",
        "stride_o_token",
    ]
)
def _causal_conv1d_update_kernel_npu_tiled(
    # Pointers
    x_ptr,  # (batch, seqlen, dim) OR (num_tokens, dim) for varlen
    w_ptr,  # (width, dim)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # same shape as x_ptr
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,  # max seqlen for varlen, or exact seqlen
    state_len,  # effective state_len computed in wrapper
    num_cache_lines,
    # Strides
    stride_x_seq,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices,
    stride_o_seq,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    # others
    pad_slot_id: tl.constexpr,
    # Meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # <= 6
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # tiling
    BLOCK_N: tl.constexpr,  # channel tile (C_TILE)
    B_TILE: tl.constexpr,  # batch tile
    T_CHUNK: tl.constexpr,  # token chunk for state update
):
    # program ids
    pid_b = tl.program_id(0)  # batch-tile id
    pid_c = tl.program_id(1)  # channel-tile id

    # channel indices for this program
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_w = idx_feats < dim

    # preload weights once per program (shared by B_TILE sequences)
    w_base = w_ptr + idx_feats * stride_w_dim
    # define to avoid "undefined" in branches
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(
            tl.float32
        )

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # token index vector for chunked copy
    tok_vec = tl.arange(0, T_CHUNK)  # [T_CHUNK]

    # process B_TILE sequences inside the same program instance
    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi  # scalar tl.int32
        lane_active = b < batch  # scalar predicate

        # -------------------------
        # APC mapping (optional)
        # -------------------------
        if IS_APC_ENABLED:
            conv_state_init = tl.load(
                initial_state_idx + b, mask=lane_active, other=0
            ).to(tl.int32)
            current_last_index = tl.load(
                block_idx_last_scheduled_token + b, mask=lane_active, other=0
            ).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(
                tl.int64
            )
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(
                tl.int64
            )
            seqlen_run = (qe - qs).to(tl.int32)
            # revise effective state_len for shorter sequences (same formula as original)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        # empty sequence -> skip (avoid early return because other lanes in tile)
        lane_active = lane_active & (seqlen_run > 0)

        # -------------------------
        # spec decoding offset (optional)
        # -------------------------
        if IS_SPEC_DECODING:
            conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(
                    tl.int64
                )
                - 1
            )
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )
        prior_tokens = (
            conv_states_base + conv_state_token_offset * stride_conv_state_tok
        )

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(
                prior_tokens + 0 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(
                prior_tokens + 1 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(
                prior_tokens + 2 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(
                prior_tokens + 3 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(
                prior_tokens + 4 * stride_conv_state_tok,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float16)

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(
            tl.int32
        )
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(
            tl.int32
        )

        # base pointers
        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = (
            conv_state_ptr
            + conv_states_offset * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            src_tok = (dst_tok + shift).to(tl.int32)  # [T_CHUNK]
            m_tok = (
                use_shift
                & (dst_tok < keep_shift)
                & (src_tok < state_len_run)
                & (dst_tok < state_len_run)
            )
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )

            src_ptrs = (
                state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            )
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            dst_tok = (keep_shift + x_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            x_tok = (tail_start + dst_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = (
                state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            )
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # -------------------------
        # STEP 3/4/5: causal conv1d (+ optional SiLU) and store output
        # This is original STEP3~5, but per-lane and without debug_barrier.
        # -------------------------
        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        # compute each token; keep tl.range so varlen can use seqlen_run as runtime trip count (like original)
        for idx_token in tl.range(seqlen_run):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # same selection logic as original (unrolled by KERNEL_WIDTH)
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    # only x[t] * w0
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(
                        x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                    ).to(tl.float16)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(
                            x_ptrs_1d, mask=lane_active & mask_w, other=0.0
                        ).to(tl.float16)

                acc += matrix_x.to(tl.float32) * matrix_w  # [BLOCK_N]

            if HAS_BIAS:
                acc += acc_bias

            acc = acc.to(tl.float16, fp_downcast_rounding="rtne")

            # roll history window
            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc.to(tl.float32)
                acc = acc / (1.0 + tl.exp(-acc))

            # store output
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def _detect_device():
    torch_npu = None
    try:
        import torch_npu  # noqa: F401
    except Exception:
        torch_npu = None

    if torch_npu is not None and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError("No CUDA or NPU device available")


def _dtype_supported(device, dtype):
    try:
        torch.zeros(1, dtype=dtype, device=device)
        return True
    except Exception:
        return False


def causal_conv1d_update_native(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """Dense (non-varlen, no-APC) Ascend tiled causal-conv1d update."""
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(1)

    if query_start_loc is None:
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width, _ = weight.shape
    num_cache_lines, state_len_total, _ = conv_state.size()

    # overwrite-on-x strategy same as original
    out = x

    stride_w_width, stride_w_dim = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_token, stride_x_dim = x.stride()
        stride_o_seq, stride_o_token, stride_o_dim = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_token, stride_istate_dim = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    # -------- tiling heuristic (native, as in the repo wrapper) --------
    # keep program count around ~[80..160]
    CORE_HINT = 40
    block_n = 512 if dim >= 512 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * CORE_HINT  # ~80
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8
    # token chunk based on block_n (32KB UB idea); conservative
    t_chunk = 1 if block_n == 512 else 48

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(1)
    return out.to(original_x_dtype)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    conv_state_indices: torch.Tensor | None = None,
):
    """Eager replica of the dense, non-varlen, non-spec kernel path.

    Mirrors the kernel's internal fp16 casts: history and ``x`` operands are
    rounded to fp16 before the fp32 accumulation, and the accumulator is
    rounded back to fp16 (rtne) before the optional SiLU and the final store.
    Returns ``(out, updated_conv_state)``.
    """
    dtype_in = x.dtype
    x = x.to(conv_state.dtype)
    batch, seqlen, dim = x.shape
    width = weight.shape[0]
    hist = width - 1  # effective state length in the non-spec path

    conv_state = conv_state.clone()
    out = torch.empty_like(x)
    w = weight.to(torch.float32)

    for b in range(batch):
        line = int(conv_state_indices[b])
        state = conv_state[line]

        # STEP 1: history columns, read before the state update.
        old = state[:hist].clone()
        # STEP 2: state <- cat(state[:hist], x)[-hist:].
        full = torch.cat([state[:hist], x[b]], dim=0)
        state[:hist].copy_(full[-hist:])

        # STEP 3: causal convolution over history + the new tokens.
        window = torch.cat([old, x[b]], dim=0).to(torch.float16).to(torch.float32)
        acc = torch.zeros(seqlen, dim, dtype=torch.float32, device=x.device)
        for t in range(seqlen):
            acc[t] = (window[t : t + width] * w).sum(0)
        if bias is not None:
            acc = acc + bias.to(torch.float32)

        acc = acc.to(torch.float16)
        if activation in ("silu", "swish"):
            acc = acc.to(torch.float32)
            acc = acc / (1.0 + torch.exp(-acc))
        out[b] = acc.to(dtype_in)

    return out, conv_state


_DETECT_ERR = ""


try:
    _DEVICE = _detect_device()
except Exception as _e:  # pragma: no cover - only hit on headless/CI hosts
    _DEVICE = None
    _DETECT_ERR = str(_e)


pytestmark = pytest.mark.skipif(_DEVICE is None, reason=f"no cuda/npu: {_DETECT_ERR}")


_TOL = {
    torch.float32: (1e-2, 1e-2),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


_PERF_SHAPES = [
    (1, 128, 1, 2),
    (8, 128, 1, 2),
    (64, 128, 1, 2),
    (1, 128, 1, 4),
    (1, 128, 1, 6),
    (1, 256, 1, 2),
    (1, 512, 1, 2),
    (1, 4096, 1, 2),
    (1, 128, 128, 2),
    (64, 4096, 8, 4),
]


def _make_case(shape, dtype, seed, state_pad=0, has_bias=True, permute=False):
    batch, dim, seqlen, width = shape
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    x = torch.randn(batch, seqlen, dim, generator=gen, dtype=dtype, device=_DEVICE)
    weight = torch.randn(width, dim, generator=gen, dtype=dtype, device=_DEVICE)
    bias = (
        torch.randn(dim, generator=gen, dtype=dtype, device=_DEVICE)
        if has_bias
        else None
    )
    num_lines = batch
    state_len = width - 1 + state_pad
    conv_state = torch.randn(
        num_lines, state_len, dim, generator=gen, dtype=dtype, device=_DEVICE
    )
    if permute:
        indices = torch.randperm(num_lines, generator=gen, device=_DEVICE)
    else:
        indices = torch.arange(num_lines, device=_DEVICE)
    conv_state_indices = indices.to(torch.int32)
    return x, conv_state, weight, bias, conv_state_indices


def _run_case(shape, dtype, seed, activation=None, state_pad=0, has_bias=True, permute=False):
    x, conv_state, weight, bias, indices = _make_case(
        shape, dtype, seed, state_pad=state_pad, has_bias=has_bias, permute=permute
    )

    out_ref, state_ref = causal_conv1d_update_ref(
        x, conv_state, weight, bias, activation=activation, conv_state_indices=indices
    )

    x_impl = x.clone()
    state_impl = conv_state.clone()
    out_impl = causal_conv1d_update_native(
        x_impl,
        state_impl,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=indices,
        pad_slot_id=PAD_SLOT_ID,
    )

    rtol, atol = _TOL[dtype]
    assert out_impl.shape == out_ref.shape and out_impl.dtype == dtype
    torch.testing.assert_close(
        out_impl.to(torch.float32), out_ref.to(torch.float32), rtol=rtol, atol=atol
    )
    # The state update is a pure copy of history/x values -> bit-exact.
    assert torch.equal(state_impl, state_ref), "conv_state update mismatch"
    return out_impl


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_D{s[1]}_T{s[2]}_W{s[3]}",
)
def test_causal_conv1d_update_performance(shape):
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )
    x, conv_state, weight, bias, indices = _make_case(shape, dtype, seed=1234)
    x_impl = x.clone()
    state_impl = conv_state.clone()

    def fn_triton():
        causal_conv1d_update_native(
            x_impl,
            state_impl,
            weight,
            bias=bias,
            activation=None,
            conv_state_indices=indices,
            pad_slot_id=PAD_SLOT_ID,
        )

    n, d, t, w = shape
    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=f"causal_conv1d_update_ori_N{n}_D{d}_T{t}_W{w}",
    )


