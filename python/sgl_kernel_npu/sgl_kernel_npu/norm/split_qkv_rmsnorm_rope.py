import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _load_tile(ptr, offs, mask, NEEDS_MASK: tl.constexpr):
    """Load with a predicate only when the tile can actually overrun the row.

    When the row divides exactly into the grid's column blocks every index is in
    range, and the predicated load + select is pure overhead (this kernel is
    vector-pipe bound)."""
    if NEEDS_MASK:
        return tl.load(ptr + offs, mask=mask, other=0.0)
    else:
        return tl.load(ptr + offs)


@triton.jit
def _store_tile(ptr, offs, value, mask, NEEDS_MASK: tl.constexpr):
    if NEEDS_MASK:
        tl.store(ptr + offs, value, mask=mask)
    else:
        tl.store(ptr + offs, value)


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    NORMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    DO_PARTIAL: tl.constexpr,
    DO_HALF: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
    if NORMS:
        weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        if NORMS:
            squares = input_values * input_values
            variances = tl.sum(squares, axis=1) / HEAD_DIM
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
                Q_BLOCK_SIZE // HEAD_DIM, 1
            )
            normalized_values = (
                input_values * reciprocal_std
            )  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
            if BIAS:
                normalized_values = normalized_values * weight_values + bias_values
            else:
                normalized_values = normalized_values * weight_values
        else:
            normalized_values = input_values

        # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        if not DO_HALF:
            sin_half = al.extract_slice(
                sin, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            cos_half = al.extract_slice(
                cos, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            sin = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

            cos = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
        if DO_PARTIAL:
            rot_x = al.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            pass_x = al.extract_slice(
                normalized_values,
                offsets=(0, ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            rot_x = normalized_values
        if DO_HALF:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, 1),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        if DO_HALF:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, 1),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        roped_q = cat_x * sin + rot_x * cos
        if DO_PARTIAL:
            normalized_values = al.insert_slice(
                normalized_values,
                roped_q,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            normalized_values = al.insert_slice(
                normalized_values,
                pass_x,
                offsets=(0, ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            normalized_values = roped_q
        # store
        tl.store(
            q_ptr + output_offset + col_indices,
            normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # k
    if NORMS:
        weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        if NORMS:
            squares = input_values * input_values
            variances = tl.sum(squares, axis=1) / HEAD_DIM
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
                KV_BLOCK_SIZE // HEAD_DIM, 1
            )
            normalized_values = (
                input_values * reciprocal_std
            )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
            if BIAS:
                normalized_values = normalized_values * weight_values + bias_values
            else:
                normalized_values = normalized_values * weight_values
        else:
            normalized_values = input_values

        # # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)

        if not DO_HALF:
            sin_half = al.extract_slice(
                sin, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            cos_half = al.extract_slice(
                cos, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            sin = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

            cos = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

        if DO_PARTIAL:
            rot_x = al.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            pass_x = al.extract_slice(
                normalized_values,
                offsets=(0, ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            rot_x = normalized_values
        if DO_HALF:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, 1),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        if DO_HALF:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, 1),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        roped_k = cat_x * sin + rot_x * cos
        if DO_PARTIAL:
            normalized_values = al.insert_slice(
                normalized_values,
                roped_k,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            normalized_values = al.insert_slice(
                normalized_values,
                pass_x,
                offsets=(0, ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            normalized_values = roped_k

        # store
        tl.store(
            k_ptr + output_offset + col_indices,
            normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # v
    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(
            input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0
        )
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


kernels = {}


# Column coverage per program: KV_BLOCK_SIZE channels of the k/v region and
# (q_hidden/kv_hidden)x as many of the q region.  Wider blocks mean larger
# transfers and fewer programs, so sin/cos is loaded once per row instead of
# once per column block; measured 16.6us -> 10.6us (-36%) on the tuned shape
# (see 1008test/perf_07_split_qkv_rmsnorm_rope/notes.md).  The block is capped
# so a very wide q row cannot blow the unified buffer; None selects the default
# whole-row pick, an explicit int overrides it (head_dim = old tiling).
_KV_BLOCK_SIZE = None
_MAX_KV_BLOCK = 512


def _pick_kv_block(kv_hidden_size, head_dim):
    """Largest multiple of head_dim that divides kv_hidden_size and fits the cap."""
    best = head_dim
    block = head_dim
    while block <= _MAX_KV_BLOCK and kv_hidden_size % block == 0:
        best = block
        block *= 2
    return best


def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    _, num_vectorcore = get_device_properties()

    KV_BLOCK_SIZE = (
        _KV_BLOCK_SIZE
        if _KV_BLOCK_SIZE is not None
        else _pick_kv_block(kv_hidden_size, head_dim)
    )
    assert KV_BLOCK_SIZE % head_dim == 0, "KV_BLOCK_SIZE must cover whole heads"
    assert kv_hidden_size % KV_BLOCK_SIZE == 0
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * KV_BLOCK_SIZE
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    n_rows = (num_vectorcore + n_cols - 1) // n_cols
    BIAS = q_bias is not None
    NORMS = eps is not None
    rope_dim = sin.shape[-1]
    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        BIAS,
        NORMS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        head_dim - rope_dim,
        DO_PARTIAL=(head_dim != rope_dim),
        DO_HALF=is_neox_style,
    )

    return q_output, k_output, v_output


@triton.jit
def split_qkvgate_gemma_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    q_weight_ptr,
    k_weight_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    Q_GATE_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    qgate_hidden_size = q_hidden_size * 2

    # Row-invariant weights are read once per program.
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0

    q_input_offset = row_pid * total_hidden_size
    k_input_offset = q_input_offset + qgate_hidden_size
    v_input_offset = k_input_offset + kv_hidden_size
    q_output_offset = row_pid * q_hidden_size
    kv_output_offset = row_pid * kv_hidden_size
    input_offset_step = row_step * total_hidden_size
    q_output_step = row_step * q_hidden_size
    kv_output_step = row_step * kv_hidden_size

    # q, k and v are processed in a single row loop: the three paths share the
    # sin/cos row and the row bookkeeping, which used to be repeated by three
    # separate loops (q, k, v) that all walked the same rows.
    for row_idx in tl.range(row_pid, batch_size, row_step):
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)

        # ---- q (+ gate) ----
        q_col = col_pid * Q_GATE_BLOCK_SIZE + tl.arange(0, Q_GATE_BLOCK_SIZE)
        q_valid = q_col < qgate_hidden_size
        qgate_values = _load_tile(
            input_ptr + q_input_offset, q_col, q_valid, NEEDS_MASK
        ).reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM * 2)
        # gate is only copied through, so it stays in the input dtype and skips
        # the fp32 conversion the q part needs.
        gate = al.extract_slice(
            qgate_values,
            offsets=(0, HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
            strides=(1, 1),
        ).reshape(Q_BLOCK_SIZE)
        q = (
            al.extract_slice(
                qgate_values,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
                strides=(1, 1),
            )
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
            .to(tl.float32)
        )

        squares = q * q
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(
            Q_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = q * reciprocal_std
        normalized_values = normalized_values * q_weight_values

        rot_x = al.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        pass_x = al.extract_slice(
            normalized_values,
            offsets=(0, ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )
        x1 = al.extract_slice(
            rot_x,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = al.extract_slice(
            rot_x,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        cat_x = al.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = al.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_q = cat_x * sin + rot_x * cos
        normalized_values = al.insert_slice(
            normalized_values,
            roped_q,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        normalized_values = al.insert_slice(
            normalized_values,
            pass_x,
            offsets=(0, ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )

        q_col_out = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        q_valid_out = q_col_out < q_hidden_size
        _store_tile(
            q_ptr + q_output_offset,
            q_col_out,
            normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            q_valid_out,
            NEEDS_MASK,
        )
        _store_tile(
            gate_ptr + q_output_offset,
            q_col_out,
            gate.to(gate_ptr.dtype.element_ty),
            q_valid_out,
            NEEDS_MASK,
        )

        # ---- k ----
        kv_col = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        kv_valid = kv_col < kv_hidden_size
        input_values = (
            _load_tile(input_ptr + k_input_offset, kv_col, kv_valid, NEEDS_MASK)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(
            KV_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = input_values * reciprocal_std
        normalized_values = normalized_values * k_weight_values

        rot_x = al.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        pass_x = al.extract_slice(
            normalized_values,
            offsets=(0, ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )
        x1 = al.extract_slice(
            rot_x,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = al.extract_slice(
            rot_x,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        cat_x = al.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = al.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_k = cat_x * sin + rot_x * cos
        normalized_values = al.insert_slice(
            normalized_values,
            roped_k,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        normalized_values = al.insert_slice(
            normalized_values,
            pass_x,
            offsets=(0, ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )
        _store_tile(
            k_ptr + kv_output_offset,
            kv_col,
            normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            kv_valid,
            NEEDS_MASK,
        )

        # ---- v: pure copy in the native dtype ----
        v_values = _load_tile(input_ptr + v_input_offset, kv_col, kv_valid, NEEDS_MASK)
        _store_tile(
            v_ptr + kv_output_offset,
            kv_col,
            v_values.to(v_ptr.dtype.element_ty),
            kv_valid,
            NEEDS_MASK,
        )

        q_input_offset += input_offset_step
        k_input_offset += input_offset_step
        v_input_offset += input_offset_step
        q_output_offset += q_output_step
        kv_output_offset += kv_output_step

kernels = {}


def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    _, num_vectorcore = get_device_properties()

    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    Q_GATE_BLOCK_SIZE = Q_BLOCK_SIZE * 2
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size * 2 + kv_hidden_size * 2
    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    gate_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    n_rows = (num_vectorcore + n_cols - 1) // n_cols
    split_qkvgate_gemma_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        gate_output,
        q_weight,
        k_weight,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        Q_GATE_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        head_dim,
        rope_dim,
        rope_dim // 2,
        head_dim - rope_dim,
        # Predicated loads/stores are only needed when a column block can
        # overrun the row; for evenly divisible rows they are pure overhead.
        not (
            (q_hidden_size == n_cols * Q_BLOCK_SIZE)
            and (kv_hidden_size == n_cols * KV_BLOCK_SIZE)
        ),
    )

    return q_output, k_output, v_output, gate_output
