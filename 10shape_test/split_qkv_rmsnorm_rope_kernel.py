"""Precision test for ``split_qkv_rmsnorm_rope_kernel``.

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one input row of ``[batch, q + 2*kv]``, independently):
  1. the row is split into ``Q`` (``q_hidden_size``), ``K`` and ``V``
     (``kv_hidden_size`` each, in that order);
  2. ``NORMS`` (``eps is not None``): each K/Q head (``head_dim`` wide) is
     RMS-normalised, then scaled by the per-``head_dim`` weight and shifted by
     the (optional) bias.  When ``NORMS`` is false both weight and bias are
     skipped entirely;
  3. rotary is applied to the first ``ROPE_DIM`` dims of every Q and K head.
     Neox style rotates ``(d, d + ROPE_DIM//2)``; GPT-J style rotates
     ``(2d, 2d + 1)`` and broadcasts the first ``ROPE_DIM//2`` sin/cos values
     onto both lanes.  Dims in ``[ROPE_DIM, HEAD_DIM)`` pass through unchanged;
  4. ``V`` is a plain copy.

Source of truth: ``sgl_kernel_npu/.../norm/split_qkv_rmsnorm_rope.py``.
"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test

try:  # Ascend-only Triton extension; stock CUDA triton does not provide it.
    import triton.language.extra.cann.extension as al
except Exception:  # pragma: no cover - exercised on stock CUDA triton
    al = None

_HAS_CANN = al is not None


# --------------------------------------------------------------------------- #
# Triton kernel (copied verbatim from split_qkv_rmsnorm_rope.py)               #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Backend discovery + core count                                               #
# --------------------------------------------------------------------------- #
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


def _get_num_cores(device):
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    try:
        idx = 0 if device.index is None else device.index
        props = triton.runtime.driver.active.utils.get_device_properties(idx)
        if isinstance(props, dict):
            for key in ("num_vectorcore", "multiprocessor_count", "num_sm"):
                v = props.get(key)
                if v and v > 0:
                    return v
    except Exception:
        pass
    return 48


def _dtype_supported(device, dtype):
    try:
        torch.zeros(1, dtype=dtype, device=device)
        return True
    except Exception:
        return False


def _require_cann():
    if not _HAS_CANN:
        pytest.skip("triton CANN extension (al.extract_slice) not available")


# --------------------------------------------------------------------------- #
# Device-agnostic replica of ``split_qkv_rmsnorm_rope``                        #
# --------------------------------------------------------------------------- #
def split_qkv_rmsnorm_rope_impl(
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
    num_vectorcore = _get_num_cores(input.device)

    KV_BLOCK_SIZE = head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
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


# --------------------------------------------------------------------------- #
# Eager torch reference                                                       #
# --------------------------------------------------------------------------- #
def _rms_norm_last_dim(x, weight, bias, eps):
    """``x``: [..., head_dim] fp32; ``weight``/``bias`` broadcast on head_dim."""
    reciprocal_std = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = x * reciprocal_std
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _rope_first_dims(x_rot, sin, cos, is_neox_style):
    """``x_rot``: [..., ROPE_DIM] fp32; ``sin``/``cos``: [batch, 1, ROPE_DIM] fp32."""
    rope_dim = x_rot.shape[-1]
    half = rope_dim // 2
    if is_neox_style:
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        rot = torch.cat((-x2, x1), dim=-1)
        sin_use, cos_use = sin, cos
    else:
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        sin_use = sin[..., :half].repeat_interleave(2, dim=-1)
        cos_use = cos[..., :half].repeat_interleave(2, dim=-1)
    return rot * sin_use + x_rot * cos_use


def split_qkv_rmsnorm_rope_ref(
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
    """Eager torch reference computed in float32, cast back to the input dtype."""
    out_dtype = input.dtype
    batch = input.shape[0]
    rope_dim = sin.shape[-1]

    x = input.to(torch.float32)
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim

    q = x[:, :q_hidden_size].reshape(batch, num_q_heads, head_dim)
    k = x[:, q_hidden_size : q_hidden_size + kv_hidden_size].reshape(
        batch, num_kv_heads, head_dim
    )
    v = input[:, q_hidden_size + kv_hidden_size :]

    if eps is None:
        q_norm, k_norm = q, k
    else:
        q_norm = _rms_norm_last_dim(
            q,
            q_weight.to(torch.float32),
            q_bias.to(torch.float32) if q_bias is not None else None,
            eps,
        )
        k_norm = _rms_norm_last_dim(
            k,
            k_weight.to(torch.float32),
            k_bias.to(torch.float32) if k_bias is not None else None,
            eps,
        )

    sin_f = sin.to(torch.float32).reshape(batch, rope_dim)[:, None, :]
    cos_f = cos.to(torch.float32).reshape(batch, rope_dim)[:, None, :]

    q_rot = _rope_first_dims(q_norm[..., :rope_dim], sin_f, cos_f, is_neox_style)
    q_out = torch.cat((q_rot, q_norm[..., rope_dim:]), dim=-1).reshape(
        batch, q_hidden_size
    )

    k_rot = _rope_first_dims(k_norm[..., :rope_dim], sin_f, cos_f, is_neox_style)
    k_out = torch.cat((k_rot, k_norm[..., rope_dim:]), dim=-1).reshape(
        batch, kv_hidden_size
    )

    return q_out.to(out_dtype), k_out.to(out_dtype), v.clone()


# --------------------------------------------------------------------------- #
# pytest harness                                                              #
# --------------------------------------------------------------------------- #
_DETECT_ERR = ""
try:
    _DEVICE = _detect_device()
except Exception as _e:  # pragma: no cover - only hit on headless/CI hosts
    _DEVICE = None
    _DETECT_ERR = str(_e)

pytestmark = pytest.mark.skipif(_DEVICE is None, reason=f"no cuda/npu: {_DETECT_ERR}")

_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}

_EPS = 1e-6

# (batch_size, q_hidden_size, kv_hidden_size, head_dim, rope_dim) from 10shape.md.
_PERF_SHAPES = [
    (1, 2048, 512, 128, 128),
    (8, 2048, 512, 128, 128),
    (32, 2048, 512, 128, 128),
    (128, 2048, 512, 128, 128),
    (8, 4096, 1024, 128, 128),
    (8, 8192, 1024, 128, 128),
    (8, 2048, 512, 256, 128),
    (8, 2048, 512, 128, 64),
    (8, 1024, 256, 64, 64),
    (8, 4096, 512, 128, 128),
]


def _make_case(shape, dtype, seed, with_bias=True, with_norms=True):
    batch, q_hidden, kv_hidden, head_dim, rope_dim = shape
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    total = q_hidden + 2 * kv_hidden
    inp = torch.randn(batch, total, generator=gen, dtype=dtype, device=_DEVICE)
    sin = torch.randn(batch, rope_dim, generator=gen, dtype=dtype, device=_DEVICE)
    cos = torch.randn(batch, rope_dim, generator=gen, dtype=dtype, device=_DEVICE)

    q_weight = (
        torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
        if with_norms
        else None
    )
    k_weight = (
        torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
        if with_norms
        else None
    )
    q_bias = (
        torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
        if (with_norms and with_bias)
        else None
    )
    k_bias = (
        torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
        if (with_norms and with_bias)
        else None
    )
    return inp, sin, cos, q_weight, k_weight, q_bias, k_bias


def _run_case(shape, dtype, seed, is_neox=True, with_bias=True, with_norms=True):
    inp, sin, cos, q_weight, k_weight, q_bias, k_bias = _make_case(
        shape, dtype, seed, with_bias=with_bias, with_norms=with_norms
    )
    eps = _EPS if with_norms else None

    kwargs = dict(
        q_hidden_size=shape[1],
        kv_hidden_size=shape[2],
        head_dim=shape[3],
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        is_neox_style=is_neox,
    )
    q, k, v = split_qkv_rmsnorm_rope_impl(inp, sin, cos, **kwargs)
    q_ref, k_ref, v_ref = split_qkv_rmsnorm_rope_ref(inp, sin, cos, **kwargs)

    rtol, atol = _TOL[dtype]
    torch.testing.assert_close(
        q.to(torch.float32), q_ref.to(torch.float32), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        k.to(torch.float32), k_ref.to(torch.float32), rtol=rtol, atol=atol
    )
    # V is a pure copy and must be bit-exact.
    assert torch.equal(v, v_ref), "V copy mismatch"
    return q, k, v


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Q{s[1]}_KV{s[2]}_D{s[3]}_R{s[4]}",
)
def test_split_qkv_rmsnorm_rope_precision(dtype, shape):
    _require_cann()
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    _run_case(shape, dtype, seed=20240801 + shape[1] + shape[3])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_qkv_rmsnorm_rope_gptj_style(dtype):
    """GPT-J (non-Neox) interleaved rotation, including a partial-rope shape."""
    _require_cann()
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    _run_case((8, 2048, 512, 128, 128), dtype, seed=21, is_neox=False)
    _run_case((8, 2048, 512, 128, 64), dtype, seed=22, is_neox=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_qkv_rmsnorm_rope_no_norms(dtype):
    """``eps is None`` -> no RMSNorm and no weight/bias applied, rope only."""
    _require_cann()
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    _run_case((8, 1024, 256, 64, 64), dtype, seed=31, with_norms=False)
    _run_case((8, 2048, 512, 128, 128), dtype, seed=32, with_norms=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_qkv_rmsnorm_rope_no_bias(dtype):
    _require_cann()
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    _run_case((8, 4096, 1024, 128, 128), dtype, seed=41, with_bias=False)


def test_split_qkv_rmsnorm_rope_fp32_accuracy():
    """fp32 in / fp32 out should be far tighter than the half-precision budgets."""
    _require_cann()
    dtype = torch.float32
    _run_case((32, 2048, 512, 128, 128), dtype, seed=51)
    _run_case((32, 2048, 512, 128, 64), dtype, seed=52, is_neox=False)


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Q{s[1]}_KV{s[2]}_D{s[3]}_R{s[4]}",
)
def test_split_qkv_rmsnorm_rope_performance(shape):
    _require_cann()
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )
    inp, sin, cos, q_weight, k_weight, _, _ = _make_case(
        shape, dtype, seed=1234, with_bias=False
    )

    def fn_triton():
        split_qkv_rmsnorm_rope_impl(
            inp,
            sin,
            cos,
            shape[1],
            shape[2],
            shape[3],
            eps=_EPS,
            q_weight=q_weight,
            k_weight=k_weight,
        )

    n, q, kv, d, r = shape
    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=f"split_qkv_rmsnorm_rope_N{n}_Q{q}_KV{kv}_D{d}_R{r}",
    )
