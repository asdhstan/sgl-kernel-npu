"""Precision test for ``split_qkvgate_gemma_rmsnorm_rope_kernel``.

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one input row of ``[batch, 2*q + 2*kv]``, independently):
  1. the Q/gate region (``2 * q_hidden_size``) is laid out per head as
     ``[q_head | gate_head]`` (each ``head_dim`` wide): ``q`` and ``gate`` are
     split out of every ``2 * head_dim`` group;
  2. ``q``/``k`` heads are Gemma-RMS-normalised:
     ``x * rsqrt(mean(x**2) + eps) * (weight + 1)``;
  3. Neox-style rotary is applied to the first ``ROPE_DIM`` dims of every Q and
     K head: ``rot = concat(-x2, x1)`` with ``x1 = x[:ROPE_DIM//2]`` and
     ``x2 = x[ROPE_DIM//2:]``; dims in ``[ROPE_DIM, HEAD_DIM)`` pass through;
  4. ``V`` and ``gate`` are copied out unmodified.

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
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    qgate_hidden_size = q_hidden_size * 2

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_GATE_BLOCK_SIZE + tl.arange(0, Q_GATE_BLOCK_SIZE)
        valid_mask = col_indices < qgate_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM * 2)
        )
        q = al.extract_slice(
            input_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
            strides=(1, 1),
        ).reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)

        gate = al.extract_slice(
            input_values,
            offsets=(0, HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
            strides=(1, 1),
        ).reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)

        squares = q * q
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = q * reciprocal_std  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        normalized_values = normalized_values * weight_values

        # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
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

        # store
        col_indices_out = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask_out = col_indices_out < q_hidden_size
        tl.store(
            q_ptr + output_offset + col_indices_out,
            normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask_out,
        )
        tl.store(
            gate_ptr + output_offset + col_indices_out,
            gate.reshape(Q_BLOCK_SIZE).to(gate_ptr.dtype.element_ty),
            mask=valid_mask_out,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # k
    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0
    input_offset = row_pid * total_hidden_size + q_hidden_size * 2
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
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = (
            input_values * reciprocal_std
        )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        normalized_values = normalized_values * weight_values

        # # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)

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

        # store
        tl.store(
            k_ptr + output_offset + col_indices,
            normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # v
    input_offset = row_pid * total_hidden_size + q_hidden_size * 2 + kv_hidden_size
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
# Device-agnostic replica of ``split_qkvgate_gemma_rmsnorm_rope``               #
# --------------------------------------------------------------------------- #
def split_qkvgate_gemma_rmsnorm_rope_impl(
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
    num_vectorcore = _get_num_cores(input.device)

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
    )

    return q_output, k_output, v_output, gate_output


# --------------------------------------------------------------------------- #
# Eager torch reference                                                       #
# --------------------------------------------------------------------------- #
def _gemma_rms_norm_last_dim(x, weight, eps):
    """``x``: [..., head_dim] fp32; Gemma style uses ``weight + 1``."""
    reciprocal_std = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * reciprocal_std * (weight + 1.0)


def _rope_rotate_half(x_rot, sin, cos):
    """``x_rot``: [..., ROPE_DIM] fp32; ``sin``/``cos``: [batch, 1, ROPE_DIM]."""
    half = x_rot.shape[-1] // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return rot * sin + x_rot * cos


def split_qkvgate_gemma_rmsnorm_rope_ref(
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
    """Eager torch reference computed in float32, cast back to the input dtype."""
    out_dtype = input.dtype
    bs = input.shape[0]
    x = input.to(torch.float32)

    q_gate = x[:, : 2 * q_hidden_size].reshape(
        bs, q_hidden_size // head_dim, 2 * head_dim
    )
    q_in = q_gate[..., :head_dim]
    gate_in = q_gate[..., head_dim:]

    k_in = x[
        :, 2 * q_hidden_size : 2 * q_hidden_size + kv_hidden_size
    ].reshape(bs, kv_hidden_size // head_dim, head_dim)
    v_in = input[:, 2 * q_hidden_size + kv_hidden_size :]

    q = _gemma_rms_norm_last_dim(q_in, q_weight.to(torch.float32), eps)
    k = _gemma_rms_norm_last_dim(k_in, k_weight.to(torch.float32), eps)

    sin_f = sin.to(torch.float32).reshape(bs, rope_dim)[:, None, :]
    cos_f = cos.to(torch.float32).reshape(bs, rope_dim)[:, None, :]

    q_rot = _rope_rotate_half(q[..., :rope_dim], sin_f, cos_f)
    q_out = torch.cat((q_rot, q[..., rope_dim:]), dim=-1).reshape(bs, q_hidden_size)

    k_rot = _rope_rotate_half(k[..., :rope_dim], sin_f, cos_f)
    k_out = torch.cat((k_rot, k[..., rope_dim:]), dim=-1).reshape(bs, kv_hidden_size)

    gate_out = gate_in.reshape(bs, q_hidden_size)

    return (
        q_out.to(out_dtype),
        k_out.to(out_dtype),
        v_in.clone(),
        gate_out.to(out_dtype),
    )


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


def _make_case(shape, dtype, seed):
    batch, q_hidden, kv_hidden, head_dim, rope_dim = shape
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    total = 2 * q_hidden + 2 * kv_hidden
    inp = torch.randn(batch, total, generator=gen, dtype=dtype, device=_DEVICE)
    sin = torch.randn(batch, rope_dim, generator=gen, dtype=dtype, device=_DEVICE)
    cos = torch.randn(batch, rope_dim, generator=gen, dtype=dtype, device=_DEVICE)
    q_weight = torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
    k_weight = torch.randn(head_dim, generator=gen, dtype=dtype, device=_DEVICE)
    return inp, sin, cos, q_weight, k_weight


def _require_partial_rope(shape):
    """Reject ``rope_dim == head_dim`` shapes.

    The kernel unconditionally extracts the ``[ROPE_DIM, HEAD_DIM)`` pass-through
    slice, and ``al.extract_slice`` asserts ``size >= 1`` on every dimension, so a
    zero-width pass-through aborts during tracing.  The kernel therefore only
    supports ``rope_dim < head_dim``.
    """
    if shape[3] == shape[4]:
        pytest.skip(
            "split_qkvgate_gemma_rmsnorm_rope_kernel requires rope_dim < head_dim "
            f"(got head_dim=rope_dim={shape[3]}): the pass-through slice would be "
            "zero-width, which al.extract_slice rejects"
        )


def _run_case(shape, dtype, seed):
    _require_partial_rope(shape)
    batch, q_hidden, kv_hidden, head_dim, rope_dim = shape
    inp, sin, cos, q_weight, k_weight = _make_case(shape, dtype, seed)

    args = (inp, sin, cos, q_hidden, kv_hidden, head_dim, rope_dim, _EPS)
    q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope_impl(
        *args, q_weight, k_weight
    )
    q_ref, k_ref, v_ref, gate_ref = split_qkvgate_gemma_rmsnorm_rope_ref(
        *args, q_weight, k_weight
    )

    rtol, atol = _TOL[dtype]
    torch.testing.assert_close(
        q.to(torch.float32), q_ref.to(torch.float32), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        k.to(torch.float32), k_ref.to(torch.float32), rtol=rtol, atol=atol
    )
    # V and gate are pure copies and must be bit-exact.
    assert torch.equal(v, v_ref), "V copy mismatch"
    assert torch.equal(gate, gate_ref), "gate copy mismatch"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Q{s[1]}_KV{s[2]}_D{s[3]}_R{s[4]}",
)
def test_split_qkvgate_gemma_rmsnorm_rope_precision(dtype, shape):
    _require_cann()
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    _run_case(shape, dtype, seed=20240801 + shape[1] + shape[3])


def test_split_qkvgate_gemma_rmsnorm_rope_partial_rope_variants():
    """Partial rope only: the pass-through slice must be non-empty."""
    _require_cann()
    dtype = torch.bfloat16
    _run_case((8, 2048, 512, 128, 64), dtype, seed=61)
    _run_case((8, 2048, 512, 256, 128), dtype, seed=62)


def test_split_qkvgate_gemma_rmsnorm_rope_fp32_accuracy():
    _require_cann()
    dtype = torch.float32
    _run_case((32, 2048, 512, 128, 64), dtype, seed=71)
    _run_case((16, 4096, 1024, 128, 64), dtype, seed=72)


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Q{s[1]}_KV{s[2]}_D{s[3]}_R{s[4]}",
)
def test_split_qkvgate_gemma_rmsnorm_rope_performance(shape):
    _require_cann()
    _require_partial_rope(shape)
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )
    batch, q_hidden, kv_hidden, head_dim, rope_dim = shape
    inp, sin, cos, q_weight, k_weight = _make_case(shape, dtype, seed=1234)

    def fn_triton():
        split_qkvgate_gemma_rmsnorm_rope_impl(
            inp,
            sin,
            cos,
            q_hidden,
            kv_hidden,
            head_dim,
            rope_dim,
            _EPS,
            q_weight,
            k_weight,
        )

    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=(
            f"split_qkvgate_gemma_rmsnorm_rope_N{batch}"
            f"_Q{q_hidden}_KV{kv_hidden}_D{head_dim}_R{rope_dim}"
        ),
    )
