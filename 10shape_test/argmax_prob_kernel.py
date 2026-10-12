"""Precision test for ``_argmax_prob_kernel`` (fused argmax + argmax probability).

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one ``row`` of a ``[B, V]`` logits matrix, independently):
  1. walk the vocabulary in ``BLOCK_V`` tiles, carrying an online-softmax
     ``(running_max, running_sumexp, argmax)`` triple in registers;
  2. ``argmax`` is the token with the largest logit; an exact tie keeps the
     *lower* index (strict ``>`` in the online update, first hit inside a tile);
  3. the argmax token's logit is the running max, so its ``exp(max - max)`` is
     ``1`` and its softmax probability is exactly ``1 / sumexp``.

Source of truth: ``sgl_kernel_npu/.../sample/argmax_softmax_prob.py``.
"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test


# --------------------------------------------------------------------------- #
# Triton kernel (copied verbatim from argmax_softmax_prob.py)                  #
# --------------------------------------------------------------------------- #
@triton.jit
def _argmax_prob_kernel(
    logits_ptr,  # [B, V]
    argmax_ptr,  # [B] int64
    prob_ptr,  # [B] float32
    B,
    V,
    stride_b,
    BLOCK_V: tl.constexpr,
):
    row_pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    for row in tl.range(row_pid, B, n_prog):
        base = row * stride_b
        m = tl.full((), float("-inf"), tl.float32)
        arg = tl.zeros((), tl.int64)
        s = tl.zeros((), tl.float32)
        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            chunk_max = tl.max(x, axis=0)
            new_m = tl.maximum(m, chunk_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
            chunk_arg = tl.argmax(x, axis=0).to(tl.int64) + start
            # Strict >: a tie keeps the earlier index, matching torch.argmax.
            arg = tl.where(chunk_max > m, chunk_arg, arg)
            m = new_m
        tl.store(argmax_ptr + row, arg)
        tl.store(prob_ptr + row, 1.0 / s)


# --------------------------------------------------------------------------- #
# Backend discovery + core count (grid size only affects perf, not accuracy)   #
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
    # Ascend NPU: replicate get_device_properties() from the repo.
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


# --------------------------------------------------------------------------- #
# Vocab tile budget per program, in bytes (same constant as the repo wrapper). #
# --------------------------------------------------------------------------- #
_TILE_BYTES = 32 * 1024


def argmax_prob_impl(logits: torch.Tensor, block_v=None):
    """Device-agnostic replica of ``argmax_softmax_prob_fused``."""
    assert logits.dim() == 2 and logits.stride(1) == 1
    B, V = logits.shape
    device = logits.device
    argmax = torch.empty(B, dtype=torch.int64, device=device)
    prob = torch.empty(B, dtype=torch.float32, device=device)
    grid = (min(_get_num_cores(device), B),)
    if block_v is None:
        block_v = _TILE_BYTES // logits.element_size()
    # A tile wider than the vocab only wastes unified buffer (and overflows it
    # once the single-iteration loop is unrolled).
    block_v = min(block_v, triton.next_power_of_2(V))
    _argmax_prob_kernel[grid](
        logits, argmax, prob, B, V, logits.stride(0), BLOCK_V=block_v
    )
    return argmax, prob


def argmax_prob_ref(logits: torch.Tensor):
    """Eager torch reference computed in float32."""
    ref = logits.float()
    argmax = ref.argmax(dim=-1)
    prob = torch.softmax(ref, dim=-1).gather(1, argmax.unsqueeze(1)).squeeze(1)
    return argmax, prob


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
    torch.float32: (1e-5, 1e-6),
    torch.float16: (1e-5, 1e-6),
    torch.bfloat16: (1e-5, 1e-6),
}

# (B, V) tuples from 10shape.md.
_PERF_SHAPES = [
    (1, 157184),
    (8, 157184),
    (32, 157184),
    (128, 157184),
    (512, 157184),
    (32, 32000),
    (32, 4096),
    (32, 256000),
    (32, 1024),
    (1024, 128000),
]


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"B{s[0]}_V{s[1]}",
)
def test_argmax_prob_precision(shape):
    B, V = shape
    dtype = torch.float16

    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    gen = torch.Generator(device=_DEVICE).manual_seed(20240801 + B + V)
    logits = torch.randn(B, V, generator=gen, dtype=dtype, device=_DEVICE)

    argmax, prob = argmax_prob_impl(logits)
    ref_argmax, ref_prob = argmax_prob_ref(logits)

    assert argmax.shape == ref_argmax.shape
    assert argmax.dtype == torch.int64
    assert prob.shape == ref_prob.shape
    assert prob.dtype == torch.float32
    # Ties keep the lower index on both sides, so the ids must match exactly.
    assert torch.equal(argmax, ref_argmax), (
        f"argmax mismatch: {(argmax != ref_argmax).sum().item()} rows, "
        f"shape={shape} dtype={dtype}"
    )
    rtol, atol = _TOL[dtype]
    torch.testing.assert_close(prob, ref_prob, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_argmax_prob_dtypes(dtype):
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    B, V = 32, 32000
    gen = torch.Generator(device=_DEVICE).manual_seed(11 * B + V)
    logits = torch.randn(B, V, generator=gen, dtype=dtype, device=_DEVICE)

    argmax, prob = argmax_prob_impl(logits)
    ref_argmax, ref_prob = argmax_prob_ref(logits)

    assert torch.equal(argmax, ref_argmax)
    rtol, atol = _TOL[dtype]
    torch.testing.assert_close(prob, ref_prob, rtol=rtol, atol=atol)


def test_argmax_prob_small_vocab_clamps_tile():
    """A vocab below the default tile must clamp BLOCK_V, not read past the row."""
    B, V = 8, 17
    gen = torch.Generator(device=_DEVICE).manual_seed(7)
    logits = torch.randn(B, V, generator=gen, dtype=torch.float32, device=_DEVICE)

    argmax, prob = argmax_prob_impl(logits)
    ref_argmax, ref_prob = argmax_prob_ref(logits)

    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


def test_argmax_prob_row_stride_is_honoured():
    """A vocab-truncated view must still address rows through ``stride(0)``."""
    padded = torch.randn(
        16, 4096, dtype=torch.bfloat16, device=_DEVICE
    )
    view = padded[:, :3000]
    assert view.stride(0) == 4096 and view.stride(1) == 1

    argmax, prob = argmax_prob_impl(view)
    ref_argmax, ref_prob = argmax_prob_ref(view)

    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


def test_argmax_prob_ties_keep_lower_index():
    """Matches torch.argmax: an exact tie keeps the earlier index, including
    across the kernel's vocab-tile boundary."""
    B, V = 4, 40000
    logits = torch.full((B, V), -5.0, dtype=torch.float32, device=_DEVICE)
    logits[:, 100] = 7.0
    logits[:, 30000] = 7.0

    argmax, prob = argmax_prob_impl(logits)
    ref_argmax, ref_prob = argmax_prob_ref(logits)

    assert torch.equal(argmax, torch.full((B,), 100, dtype=torch.int64, device=_DEVICE))
    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(prob, ref_prob, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"B{s[0]}_V{s[1]}",
)
def test_argmax_prob_performance(shape):
    B, V = shape
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )

    gen = torch.Generator(device=_DEVICE).manual_seed(1234 + B + V)
    logits = torch.randn(B, V, generator=gen, dtype=dtype, device=_DEVICE)

    def fn_triton():
        argmax_prob_impl(logits)

    profiling_test(
        fn_triton, _DEVICE.type, _DEVICE, name=f"argmax_prob_B{B}_V{V}"
    )
