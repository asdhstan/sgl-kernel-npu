"""Performance test for the *native* ``_argmax_prob_kernel`` launch configuration.

This is the un-tuned counterpart of ``argmax_prob_kernel.py``: the kernel is a
plain ``@triton.jit`` (no autotune) and the wrapper reproduces
``argmax_softmax_prob_fused`` exactly, i.e.

  * ``BLOCK_V = min(32 KiB // logits.element_size(), next_power_of_2(V))``
    -> 16384 for fp16/bf16, 8192 for fp32 (clamped by the vocab);
  * ``grid = (min(num_vectorcore, B),)``;
  * ``num_warps``/``num_stages``/``num_ctas`` are not passed, so the Ascend
    backend defaults apply (``num_warps=32``, ``num_stages=2`` -> auto
    multi-buffer on, ``num_ctas=1``; ``num_warps`` is only forwarded to
    bishengir in pure-SIMT mode, so it does not affect the default SIMD path).

Only the performance case is kept, so the native and the autotuned variant can
be profiled against each other; traces land in
``result_dir/argmax_prob_ori_<shape>`` to keep them apart.

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
    """``get_device_properties()[1]`` (num_vectorcore) from the repo, device-agnostic."""
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


# --------------------------------------------------------------------------- #
# Native wrapper logic of ``argmax_softmax_prob_fused``                        #
# --------------------------------------------------------------------------- #
# Vocab tile budget per program, in bytes. 32 KiB is the widest tile that still
# compiles for a 4-byte dtype (64 KiB overflows the unified buffer once
# auto-multi-buffering claims its share); 2-byte dtypes get twice the elements.
_TILE_BYTES = 32 * 1024


def argmax_prob_native(logits: torch.Tensor, block_v: int | None = None):
    """Device-agnostic replica of the native ``argmax_softmax_prob_fused``."""
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
def test_argmax_prob_native_performance(shape):
    B, V = shape
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )

    gen = torch.Generator(device=_DEVICE).manual_seed(1234 + B + V)
    logits = torch.randn(B, V, generator=gen, dtype=dtype, device=_DEVICE)

    def fn_triton():
        argmax_prob_native(logits)

    profiling_test(
        fn_triton, _DEVICE.type, _DEVICE, name=f"argmax_prob_ori_B{B}_V{V}"
    )
