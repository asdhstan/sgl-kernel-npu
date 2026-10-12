"""Precision test for ``_mix_fused_kernel`` (attention-residual scored mixing).

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one ``token`` independently):
  1. candidate vectors = first ``B`` rows of ``bank[token]`` + ``prefix_sum[token]``
     -> ``B + 1`` vectors of size ``H``.
  2. RMS-normalize each candidate: ``1 / sqrt(mean(v**2) + eps)``.
  3. ``score = (normalized @ combined_weight)`` (elementwise-then-sum).
  4. softmax over the ``B + 1`` scores.
  5. ``output = sum_i prob_i * candidate_i``.

Source of truth: ``sgl-kernel-npu/.../kimi_k3/attn_residual.py``.
"""

import torch
import triton
import triton.language as tl

import pytest

from test_common import profiling_test


# --------------------------------------------------------------------------- #
# Triton kernel (copied verbatim from attn_residual.py, minus repo imports)    #
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["N", "B"])
def _mix_fused_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    out_ptr,
    N,
    B,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_om,
    H: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    token_start = pid * block_size
    if token_start >= N:
        return
    token_end = tl.minimum(token_start + block_size, N)

    hidden_offsets = tl.arange(0, H)
    row_offsets = tl.arange(0, NB)
    combined_weight = tl.load(cw_ptr + hidden_offsets).to(tl.float32)

    for token in range(token_start, token_end):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            inverse_rms = tl.rsqrt(tl.sum(value * value) / H + EPS)
            score = tl.sum(value * inverse_rms * combined_weight)
            scores = tl.where(row_offsets == row, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        probabilities = exp_scores / tl.sum(exp_scores)

        output = tl.zeros([H], dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
            output += probability * value

        tl.store(
            out_ptr + token * stride_om + hidden_offsets,
            output.to(out_ptr.dtype.element_ty),
        )


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


def mix_fused_impl(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Device-agnostic replica of ``mix_fused`` (without repo imports)."""
    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum

    device = prefix_sum.device
    output = torch.empty_like(prefix_sum)
    num_cores = _get_num_cores(device)

    launch_kwargs = {}
    if device.type != "cuda":
        # multibuffer is an Ascend Triton extension, unknown to stock CUDA triton.
        launch_kwargs["multibuffer"] = True

    _mix_fused_kernel[(num_cores,)](
        prefix_sum,
        bank,
        combined_weight,
        output,
        num_tokens,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        output.stride(0),
        H=hidden_size,
        EPS=float(variance_epsilon),
        NUM_CORES=num_cores,
        NB=triton.next_power_of_2(num_valid_blocks + 1),
        **launch_kwargs,
    )
    return output


def mix_fused_ref(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Eager torch reference computed in float32, cast back to input dtype."""
    out_dtype = prefix_sum.dtype
    prefix = prefix_sum.float()
    bank_f = bank.float()
    cw = combined_weight.float()

    candidates = torch.cat(
        [bank_f[:, :num_valid_blocks, :], prefix.unsqueeze(1)], dim=1
    )  # [N, B+1, H]

    rms = torch.sqrt((candidates * candidates).mean(-1, keepdim=True) + variance_epsilon)
    normalized = candidates / rms
    scores = (normalized * cw).sum(-1)  # [N, B+1]
    probs = torch.softmax(scores, dim=-1)
    out = (probs.unsqueeze(-1) * candidates).sum(1)  # [N, H]
    return out.to(out_dtype)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 512, 4, 16),
        (128, 256, 2, 8),
        (1, 1024, 1, 1),
        (257, 128, 8, 16),
    ],
    ids=lambda s: f"N{s[0]}_H{s[1]}_B{s[2]}_bank{s[3]}",
)
def test_mix_fused_precision(dtype, shape):
    N, H, B, bank_b = shape

    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    gen = torch.Generator(device=_DEVICE).manual_seed(20240801 + N + H)

    prefix = torch.randn(N, H, generator=gen, dtype=dtype, device=_DEVICE)
    bank = torch.randn(N, bank_b, H, generator=gen, dtype=dtype, device=_DEVICE)
    # bank width must fit B rows; the valid prefix width is B.
    combined_weight = torch.randn(H, generator=gen, dtype=dtype, device=_DEVICE)
    eps = 1e-6

    out = mix_fused_impl(prefix, bank, B, combined_weight, eps)
    ref = mix_fused_ref(prefix, bank, B, combined_weight, eps)

    rtol, atol = _TOL[dtype]
    max_abs = (out.float() - ref.float()).abs().max().item()

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"max_abs_diff={max_abs:.6e} dtype={dtype} shape={shape} "
        f"rtol={rtol} atol={atol}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mix_fused_zero_valid_blocks(dtype):
    """Edge case: num_valid_blocks == 0 -> only the prefix row is a candidate."""
    if not _dtype_supported(_DEVICE, dtype):
        pytest.skip(f"dtype {dtype} not supported on {_DEVICE}")

    N, H = 32, 256
    gen = torch.Generator(device=_DEVICE).manual_seed(7)
    prefix = torch.randn(N, H, generator=gen, dtype=dtype, device=_DEVICE)
    bank = torch.randn(N, 4, H, generator=gen, dtype=dtype, device=_DEVICE)
    cw = torch.randn(H, generator=gen, dtype=dtype, device=_DEVICE)

    out = mix_fused_impl(prefix, bank, 0, cw, 1e-6)
    ref = mix_fused_ref(prefix, bank, 0, cw, 1e-6)

    rtol, atol = _TOL[dtype]
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
# (N, B, H) tuples. bank width is set to B (i.e. bank == num_valid_blocks).
_PERF_SHAPES = [
    (1, 1, 128),
    (32, 1, 128),
    (1024, 1, 128),
    (32, 8, 128),
    (32, 64, 128),
    (32, 1, 256),
    (32, 1, 1024),
    (32, 1, 8192),
    (32, 2, 512),
    (256, 32, 2048),
]


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_B{s[1]}_H{s[2]}",
)
def test_mix_fused_performance(shape):
    n, b, h = shape
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )

    device = _DEVICE
    bank_b = max(b, 1)
    gen = torch.Generator(device=device).manual_seed(1234)
    prefix = torch.randn(n, h, generator=gen, dtype=dtype, device=device)
    bank = torch.randn(n, bank_b, h, generator=gen, dtype=dtype, device=device)
    cw = torch.randn(h, generator=gen, dtype=dtype, device=device)

    def fn_triton():
        mix_fused_impl(prefix, bank, b, cw, 1e-6)

    profiling_test(fn_triton, device.type, device)