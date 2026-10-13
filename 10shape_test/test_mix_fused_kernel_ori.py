"""the native ``mix_fused`` (``NUM_CORES`` = vector cores, ``multibuffer=True``)"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test
import torch
import triton
import triton.language as tl
import pytest
from test_common import profiling_test

# num_warps_range = [1, 2, 4, 8, 16, 32, 64]
# superblock_factor_range = [1, 2, 4, 8, 16, 32]

# configs = []
# for n_w in num_warps_range:
#     for s_f in superblock_factor_range:
#         product = s_f * n_w
#         if 1 <= product <= 64:
#             configs.append(
#                 triton.Config(
#                     {'superblock_factor': s_f},
#                     num_warps = n_w
#                 )
#             )

# @triton.autotune(
#     configs=configs,
#     key=["superblock_factor"],
# )

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


def mix_fused_native(
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
        # compile_mode='simt_only',
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
        mix_fused_native(prefix, bank, b, cw, 1e-6)

    profiling_test(
        fn_triton, device.type, device, name=f"mix_fused_ori_N{n}_B{b}_H{h}"
    )

