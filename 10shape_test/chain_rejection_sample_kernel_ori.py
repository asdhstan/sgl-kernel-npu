"""the native ``_chain_rejection_sample_kernel`` launch"""

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


@triton.jit
def _chain_rejection_sample_kernel(
    predicts,
    target_probs,
    draft_probs,
    metadata,
    block_sums,
    uniform_samples_for_final_sampling,
    num_draft_tokens: tl.constexpr,
    num_draft_prob_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    vocab_block_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    pad_num_vocab_blocks: tl.constexpr,
):
    req_idx = tl.program_id(0)
    metadata_offset = req_idx * 3
    target_row = tl.load(metadata + metadata_offset).to(tl.int64)
    output_idx = tl.load(metadata + metadata_offset + 1).to(tl.int64)
    all_accepted = tl.load(metadata + metadata_offset + 2).to(tl.int32)

    block_offsets = tl.arange(0, pad_num_vocab_blocks)
    block_mask = block_offsets < num_vocab_blocks
    sums = tl.load(
        block_sums + req_idx * num_vocab_blocks + block_offsets,
        mask=block_mask,
        other=0.0,
    ).to(tl.float32)
    block_cdf = tl.cumsum(sums, axis=0)
    total = tl.sum(sums, axis=0)
    coin = tl.load(uniform_samples_for_final_sampling + req_idx).to(tl.float32)
    target_value = coin * total

    selected_block = tl.sum(
        ((block_cdf <= target_value) & block_mask).to(tl.int32), axis=0
    )
    selected_block = tl.minimum(selected_block, num_vocab_blocks - 1)
    prefix_sum = tl.sum(tl.where(block_offsets < selected_block, sums, 0.0), axis=0)
    local_target = tl.maximum(target_value - prefix_sum, 0.0)

    local_offsets = tl.arange(0, vocab_block_size)
    vocab_offsets = selected_block * vocab_block_size + local_offsets
    vocab_mask = vocab_offsets < vocab_size
    target_offset = (req_idx * num_draft_tokens + target_row) * vocab_size
    target = tl.load(
        target_probs + target_offset + vocab_offsets,
        mask=vocab_mask,
        other=0.0,
    ).to(tl.float32)
    if all_accepted == 1:
        residual = target
    else:
        draft_offset = (req_idx * num_draft_prob_rows + target_row) * vocab_size
        draft = tl.load(
            draft_probs + draft_offset + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.maximum(target - draft, 0.0)

    local_cdf = tl.cumsum(residual, axis=0)
    local_index = tl.sum(
        ((local_cdf <= local_target) & vocab_mask).to(tl.int32), axis=0
    )
    last_valid_local = tl.max(
        tl.where((residual > 0.0) & vocab_mask, local_offsets, -1), axis=0
    )
    valid_local_count = tl.minimum(
        vocab_block_size,
        vocab_size - selected_block * vocab_block_size,
    )
    sampled_local = tl.where(
        local_index < valid_local_count,
        local_index,
        last_valid_local,
    )
    sampled_token = tl.where(
        sampled_local >= 0,
        selected_block * vocab_block_size + sampled_local,
        vocab_size - 1,
    )
    tl.store(predicts + output_idx, tl.minimum(sampled_token, vocab_size - 1))


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


_VOCAB_BLOCK_SIZE = 2048


def chain_rejection_block_sum_ref(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    metadata: torch.Tensor,
    vocab_block_size: int = _VOCAB_BLOCK_SIZE,
) -> torch.Tensor:
    """Eager float32 reference for ``_chain_rejection_block_sum_kernel``."""
    batch_size, _, vocab_size = target_probs.shape
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    device = target_probs.device

    row = metadata[:, 0].to(torch.int64)
    all_accepted = metadata[:, 2].to(torch.int64)
    batch_idx = torch.arange(batch_size, device=device)

    target = target_probs.to(torch.float32)[batch_idx, row]
    residual = target.clone()
    rejected = all_accepted != 1
    if rejected.any():
        draft = draft_probs.to(torch.float32)[batch_idx[rejected], row[rejected]]
        residual[rejected] = (target[rejected] - draft).clamp_min(0.0)

    pad = num_vocab_blocks * vocab_block_size - vocab_size
    if pad:
        residual = torch.nn.functional.pad(residual, (0, pad))
    return residual.reshape(batch_size, num_vocab_blocks, vocab_block_size).sum(-1)


def chain_rejection_sample_native(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    metadata: torch.Tensor,
    block_sums: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    vocab_block_size: int = _VOCAB_BLOCK_SIZE,
) -> torch.Tensor:
    """Launch the final-token sampling kernel; returns ``predicts``."""
    batch_size, num_draft_tokens, vocab_size = target_probs.shape
    num_draft_prob_rows = draft_probs.shape[1]
    num_vocab_blocks = block_sums.shape[1]
    pad_num_vocab_blocks = triton.next_power_of_2(num_vocab_blocks)
    device = target_probs.device

    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    _chain_rejection_sample_kernel[(batch_size,)](
        predicts,
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        uniform_samples_for_final_sampling,
        num_draft_tokens=num_draft_tokens,
        num_draft_prob_rows=num_draft_prob_rows,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
        pad_num_vocab_blocks=pad_num_vocab_blocks,
    )
    return predicts


_DETECT_ERR = ""


try:
    _DEVICE = _detect_device()
except Exception as _e:  # pragma: no cover - only hit on headless/CI hosts
    _DEVICE = None
    _DETECT_ERR = str(_e)


pytestmark = pytest.mark.skipif(_DEVICE is None, reason=f"no cuda/npu: {_DETECT_ERR}")


_PERF_SHAPES = [
    (1, 5, 4, 157184, 2048, 77, 128, 1),
    (8, 5, 4, 157184, 2048, 77, 128, 1),
    (32, 5, 4, 157184, 2048, 77, 128, 1),
    (128, 5, 4, 157184, 2048, 77, 128, 1),
    (256, 5, 4, 157184, 2048, 77, 128, 1),
    (32, 5, 4, 157184, 2048, 77, 128, 0),
    (32, 8, 7, 157184, 2048, 77, 128, 1),
    (32, 5, 4, 32000, 2048, 16, 16, 1),
    (32, 5, 4, 4096, 2048, 2, 2, 1),
    (32, 5, 4, 256000, 2048, 125, 128, 1),
]


_SPIKE_STEP = 1024.0


_SPIKE_COUNT = 8


def _sparse_probs(batch, rows, vocab, gen):
    """``[batch, rows, vocab]`` fp32 with a few large, exactly-representable spikes.

    ``torch.Tensor.random_`` does *not* produce integers for float dtypes on NPU
    (unlike CPU), so the values are built explicitly from ``torch.randint``.
    """
    probs = torch.zeros(batch, rows, vocab, dtype=torch.float32, device=_DEVICE)
    pos = torch.randint(
        0, vocab, (batch, rows, _SPIKE_COUNT), generator=gen, device=_DEVICE
    )
    vals = torch.randint(
        1, 8, (batch, rows, _SPIKE_COUNT), generator=gen, device=_DEVICE
    ).to(torch.float32) * _SPIKE_STEP
    probs.scatter_(2, pos, vals)
    return probs


def _make_case(shape, seed):
    (
        batch_size,
        num_draft_tokens,
        num_draft_prob_rows,
        vocab_size,
        vocab_block_size,
        num_vocab_blocks,
        pad_num_vocab_blocks,
        all_accepted,
    ) = shape

    assert num_vocab_blocks == triton.cdiv(vocab_size, vocab_block_size)
    assert pad_num_vocab_blocks == triton.next_power_of_2(num_vocab_blocks)

    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    # Sparse spikes whose values are multiples of ``_SPIKE_STEP`` keep every
    # block sum / cumsum exactly representable, and leave >= _SPIKE_STEP between
    # consecutive CDF breakpoints.  That margin is many orders of magnitude above
    # the ~ulp rounding of ``coin * total``, so the discrete sampling decision is
    # reproducible instead of sitting on a floating-point boundary.
    target_probs = _sparse_probs(
        batch_size, num_draft_tokens, vocab_size, gen
    )
    draft_probs = _sparse_probs(
        batch_size, num_draft_prob_rows, vocab_size, gen
    )

    # When every draft is accepted the draft rows are never read, so the target
    # row may be the last draft token; otherwise it must fit draft_probs.
    target_row = (
        num_draft_tokens - 1 if all_accepted == 1 else num_draft_prob_rows - 1
    )
    metadata = torch.zeros(batch_size, 3, dtype=torch.int64, device=_DEVICE)
    metadata[:, 0] = target_row
    metadata[:, 1] = torch.arange(batch_size, device=_DEVICE) * num_draft_tokens
    metadata[:, 2] = all_accepted

    uniform = torch.rand(
        batch_size, generator=gen, dtype=torch.float32, device=_DEVICE
    )
    block_sums = chain_rejection_block_sum_ref(
        target_probs, draft_probs, metadata, vocab_block_size
    )
    return target_probs, draft_probs, metadata, block_sums, uniform


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Dt{s[1]}_Dr{s[2]}_V{s[3]}_B{s[4]}_NB{s[5]}_acc{s[7]}",
)
def test_chain_rejection_sample_performance(shape):
    vocab_block_size = shape[4]
    target_probs, draft_probs, metadata, block_sums, uniform = _make_case(
        shape, seed=1234
    )

    def fn_triton():
        chain_rejection_sample_native(
            target_probs,
            draft_probs,
            metadata,
            block_sums,
            uniform,
            vocab_block_size,
        )

    n, dt, dr, v, vbs, nvb, pad, acc = shape
    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=(
            f"chain_rejection_sample_ori_N{n}_Dt{dt}_Dr{dr}"
            f"_V{v}_B{vbs}_NB{nvb}_pad{pad}_acc{acc}"
        ),
    )


