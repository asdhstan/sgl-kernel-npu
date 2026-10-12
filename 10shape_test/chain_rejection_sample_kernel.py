"""Precision test for ``_chain_rejection_sample_kernel``.

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one request, independently):
  1. ``metadata[req] = [target_row, output_slot, all_accepted]`` and
     ``block_sums[req, :num_vocab_blocks]`` are produced by the accept and
     block-sum kernels;
  2. ``block_cdf = cumsum(block_sums)``; ``target_value = coin * sum(block_sums)``
     with ``coin = uniform_samples_for_final_sampling[req]``;
  3. the selected block is the number of blocks with ``block_cdf <= target_value``
     (clamped to the last block).  ``local_target = target_value - prefix`` where
     ``prefix`` is the sum of the preceding blocks;
  4. inside that block the residual is ``target_probs[req, target_row]`` when
     ``all_accepted == 1`` and ``max(target - draft, 0)`` otherwise;
  5. the sampled local index is the number of positions with
     ``local_cdf <= local_target``; if that runs past the valid (non-padded)
     positions of the block it falls back to the last position with a positive
     residual, or to ``vocab_size - 1`` when the block has no residual at all;
  6. the resulting token id is written to ``predicts[output_slot]``.

Source of truth: ``sgl_kernel_npu/.../sample/chain_speculative_sampling.py``.

Test data uses exactly representable (integer) probabilities: the block sums and
the cumsums are then exact, so the multi-way comparisons against the reference
are deterministic rather than dependent on summation-order rounding.
"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test


# --------------------------------------------------------------------------- #
# Triton kernel (copied verbatim from chain_speculative_sampling.py)           #
# --------------------------------------------------------------------------- #
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


_VOCAB_BLOCK_SIZE = 2048


# --------------------------------------------------------------------------- #
# Device-agnostic replicas of the block-sum and sample launch wrappers          #
# --------------------------------------------------------------------------- #
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


def chain_rejection_sample_impl(
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


def chain_rejection_sample_ref(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    metadata: torch.Tensor,
    block_sums: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    vocab_block_size: int = _VOCAB_BLOCK_SIZE,
) -> torch.Tensor:
    """Eager float32 reference for ``_chain_rejection_sample_kernel``.

    ``block_sums`` is taken as given (the kernel only consumes it), so this
    reference isolates the sampling decision from the block-sum reduction.
    """
    batch_size, num_draft_tokens, vocab_size = target_probs.shape
    num_vocab_blocks = block_sums.shape[1]
    device = target_probs.device

    target_f = target_probs.to(torch.float32)
    draft_f = draft_probs.to(torch.float32)

    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )

    for req in range(batch_size):
        target_row = int(metadata[req, 0])
        output_idx = int(metadata[req, 1])
        all_accepted = int(metadata[req, 2])

        sums = block_sums[req].to(torch.float32)
        block_cdf = torch.cumsum(sums, dim=0)
        total = sums.sum()
        target_value = uniform_samples_for_final_sampling[req] * total

        selected_block = int((block_cdf <= target_value).sum().item())
        selected_block = min(selected_block, num_vocab_blocks - 1)
        if selected_block > 0:
            prefix_sum = sums[:selected_block].sum()
        else:
            prefix_sum = sums.new_zeros(())
        local_target = torch.clamp_min(target_value - prefix_sum, 0.0)

        start = selected_block * vocab_block_size
        end = min(start + vocab_block_size, vocab_size)
        target = target_f[req, target_row, start:end]
        if all_accepted == 1:
            residual = target
        else:
            residual = (target - draft_f[req, target_row, start:end]).clamp_min(0.0)

        local_cdf = torch.cumsum(residual, dim=0)
        local_index = int((local_cdf <= local_target).sum().item())
        positive = torch.nonzero(residual > 0, as_tuple=False).flatten()
        last_valid_local = int(positive[-1]) if positive.numel() else -1

        valid_local_count = min(vocab_block_size, vocab_size - start)
        sampled_local = (
            local_index if local_index < valid_local_count else last_valid_local
        )
        if sampled_local >= 0:
            sampled_token = start + sampled_local
        else:
            sampled_token = vocab_size - 1
        predicts[output_idx] = min(sampled_token, vocab_size - 1)

    return predicts


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

# (batch_size, num_draft_tokens, num_draft_prob_rows, vocab_size,
#  vocab_block_size, num_vocab_blocks, pad_num_vocab_blocks, all_accepted)
# tuples from 10shape.md.
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


def _run_case(shape, seed):
    vocab_block_size = shape[4]
    target_probs, draft_probs, metadata, block_sums, uniform = _make_case(shape, seed)

    out = chain_rejection_sample_impl(
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        uniform,
        vocab_block_size,
    )
    ref = chain_rejection_sample_ref(
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        uniform,
        vocab_block_size,
    )

    assert out.shape == ref.shape == (shape[0] * shape[1],)
    assert out.dtype == torch.int32
    mismatch = (out != ref).sum().item()
    assert mismatch == 0, (
        f"token mismatch on {mismatch} rows, shape={shape}, "
        f"out={out.tolist()[:8]} ref={ref.tolist()[:8]}"
    )
    # Written slots must be real tokens.
    written = metadata[:, 1].to(torch.int64)
    assert (out[written] >= 0).all() and (out[written] < shape[3]).all()


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Dt{s[1]}_Dr{s[2]}_V{s[3]}_B{s[4]}_NB{s[5]}_acc{s[7]}",
)
def test_chain_rejection_sample_precision(shape):
    _run_case(shape, seed=20240801 + sum(shape))


def test_chain_rejection_sample_all_accepted():
    """``all_accepted == 1``: the residual is the raw target row."""
    _run_case((16, 5, 4, 4096, 2048, 2, 2, 1), seed=3)


def test_chain_rejection_sample_single_block():
    """A single vocab block removes the block-selection ambiguity entirely."""
    _run_case((8, 5, 4, 2048, 2048, 1, 1, 0), seed=4)


def test_chain_rejection_sample_ragged_last_block():
    """The last block is partially padded (vocab not a multiple of the block)."""
    _run_case((8, 5, 4, 5000, 2048, 3, 4, 1), seed=5)


def test_chain_rejection_sample_last_block_boundary():
    """Put all the mass in the final (padded) block to exercise the clamp."""
    batch_size, num_draft_tokens, vocab_size, vocab_block_size = 4, 5, 5000, 2048

    gen = torch.Generator(device=_DEVICE).manual_seed(6)
    target_probs = torch.zeros(
        batch_size, num_draft_tokens, vocab_size, dtype=torch.float32, device=_DEVICE
    )
    draft_probs = torch.zeros(
        batch_size, 4, vocab_size, dtype=torch.float32, device=_DEVICE
    )
    # Last valid index of the padded block plus its first index.
    last = vocab_size - 1
    target_probs[:, num_draft_tokens - 1, last] = 1.0
    target_probs[:, num_draft_tokens - 1, 4096] = 0.5

    metadata = torch.zeros(batch_size, 3, dtype=torch.int64, device=_DEVICE)
    metadata[:, 0] = num_draft_tokens - 1
    metadata[:, 1] = torch.arange(batch_size, device=_DEVICE) * num_draft_tokens
    metadata[:, 2] = 1

    uniform = torch.rand(
        batch_size, generator=gen, dtype=torch.float32, device=_DEVICE
    )
    block_sums = chain_rejection_block_sum_ref(
        target_probs, draft_probs, metadata, vocab_block_size
    )

    out = chain_rejection_sample_impl(
        target_probs, draft_probs, metadata, block_sums, uniform, vocab_block_size
    )
    ref = chain_rejection_sample_ref(
        target_probs, draft_probs, metadata, block_sums, uniform, vocab_block_size
    )

    assert torch.equal(out, ref)
    # Both spikes live in the padded last block, and the token must be one of them
    # (or the vocab-size fallback for the empty prefix before the spikes).
    written = metadata[:, 1].to(torch.int64)
    assert ((out[written] == 4096) | (out[written] == last)).all()


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
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
        chain_rejection_sample_impl(
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
            f"chain_rejection_sample_N{n}_Dt{dt}_Dr{dr}"
            f"_V{v}_B{vbs}_NB{nvb}_pad{pad}_acc{acc}"
        ),
    )
