"""Precision test for ``_chain_rejection_block_sum_kernel``.

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one ``(request, vocab block)`` pair, independently):
  1. ``metadata[req] = [target_row, output_slot, all_accepted]``;
  2. the target residual is

     - ``residual = target_probs[req, target_row]`` when ``all_accepted == 1``,
     - ``residual = max(target_probs[req, target_row]
                        - draft_probs[req, target_row], 0)`` otherwise;

  3. ``block_sums[req, block]`` is the sum of that residual over the vocab
     columns ``[block * vocab_block_size, (block + 1) * vocab_block_size)``
     (columns past ``vocab_size`` contribute zero).

Source of truth: ``sgl_kernel_npu/.../sample/chain_speculative_sampling.py``.
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
def _chain_rejection_block_sum_kernel(
    target_probs,
    draft_probs,
    metadata,
    block_sums,
    batch_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    num_draft_prob_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    vocab_block_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_blocks = batch_size * num_vocab_blocks

    for flat_block_idx in tl.range(program_idx, total_blocks, num_programs):
        req_idx = flat_block_idx // num_vocab_blocks
        block_idx = flat_block_idx % num_vocab_blocks
        vocab_offsets = block_idx * vocab_block_size + tl.arange(0, vocab_block_size)
        vocab_mask = vocab_offsets < vocab_size

        metadata_offset = req_idx * 3
        target_row = tl.load(metadata + metadata_offset).to(tl.int64)
        all_accepted = tl.load(metadata + metadata_offset + 2).to(tl.int32)
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

        tl.store(
            block_sums + req_idx * num_vocab_blocks + block_idx,
            tl.sum(residual, axis=0),
        )


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


def chain_rejection_block_sum_impl(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    metadata: torch.Tensor,
    vocab_block_size: int = _VOCAB_BLOCK_SIZE,
    num_programs: int | None = None,
) -> torch.Tensor:
    """Launch the block-sum kernel; ``metadata`` is the accept kernel's output."""
    batch_size, num_draft_tokens, vocab_size = target_probs.shape
    num_draft_prob_rows = draft_probs.shape[1]
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    device = target_probs.device

    if num_programs is None:
        # Same cap as the repo wrapper: one program per (request, block) pair,
        # capped at the physical vector-core count.
        num_programs = min(batch_size * num_vocab_blocks, _get_num_cores(device))

    block_sums = torch.empty(
        (batch_size, num_vocab_blocks), dtype=torch.float32, device=device
    )
    _chain_rejection_block_sum_kernel[(num_programs,)](
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        num_draft_prob_rows=num_draft_prob_rows,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
    )
    return block_sums


def chain_rejection_block_sum_ref(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    metadata: torch.Tensor,
    vocab_block_size: int = _VOCAB_BLOCK_SIZE,
) -> torch.Tensor:
    """Eager float32 reference; draft rows are only read for rejected requests."""
    batch_size, _, vocab_size = target_probs.shape
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    device = target_probs.device

    row = metadata[:, 0].to(torch.int64)
    all_accepted = metadata[:, 2].to(torch.int64)
    batch_idx = torch.arange(batch_size, device=device)

    target = target_probs.to(torch.float32)[batch_idx, row]  # [B, V]
    residual = target.clone()
    rejected = all_accepted != 1
    if rejected.any():
        draft = draft_probs.to(torch.float32)[batch_idx[rejected], row[rejected]]
        residual[rejected] = (target[rejected] - draft).clamp_min(0.0)

    pad = num_vocab_blocks * vocab_block_size - vocab_size
    if pad:
        residual = torch.nn.functional.pad(residual, (0, pad))
    return residual.reshape(batch_size, num_vocab_blocks, vocab_block_size).sum(-1)


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
#  vocab_block_size, num_vocab_blocks, all_accepted) tuples from 10shape.md.
_PERF_SHAPES = [
    (1, 5, 4, 157184, 2048, 77, 1),
    (1, 5, 4, 2048, 2048, 1, 1),
    (8, 5, 4, 157184, 2048, 77, 1),
    (32, 5, 4, 157184, 2048, 77, 1),
    (128, 5, 4, 157184, 2048, 77, 1),
    (256, 5, 4, 157184, 2048, 77, 1),
    (32, 5, 4, 157184, 2048, 77, 0),
    (32, 8, 7, 157184, 2048, 77, 1),
    (32, 5, 4, 32000, 2048, 16, 1),
    (32, 5, 4, 256000, 2048, 125, 1),
]


def _make_case(shape, seed):
    (
        batch_size,
        num_draft_tokens,
        num_draft_prob_rows,
        vocab_size,
        vocab_block_size,
        num_vocab_blocks,
        all_accepted,
    ) = shape

    assert num_vocab_blocks == triton.cdiv(vocab_size, vocab_block_size)

    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    # Non-negative "probabilities"; normalisation is irrelevant to the kernel.
    target_probs = torch.rand(
        batch_size,
        num_draft_tokens,
        vocab_size,
        generator=gen,
        dtype=torch.float32,
        device=_DEVICE,
    )
    draft_probs = torch.rand(
        batch_size,
        num_draft_prob_rows,
        vocab_size,
        generator=gen,
        dtype=torch.float32,
        device=_DEVICE,
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
    return target_probs, draft_probs, metadata


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Dt{s[1]}_Dr{s[2]}_V{s[3]}_B{s[4]}_NB{s[5]}_acc{s[6]}",
)
def test_chain_rejection_block_sum_precision(shape):
    vocab_block_size = shape[4]
    target_probs, draft_probs, metadata = _make_case(
        shape, seed=20240801 + sum(shape)
    )

    out = chain_rejection_block_sum_impl(
        target_probs, draft_probs, metadata, vocab_block_size
    )
    ref = chain_rejection_block_sum_ref(
        target_probs, draft_probs, metadata, vocab_block_size
    )

    assert out.shape == ref.shape == (shape[0], shape[5])
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_chain_rejection_block_sum_single_block():
    """``num_vocab_blocks == 1``: the whole vocab is one block."""
    shape = (1, 5, 4, 2048, 2048, 1, 1)
    target_probs, draft_probs, metadata = _make_case(shape, seed=3)

    out = chain_rejection_block_sum_impl(target_probs, draft_probs, metadata, 2048)
    ref = chain_rejection_block_sum_ref(target_probs, draft_probs, metadata, 2048)

    assert out.shape == (1, 1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_chain_rejection_block_sum_ragged_vocab():
    """A vocab that is not a multiple of the block size stays within bounds."""
    shape = (4, 5, 4, 5000, 2048, 3, 0)
    target_probs, draft_probs, metadata = _make_case(shape, seed=4)

    out = chain_rejection_block_sum_impl(target_probs, draft_probs, metadata, 2048)
    ref = chain_rejection_block_sum_ref(target_probs, draft_probs, metadata, 2048)

    assert out.shape == (4, 3)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_chain_rejection_block_sum_fewer_programs_than_blocks():
    """Force the strided program loop by capping the grid below the block count."""
    shape = (32, 5, 4, 157184, 2048, 77, 1)
    target_probs, draft_probs, metadata = _make_case(shape, seed=5)

    out = chain_rejection_block_sum_impl(
        target_probs, draft_probs, metadata, 2048, num_programs=7
    )
    ref = chain_rejection_block_sum_ref(target_probs, draft_probs, metadata, 2048)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"N{s[0]}_Dt{s[1]}_Dr{s[2]}_V{s[3]}_B{s[4]}_NB{s[5]}_acc{s[6]}",
)
def test_chain_rejection_block_sum_performance(shape):
    vocab_block_size = shape[4]
    target_probs, draft_probs, metadata = _make_case(shape, seed=1234)

    def fn_triton():
        chain_rejection_block_sum_impl(
            target_probs, draft_probs, metadata, vocab_block_size
        )

    profiling_test(fn_triton, _DEVICE.type, _DEVICE)
