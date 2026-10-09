"""Single-pass fused argmax + softmax-probability-of-argmax kernel.

Confidence-driven decoders (diffusion LLMs in particular) need, per row, the
argmax token and the softmax probability of that token -- the score compared
against an unmask threshold. Done portably that is a separate max / sub / exp /
sum / gather chain over ``[rows, vocab]``, reading the logits five to six times
and materializing full-width temporaries.

This kernel reads each logit once, carrying running ``(max, argmax, sumexp)`` in
registers via the online-softmax recurrence, and returns ``(argmax_id, 1/sumexp)``:
the argmax token's logit is the running max, so its ``exp(max - max)`` is 1 and
its softmax probability is exactly the reciprocal of the sum. Nothing of shape
``[rows, vocab]`` is ever written.

Measured on Ascend 910B3 at vocab 157184, bf16 logits, against an fp32
``argmax`` + ``softmax`` + ``gather`` reference: argmax identical on every row,
probability max relative error 3.5e-7, and no decision flips at a 0.5 threshold.
Against a chunked in-framework reduction: 6.6x at 32 rows (launch-bound) and
2.7x at 4096 rows.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


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


@lru_cache(maxsize=1)
def _num_cores() -> int:
    return get_device_properties()[1]


# Number of vocab chunks per row for the tuned path (see
# 1008test/perf_08_argmax_prob/notes.md).  The single-kernel path exposes only
# min(cores, B) programs, which is too little parallelism to stream a long
# vocab; splitting every row into _NUM_CHUNKS chunks raises the program count to
# B * _NUM_CHUNKS and a tiny merge kernel combines the per-chunk partials.
_NUM_CHUNKS = 16


@triton.jit
def _argmax_prob_partial_kernel(
    logits_ptr,  # [B, V]
    part_max_ptr,  # [B, NCHUNK] fp32
    part_sum_ptr,  # [B, NCHUNK] fp32
    part_arg_ptr,  # [B, NCHUNK] int64
    V,
    stride_b,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Running max / sum(exp) / argmax over one vocab chunk of one row."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    base = row * stride_b
    v_start = chunk * CHUNK
    v_end = tl.minimum(v_start + CHUNK, V)

    m = tl.full((), float("-inf"), tl.float32)
    arg = tl.zeros((), tl.int64)
    s = tl.zeros((), tl.float32)
    for start in range(v_start, v_end, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < v_end
        x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        # One reduction for both the tile max and the argmax position
        # (tie_break_left matches torch.argmax's lowest-index rule).
        chunk_max, chunk_idx = tl.max(x, axis=0, return_indices=True)
        new_m = tl.maximum(m, chunk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        chunk_arg = chunk_idx.to(tl.int64) + start
        # Strict >: a tie keeps the earlier index, matching torch.argmax.
        arg = tl.where(chunk_max > m, chunk_arg, arg)
        m = new_m
    tl.store(part_max_ptr + row * NCHUNK + chunk, m)
    tl.store(part_sum_ptr + row * NCHUNK + chunk, s)
    tl.store(part_arg_ptr + row * NCHUNK + chunk, arg)


@triton.jit
def _argmax_prob_merge_kernel(
    part_max_ptr,
    part_sum_ptr,
    part_arg_ptr,
    argmax_ptr,
    prob_ptr,
    NCHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Combine the per-chunk partials into the row's argmax id and probability."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < NCHUNK
    m = tl.load(part_max_ptr + row * NCHUNK + offs, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    s = tl.load(part_sum_ptr + row * NCHUNK + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(part_arg_ptr + row * NCHUNK + offs, mask=mask, other=0).to(tl.int64)

    row_max = tl.max(m, axis=0)
    total = tl.sum(tl.where(mask, s * tl.exp(m - row_max), 0.0), axis=0)
    # tl.argmax keeps the lowest index, i.e. the earliest chunk; `a` already
    # holds the lowest matching index inside the winning chunk.
    winner_idx = tl.argmax(tl.where(mask, m, float("-inf")), axis=0)
    winner = tl.sum(tl.where(offs == winner_idx, a, 0), axis=0)
    tl.store(argmax_ptr + row, winner)
    tl.store(prob_ptr + row, 1.0 / total)


# Vocab tile budget per program, in bytes. 32 KiB is the widest tile that still
# compiles for a 4-byte dtype on 910B3 (64 KiB overflows the unified buffer once
# auto-multi-buffering claims its share); 2-byte dtypes get twice the elements.
_TILE_BYTES = 32 * 1024


def argmax_softmax_prob_fused(
    logits: torch.Tensor, block_v: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row argmax id and the softmax probability of that argmax token.

    ``logits``: ``[B, V]``, any float dtype, last dim contiguous (an arbitrary
    row stride is fine, so a vocab-truncated view costs no copy). Returns
    ``(argmax_ids [B] int64, prob [B] float32)``. Accumulates in fp32; never
    materializes a ``[B, V]`` temporary.

    ``block_v`` overrides the vocab tile width; the default derives it from the
    logits dtype so the tile stays within the unified buffer.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1
    B, V = logits.shape
    argmax = torch.empty(B, dtype=torch.int64, device=logits.device)
    prob = torch.empty(B, dtype=torch.float32, device=logits.device)
    if block_v is None:
        block_v = _TILE_BYTES // logits.element_size()
    # A tile wider than the vocab only wastes unified buffer (and overflows it
    # once the single-iteration loop is unrolled).
    block_v = min(block_v, triton.next_power_of_2(V))

    # Split the vocab when the row is wide enough that per-row parallelism is
    # the limit; otherwise keep the single-kernel path.
    nchunks = max(1, min(_NUM_CHUNKS, triton.cdiv(V, block_v)))
    if nchunks <= 1:
        grid = (min(_num_cores(), B),)
        _argmax_prob_kernel[grid](
            logits, argmax, prob, B, V, logits.stride(0), BLOCK_V=block_v
        )
        return argmax, prob

    chunk = triton.cdiv(V, nchunks)
    part_max = torch.empty(B, nchunks, dtype=torch.float32, device=logits.device)
    part_sum = torch.empty(B, nchunks, dtype=torch.float32, device=logits.device)
    part_arg = torch.empty(B, nchunks, dtype=torch.int64, device=logits.device)
    _argmax_prob_partial_kernel[(B, nchunks)](
        logits,
        part_max,
        part_sum,
        part_arg,
        V,
        logits.stride(0),
        CHUNK=chunk,
        NCHUNK=nchunks,
        BLOCK_V=block_v,
    )
    _argmax_prob_merge_kernel[(B,)](
        part_max,
        part_sum,
        part_arg,
        argmax,
        prob,
        NCHUNK=nchunks,
        BLOCK_C=triton.next_power_of_2(nchunks),
    )
    return argmax, prob
