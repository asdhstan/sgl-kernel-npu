import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused


def mix_fused_reference(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """fp32 reference for the kernel's score -> softmax -> mix pipeline.

    Candidates are the ``num_valid_blocks`` bank rows followed by ``prefix_sum``
    itself, matching the kernel's ``for row in range(B + 1)`` loop.  Every
    candidate is RMS-normalised, dotted with ``combined_weight`` to obtain its
    score, softmaxed, and the raw candidate values are mixed with those
    probabilities.
    """
    candidates = [bank[:, row, :].float() for row in range(num_valid_blocks)]
    candidates.append(prefix_sum.float())
    values = torch.stack(candidates, dim=1)  # [N, B + 1, H]

    hidden = values.shape[-1]
    inverse_rms = torch.rsqrt(values.pow(2).sum(-1) / hidden + variance_epsilon)
    scores = (values * inverse_rms.unsqueeze(-1) * combined_weight.float()).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * values).sum(dim=1)
    return mixed.to(prefix_sum.dtype)


def _make_inputs(
    num_tokens: int,
    hidden: int,
    num_valid_blocks: int,
    bank_rows: int = None,
    dtype: torch.dtype = torch.bfloat16,
):
    bank_rows = num_valid_blocks if bank_rows is None else bank_rows
    prefix_sum = torch.randn(num_tokens, hidden, dtype=dtype, device="npu")
    bank = torch.randn(num_tokens, bank_rows, hidden, dtype=dtype, device="npu")
    combined_weight = torch.randn(hidden, dtype=dtype, device="npu")
    return prefix_sum, bank, combined_weight


def _tolerance(dtype: torch.dtype) -> float:
    return 2e-2 if dtype == torch.bfloat16 else 2e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden,num_valid_blocks",
    [
        (8, 256, 3),
        (64, 512, 1),
        (1000, 128, 4),  # more tokens than vector cores, and not a multiple
    ],
)
def test_mix_fused_matches_reference(num_tokens, hidden, num_valid_blocks, dtype):
    torch.manual_seed(0)
    prefix_sum, bank, combined_weight = _make_inputs(
        num_tokens, hidden, num_valid_blocks, dtype=dtype
    )
    eps = 1e-5

    expected = mix_fused_reference(
        prefix_sum, bank, num_valid_blocks, combined_weight, eps
    )
    actual = mix_fused(prefix_sum, bank, num_valid_blocks, combined_weight, eps)

    torch.testing.assert_close(
        actual, expected, rtol=_tolerance(dtype), atol=_tolerance(dtype)
    )


def test_single_candidate_returns_prefix_sum_exactly():
    """With no valid bank blocks the prefix itself is the only candidate, so its
    softmax weight is 1 and the mix has to be the identity."""
    torch.manual_seed(0)
    prefix_sum, bank, combined_weight = _make_inputs(16, 256, 0, bank_rows=0)

    actual = mix_fused(prefix_sum, bank, 0, combined_weight, 1e-5)

    torch.testing.assert_close(actual, prefix_sum, rtol=0.0, atol=0.0)


def test_extra_bank_rows_are_ignored():
    """Only the first num_valid_blocks bank rows are candidates; padding rows
    (poisoned here) must not influence the result."""
    torch.manual_seed(0)
    num_tokens, hidden, num_valid_blocks = 32, 256, 2
    prefix_sum, bank, combined_weight = _make_inputs(
        num_tokens, hidden, num_valid_blocks, bank_rows=6
    )
    bank[:, num_valid_blocks:, :] = 1e4

    expected = mix_fused_reference(
        prefix_sum, bank, num_valid_blocks, combined_weight, 1e-5
    )
    actual = mix_fused(prefix_sum, bank, num_valid_blocks, combined_weight, 1e-5)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_bank_row_stride_is_honoured():
    """The wrapper passes the bank token/row strides, so a bank view whose token
    stride is wider than its row block must still address the right rows."""
    torch.manual_seed(0)
    num_tokens, hidden, num_valid_blocks = 8, 128, 2
    padded = torch.randn(
        num_tokens,
        num_valid_blocks + 1,
        hidden,
        dtype=torch.bfloat16,
        device="npu",
    )
    bank = padded[:, :num_valid_blocks, :]
    assert bank.stride(0) == (num_valid_blocks + 1) * hidden
    prefix_sum = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")
    combined_weight = torch.randn(hidden, dtype=torch.bfloat16, device="npu")

    expected = mix_fused_reference(
        prefix_sum, bank, num_valid_blocks, combined_weight, 1e-5
    )
    actual = mix_fused(prefix_sum, bank, num_valid_blocks, combined_weight, 1e-5)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_valid_blocks", [-1, 4])
def test_num_valid_blocks_out_of_range_raises(num_valid_blocks):
    prefix_sum, bank, combined_weight = _make_inputs(4, 64, 3)

    with pytest.raises(ValueError):
        mix_fused(prefix_sum, bank, num_valid_blocks, combined_weight, 1e-5)


def test_empty_tokens_is_a_noop():
    prefix_sum = torch.empty(0, 128, dtype=torch.bfloat16, device="npu")
    bank = torch.empty(0, 2, 128, dtype=torch.bfloat16, device="npu")
    combined_weight = torch.randn(128, dtype=torch.bfloat16, device="npu")

    assert mix_fused(prefix_sum, bank, 2, combined_weight, 1e-5) is prefix_sum


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
