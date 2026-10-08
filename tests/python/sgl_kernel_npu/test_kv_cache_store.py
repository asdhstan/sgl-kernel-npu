import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.kv_cache_store import (
    store_kv_cache_prefix_valid_npu_triton,
)


def store_kv_cache_prefix_valid_reference(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
):
    """Reference: copy only each batch row's committed prefix into the slots.

    ``loc_2d`` is ``[batch, width]`` and the source rows are its row-major
    flattening, so row ``r`` of batch ``b`` is ``src[b * width + r]``.  Rows at
    or beyond ``commit_lens[b]`` must be left untouched in the destination.
    """
    expected_k = dst_k.clone()
    expected_v = dst_v.clone()
    batch, width = loc_2d.shape
    loc_host = loc_2d.cpu().tolist()
    commit_host = commit_lens.cpu().tolist()
    for b in range(batch):
        for r in range(int(commit_host[b])):
            src_row = b * width + r
            slot = int(loc_host[b][r])
            expected_k[slot] = src_k[src_row]
            expected_v[slot] = src_v[src_row]
    return expected_k, expected_v


def _make_case(
    batch,
    width,
    heads,
    k_dim,
    v_dim,
    slots,
    commit_lens,
    dtype=torch.bfloat16,
    loc_dtype=torch.int32,
):
    torch.manual_seed(0)
    num_rows = batch * width
    assert num_rows <= slots, "slots must hold every source row"
    dst_k = torch.full((slots, heads, k_dim), -7.0, dtype=dtype, device="npu")
    dst_v = torch.full((slots, heads, v_dim), -9.0, dtype=dtype, device="npu")
    src_k = torch.randn(num_rows, heads, k_dim, dtype=dtype, device="npu")
    src_v = torch.randn(num_rows, heads, v_dim, dtype=dtype, device="npu")
    # A permutation keeps every (batch, row) pair on a distinct slot, so the
    # comparison does not depend on the order of concurrent writes.
    loc_2d = (
        torch.randperm(slots)[:num_rows].to(loc_dtype).reshape(batch, width).npu()
    )
    commit = torch.tensor(commit_lens, dtype=torch.int32, device="npu")
    return dst_k, dst_v, src_k, src_v, loc_2d, commit


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,width,heads,k_dim,v_dim,commit_lens",
    [
        (2, 4, 2, 64, 64, [4, 2]),  # partial prefix on the second batch
        (3, 2, 1, 128, 64, [2, 0, 1]),  # one batch commits nothing
        (1, 1, 2, 100, 100, [1]),  # dim not a multiple of the 128 tile
        (2, 3, 4, 128, 96, [3, 3]),  # K and V rows differ in width
    ],
)
def test_matches_reference(batch, width, heads, k_dim, v_dim, commit_lens, dtype, loc_dtype):
    slots = 32
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(
        batch,
        width,
        heads,
        k_dim,
        v_dim,
        slots,
        commit_lens,
        dtype=dtype,
        loc_dtype=loc_dtype,
    )
    expected_k, expected_v = store_kv_cache_prefix_valid_reference(
        dst_k, dst_v, src_k, src_v, loc_2d, commit
    )

    store_kv_cache_prefix_valid_npu_triton(dst_k, dst_v, src_k, src_v, loc_2d, commit)

    torch.testing.assert_close(dst_k, expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dst_v, expected_v, rtol=0.0, atol=0.0)


def test_zero_commit_lengths_leave_the_cache_untouched():
    slots = 16
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(
        2, 3, 2, 64, 64, slots, [0, 0]
    )
    dst_k_before = dst_k.clone()
    dst_v_before = dst_v.clone()

    store_kv_cache_prefix_valid_npu_triton(dst_k, dst_v, src_k, src_v, loc_2d, commit)

    torch.testing.assert_close(dst_k, dst_k_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dst_v, dst_v_before, rtol=0.0, atol=0.0)


def test_slot_row_stride_is_honoured():
    """The destination may be a row-strided view of the flattened cache (every
    other slot here), which the kernel addresses with the destination row
    strides it is given."""
    batch, width, heads, k_dim, v_dim, slots = 2, 3, 2, 64, 64, 8
    num_rows = batch * width
    dtype = torch.bfloat16
    torch.manual_seed(0)

    dst_k_full = torch.full(
        (slots * 2, heads, k_dim), -7.0, dtype=dtype, device="npu"
    )
    dst_v_full = torch.full(
        (slots * 2, heads, v_dim), -9.0, dtype=dtype, device="npu"
    )
    dst_k = dst_k_full[::2]
    dst_v = dst_v_full[::2]
    assert dst_k.stride(0) == 2 * heads * k_dim and dst_k.stride(-1) == 1
    assert dst_v.stride(0) == 2 * heads * v_dim and dst_v.stride(-2) == v_dim

    src_k = torch.randn(num_rows, heads, k_dim, dtype=dtype, device="npu")
    src_v = torch.randn(num_rows, heads, v_dim, dtype=dtype, device="npu")
    loc_2d = torch.randperm(slots)[:num_rows].to(torch.int32).reshape(batch, width).npu()
    commit = torch.tensor([width, width - 1], dtype=torch.int32, device="npu")

    expected_k, expected_v = store_kv_cache_prefix_valid_reference(
        dst_k, dst_v, src_k, src_v, loc_2d, commit
    )

    store_kv_cache_prefix_valid_npu_triton(dst_k, dst_v, src_k, src_v, loc_2d, commit)

    torch.testing.assert_close(dst_k, expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dst_v, expected_v, rtol=0.0, atol=0.0)


def test_loc_2d_must_be_rank_two():
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(2, 2, 1, 64, 64, 16, [2, 2])

    with pytest.raises(ValueError):
        store_kv_cache_prefix_valid_npu_triton(
            dst_k, dst_v, src_k, src_v, loc_2d.flatten(), commit
        )


def test_commit_lens_must_match_the_batch():
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(2, 2, 1, 64, 64, 16, [2, 2])

    with pytest.raises(ValueError):
        store_kv_cache_prefix_valid_npu_triton(
            dst_k, dst_v, src_k, src_v, loc_2d, commit[:1]
        )


def test_source_rows_must_match_the_location_grid():
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(2, 2, 1, 64, 64, 16, [2, 2])

    with pytest.raises(ValueError):
        store_kv_cache_prefix_valid_npu_triton(
            dst_k, dst_v, src_k[:-1], src_v, loc_2d, commit
        )


def test_kv_dtypes_must_match():
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(2, 2, 1, 64, 64, 16, [2, 2])
    src_v = src_v.to(torch.float16)

    with pytest.raises(ValueError):
        store_kv_cache_prefix_valid_npu_triton(
            dst_k, dst_v, src_k, src_v, loc_2d, commit
        )


def test_kv_row_shapes_must_match():
    dst_k, dst_v, src_k, src_v, loc_2d, commit = _make_case(2, 2, 1, 64, 64, 16, [2, 2])
    src_k = src_k[:, :, :32]

    with pytest.raises(ValueError):
        store_kv_cache_prefix_valid_npu_triton(
            dst_k, dst_v, src_k, src_v, loc_2d, commit
        )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
