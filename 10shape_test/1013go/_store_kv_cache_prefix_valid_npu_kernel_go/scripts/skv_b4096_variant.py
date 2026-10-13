"""Precision test for ``_store_kv_cache_prefix_valid_npu_kernel``.

This is a standalone script: it does NOT import ``sgl_kernel_npu``, so it can be
run with pytest directly against either a CUDA or an Ascend NPU backend.

Kernel semantics (one ``src_row`` of a flattened ``[batch * prefix_width]`` row
matrix, independently):
  1. ``src_row`` decomposes into ``batch_id = src_row // PREFIX_WIDTH`` and
     ``row_in_batch = src_row % PREFIX_WIDTH``;
  2. the row is active only when ``row_in_batch < commit_lens[batch_id]``;
  3. ``loc = loc_2d[batch_id, row_in_batch]`` selects the destination cache slot;
  4. the first ``K_ROW_DIM``/``V_ROW_DIM`` elements of the source K/V row are
     copied into that slot (axis 2 of the grid selects K vs V).

Rows that are not committed must be left completely untouched.

Source of truth: ``sgl_kernel_npu/.../mem_cache/kv_cache_store.py``.
"""

import pytest
import torch
import triton
import triton.language as tl

from test_common import profiling_test


# --------------------------------------------------------------------------- #
# Triton kernel (body copied verbatim from kv_cache_store.py; the autotune      #
# decorator is added here, following topk_index_kernel.py)                      #
# --------------------------------------------------------------------------- #
# Tunable: the channel block each K/V row is split into.  Both row dimensions are
# constexpr, so they double as the autotune key; the grid follows BLOCK via META.
_SKV_CONFIGS = [
    triton.Config({"BLOCK": block}) for block in (4096,)
]


@triton.autotune(
    configs=_SKV_CONFIGS,
    key=["K_ROW_DIM", "V_ROW_DIM", "PREFIX_WIDTH"],
)
@triton.jit
def _store_kv_cache_prefix_valid_npu_kernel(
    dst_k_ptr,
    dst_v_ptr,
    src_k_ptr,
    src_v_ptr,
    loc_ptr,
    commit_lens_ptr,
    dst_k_row_stride,
    dst_v_row_stride,
    src_k_row_stride,
    src_v_row_stride,
    K_ROW_DIM: tl.constexpr,
    V_ROW_DIM: tl.constexpr,
    PREFIX_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    src_row = tl.program_id(0)
    block_id = tl.program_id(1)
    kv_id = tl.program_id(2)

    batch_id = src_row // PREFIX_WIDTH
    row_in_batch = src_row % PREFIX_WIDTH
    active = row_in_batch < tl.load(commit_lens_ptr + batch_id)
    loc = tl.load(loc_ptr + src_row, mask=active, other=0).to(tl.int64)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)

    if kv_id == 0:
        mask = active & (offsets < K_ROW_DIM)
        values = tl.load(
            src_k_ptr + src_row * src_k_row_stride + offsets,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_k_ptr + loc * dst_k_row_stride + offsets,
            values,
            mask=mask,
        )
    else:
        mask = active & (offsets < V_ROW_DIM)
        values = tl.load(
            src_v_ptr + src_row * src_v_row_stride + offsets,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_v_ptr + loc * dst_v_row_stride + offsets,
            values,
            mask=mask,
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


# --------------------------------------------------------------------------- #
# Device-agnostic replica of the repo wrapper (validation kept verbatim)       #
# --------------------------------------------------------------------------- #
def store_kv_cache_prefix_valid_impl(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Commit valid K/V row prefixes into an NPU KV cache with Triton-Ascend."""
    if loc_2d.ndim != 2:
        raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}")
    if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
        raise ValueError(
            "commit_lens must match loc_2d batch size: "
            f"commit_lens={tuple(commit_lens.shape)}, loc_2d={tuple(loc_2d.shape)}"
        )
    if dst_k.ndim != 3 or dst_v.ndim != 3:
        raise ValueError(
            "dst_k and dst_v must be flattened [slots, heads, dim] views, "
            f"got {tuple(dst_k.shape)} and {tuple(dst_v.shape)}"
        )
    if src_k.ndim != 3 or src_v.ndim != 3:
        raise ValueError(
            "src_k and src_v must be [rows, heads, dim], "
            f"got {tuple(src_k.shape)} and {tuple(src_v.shape)}"
        )

    num_rows = loc_2d.numel()
    if src_k.shape[0] != num_rows or src_v.shape[0] != num_rows:
        raise ValueError(
            "source KV rows must match loc size: "
            f"src_k={tuple(src_k.shape)}, src_v={tuple(src_v.shape)}, "
            f"loc_2d={tuple(loc_2d.shape)}"
        )
    if dst_k.shape[1:] != src_k.shape[1:]:
        raise ValueError(
            f"K row shape mismatch: dst={tuple(dst_k.shape)}, src={tuple(src_k.shape)}"
        )
    if dst_v.shape[1:] != src_v.shape[1:]:
        raise ValueError(
            f"V row shape mismatch: dst={tuple(dst_v.shape)}, src={tuple(src_v.shape)}"
        )
    if not (dst_k.dtype == dst_v.dtype == src_k.dtype == src_v.dtype):
        raise ValueError(
            "K/V dtypes must match: "
            f"dst_k={dst_k.dtype}, dst_v={dst_v.dtype}, "
            f"src_k={src_k.dtype}, src_v={src_v.dtype}"
        )
    if not (
        dst_k.device
        == dst_v.device
        == src_k.device
        == src_v.device
        == loc_2d.device
        == commit_lens.device
    ):
        raise ValueError(
            "all K/V, location, and commit-length tensors must be on the same device"
        )
    if any(
        tensor.stride(-1) != 1 or tensor.stride(-2) != tensor.shape[-1]
        for tensor in (dst_k, dst_v, src_k, src_v)
    ):
        raise ValueError("K/V head and head-dim axes must be contiguous")
    if num_rows == 0:
        return

    commit_lens = commit_lens.contiguous()
    loc_2d = loc_2d.contiguous()
    prefix_width = loc_2d.shape[1]
    k_row_dim = src_k.shape[1] * src_k.shape[2]
    v_row_dim = src_v.shape[1] * src_v.shape[2]
    def grid(META):
        return (
            num_rows,
            triton.cdiv(max(k_row_dim, v_row_dim), META["BLOCK"]),
            2,
        )
    _store_kv_cache_prefix_valid_npu_kernel[grid](
        dst_k,
        dst_v,
        src_k,
        src_v,
        loc_2d,
        commit_lens,
        dst_k.stride(0),
        dst_v.stride(0),
        src_k.stride(0),
        src_v.stride(0),
        K_ROW_DIM=k_row_dim,
        V_ROW_DIM=v_row_dim,
        PREFIX_WIDTH=prefix_width,
    )


def store_kv_cache_prefix_valid_ref(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
):
    """Eager torch reference: scatter the committed source rows into their slots."""
    dst_k = dst_k.clone()
    dst_v = dst_v.clone()
    batch, width = loc_2d.shape
    device = loc_2d.device

    cols = torch.arange(width, device=device)[None, :]
    rows = torch.arange(batch, device=device)[:, None] * width + cols
    active = cols < commit_lens[:, None].to(torch.int64)  # [batch, width]

    rows = rows[active]
    locs = loc_2d[active].to(torch.int64)
    if rows.numel():
        dst_k[locs] = src_k[rows]
        dst_v[locs] = src_v[rows]
    return dst_k, dst_v


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

# (batch, prefix_width, num_heads, head_dim, commit_lens) tuples from 10shape.md.
_PERF_SHAPES = [
    (1, 1, 8, 128, 1),
    (1, 4, 8, 128, 4),
    (1, 8, 8, 128, 8),
    (1, 16, 8, 128, 16),
    (1, 32, 8, 128, 32),
    (1, 64, 8, 128, 64),
    (1, 16, 1, 128, 16),
    (1, 16, 32, 128, 16),
    (4, 16, 8, 128, 8),
    (1, 16, 8, 64, 16),
]

# Exactly representable in fp32/fp16/bf16, so the "untouched" check is exact.
_SENTINEL = 7.5


def _make_case(shape, dtype, seed):
    batch, prefix_width, num_heads, head_dim, commit_lens = shape
    num_rows = batch * prefix_width
    num_slots = num_rows + 5  # a few slots that must stay untouched

    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    dst_k = torch.full(
        (num_slots, num_heads, head_dim), _SENTINEL, dtype=dtype, device=_DEVICE
    )
    dst_v = torch.full(
        (num_slots, num_heads, head_dim), _SENTINEL, dtype=dtype, device=_DEVICE
    )
    src_k = torch.randn(
        num_rows, num_heads, head_dim, generator=gen, dtype=dtype, device=_DEVICE
    )
    src_v = torch.randn(
        num_rows, num_heads, head_dim, generator=gen, dtype=dtype, device=_DEVICE
    )
    # A permutation keeps every destination slot distinct (no write races).
    loc = (
        torch.randperm(num_slots, generator=gen, device=_DEVICE)[:num_rows]
        .to(torch.int32)
        .reshape(batch, prefix_width)
    )
    commit = torch.full(
        (batch,), commit_lens, dtype=torch.int32, device=_DEVICE
    )
    return dst_k, dst_v, src_k, src_v, loc, commit








# --------------------------------------------------------------------------- #
# Performance validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"B{s[0]}_W{s[1]}_H{s[2]}_D{s[3]}_C{s[4]}",
)
def test_store_kv_cache_prefix_valid_performance(shape):
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )
    (
        dst_k,
        dst_v,
        src_k,
        src_v,
        loc,
        commit,
    ) = _make_case(shape, dtype, seed=1234)

    def fn_triton():
        store_kv_cache_prefix_valid_impl(dst_k, dst_v, src_k, src_v, loc, commit)

    b, w, h, d, c = shape
    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=f"skv_b4096_B{b}_W{w}_H{h}_D{d}_C{c}",
    )
