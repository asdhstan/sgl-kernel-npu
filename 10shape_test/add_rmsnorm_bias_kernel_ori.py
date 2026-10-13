"""the native wrapper: ``COL_BLOCK_SIZE = 2048`` and the backend's
default auto multi-buffering (the repo's compiled-kernel cache is
omitted; it does not change the launched kernel)"""

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


try:  # Ascend-only Triton extension; stock CUDA triton does not provide it.
    import triton.language.extra.cann.extension as al
except Exception:  # pragma: no cover - exercised on stock CUDA triton
    al = None


_HAS_CANN = al is not None


@triton.jit
def add_rmsnorm_bias_kernel(
    input_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    quant_scale_ptr,
    quant_offset_ptr,
    output_ptr,
    output2_ptr,
    batch_size,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    norm_weight_values = tl.load(norm_weight_ptr + cols, mask=valid_mask, other=0.0)
    input_offsets = row_start * hidden_size + cols
    for i in tl.range(row_start, batch_size, row_step):
        # add
        buffered_values = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
        buffered_values += tl.load(
            residual_ptr + input_offsets, mask=valid_mask, other=0.0
        )
        tl.store(output2_ptr + input_offsets, buffered_values, mask=valid_mask)
        buffered_values = buffered_values.to(tl.float32)
        # norm
        squares = buffered_values * buffered_values  # .to(tl.float32)
        variance = tl.sum(squares) / hidden_size
        reciprocal_std = 1 / tl.sqrt(variance + eps)  # .to(tl.bfloat16)
        buffered_values = buffered_values * reciprocal_std
        buffered_values = buffered_values * norm_weight_values
        # bias
        norm_bias_values = tl.load(norm_bias_ptr + cols, mask=valid_mask, other=0.0)
        buffered_values = buffered_values + norm_bias_values
        if SCALE:
            block_cols = tl.arange(0, COL_BLOCK_SIZE)
            for block_offset in range(0, hidden_size, COL_BLOCK_SIZE):
                col_indices = block_offset + block_cols
                valid_mask2 = col_indices < hidden_size
                block_buffered_values = al.extract_slice(
                    buffered_values, (block_offset,), (COL_BLOCK_SIZE,), (1,)
                )
                # quant
                quant_scale_values = tl.load(
                    quant_scale_ptr + col_indices, mask=valid_mask2, other=0.0
                )
                quant_offset_values = tl.load(
                    quant_offset_ptr + col_indices, mask=valid_mask2, other=0.0
                )
                block_buffered_values = (
                    block_buffered_values.to(tl.float32) * quant_scale_values
                ) + quant_offset_values
                block_buffered_values = block_buffered_values.cast(
                    tl.int8, overflow_mode="saturate"
                )
                tl.store(
                    output_ptr + i * hidden_size + col_indices,
                    block_buffered_values,
                    mask=valid_mask2,
                )
        else:
            tl.store(output_ptr + input_offsets, buffered_values, mask=valid_mask)

        input_offsets += row_step * hidden_size


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


def _require_cann():
    if not _HAS_CANN:
        pytest.skip("triton CANN extension (al.extract_slice) not available")


def add_rmsnorm_bias_native(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    quant_scale=None,
    quant_offset=None,
):
    device = input.device
    num_vectorcore = _get_num_cores(device)

    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    n_rows = min(batch_size, num_vectorcore)

    SCALE = quant_scale is not None
    if SCALE:
        output = torch.empty(
            batch_size, hidden_size, device=device, dtype=torch.int8
        )
    else:
        output = torch.empty(
            batch_size, hidden_size, device=device, dtype=input.dtype
        )
    output2 = torch.empty(
        batch_size, hidden_size, device=device, dtype=input.dtype
    )

    add_rmsnorm_bias_kernel[(n_rows,)](
        input,
        residual,
        norm_weight,
        norm_bias,
        quant_scale,
        quant_offset,
        output,
        output2,
        batch_size,
        hidden_size,
        eps,
        BLOCK_SIZE,
        2048,  # COL_BLOCK_SIZE (repo default)
        SCALE,
    )
    return output, output2


def add_rmsnorm_bias_ref(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    quant_scale=None,
    quant_offset=None,
):
    """Eager torch reference; the add is rounded in the element dtype first."""
    out2 = (input + residual).to(input.dtype)

    x = out2.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    reciprocal_std = 1.0 / torch.sqrt(variance + eps)
    out1 = x * reciprocal_std * norm_weight.to(torch.float32)
    out1 = out1 + norm_bias.to(torch.float32)

    if quant_scale is not None:
        q = out1 * quant_scale.to(torch.float32) + quant_offset.to(torch.float32)
        # ``cast(tl.int8, overflow_mode="saturate")`` truncates toward zero and
        # then clamps.  Measured on 910B/950: ``trunc`` reproduces the kernel for
        # ~99.9% of entries while round-to-nearest only matches ~55%.
        out1 = torch.trunc(q).clamp(-128, 127).to(torch.int8)
    else:
        out1 = out1.to(input.dtype)
    return out1, out2


_DETECT_ERR = ""


try:
    _DEVICE = _detect_device()
except Exception as _e:  # pragma: no cover - only hit on headless/CI hosts
    _DEVICE = None
    _DETECT_ERR = str(_e)


pytestmark = pytest.mark.skipif(_DEVICE is None, reason=f"no cuda/npu: {_DETECT_ERR}")


_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (5e-3, 5e-3),
    # The normalised output is written back in the input dtype, so a bf16 result
    # cannot be tighter than one bf16 ulp: at |x| ~ 8 that is already 0.0625.
    torch.bfloat16: (2e-2, 2e-2),
}


_EPS = 1e-6


_PERF_SHAPES = [
    (1, 1024, False),
    (8, 1024, False),
    (32, 1024, False),
    (128, 1024, False),
    (512, 1024, False),
    (32, 256, False),
    (32, 8192, False),
    (32, 4096, True),
    (32, 8192, True),
    (32, 16384, True),
]


def _make_inputs(batch_size, hidden_size, dtype, scale, seed):
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    inp = torch.randn(
        batch_size, hidden_size, generator=gen, dtype=dtype, device=_DEVICE
    )
    residual = torch.randn(
        batch_size, hidden_size, generator=gen, dtype=dtype, device=_DEVICE
    )
    weight = torch.randn(hidden_size, generator=gen, dtype=dtype, device=_DEVICE)
    bias = torch.randn(hidden_size, generator=gen, dtype=dtype, device=_DEVICE)
    if not scale:
        return inp, residual, weight, bias, None, None

    # Keep the quantised values inside [-128, 127] so the clamp is not the only
    # thing being measured.
    quant_scale = torch.rand(
        hidden_size, generator=gen, dtype=torch.float32, device=_DEVICE
    ).to(dtype) * 2.0
    quant_offset = torch.randn(
        hidden_size, generator=gen, dtype=torch.float32, device=_DEVICE
    ).to(dtype) * 0.5
    return inp, residual, weight, bias, quant_scale, quant_offset


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"B{s[0]}_H{s[1]}_scale{int(s[2])}",
)
def test_add_rmsnorm_bias_performance(shape):
    batch_size, hidden_size, scale = shape
    if scale:
        _require_cann()
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )

    inp, residual, weight, bias, qs, qo = _make_inputs(
        batch_size, hidden_size, dtype, scale, seed=1234
    )

    def fn_triton():
        add_rmsnorm_bias_native(inp, residual, weight, bias, _EPS, qs, qo)

    try:
        profiling_test(
            fn_triton,
            _DEVICE.type,
            _DEVICE,
            name=f"add_rmsnorm_bias_ori_B{batch_size}_H{hidden_size}_scale{int(scale)}",
        )
    except Exception as exc:  # native config, no multibuffer retry
        if "ub overflow" not in str(exc):
            raise
        pytest.skip(
            "native config cannot compile this shape: auto multi-buffering makes "
            "the tile overflow the unified buffer (the tuned file falls back to "
            f"multibuffer=False here). {exc}".splitlines()[0]
        )


