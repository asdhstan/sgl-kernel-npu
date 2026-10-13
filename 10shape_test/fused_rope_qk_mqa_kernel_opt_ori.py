"""the native ``fused_rope_qk_mqa`` (nothing is tunable here, so this is
the same launch as the tuned file)"""

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
def fused_rope_qk_mqa_kernel_opt(
    query_ptr,  # [T, Hq, D]
    key_ptr,  # [T, Hk, D]
    cos_sin_ptr,  # [max_pos, D_ROPE]
    out_q_ptr,
    out_k_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ct,
    stride_cd,
    stride_oqt,
    stride_oqh,
    stride_oqd,
    stride_okt,
    stride_okh,
    stride_okd,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    pid_t = tl.program_id(0)

    # -------- rotary indices
    d = tl.arange(0, D_ROPE // 2)
    if IS_NEOX_STYLE:
        idx_even = d
        idx_odd = d + D_ROPE // 2
    else:
        idx_even = d * 2
        idx_odd = d * 2 + 1

    # cos / sin (shared across all heads for this position)
    cos = tl.load(cos_sin_ptr + pid_t * stride_ct + d * stride_cd)
    sin = tl.load(cos_sin_ptr + pid_t * stride_ct + (d + D_ROPE // 2) * stride_cd)

    # ================= Q (all Hq heads at once) =================
    head_offs = tl.arange(0, Hq)
    q_base = query_ptr + pid_t * stride_qt

    q1 = tl.load(
        q_base + head_offs[:, None] * stride_qh + idx_even[None, :] * stride_qd
    )
    q2 = tl.load(q_base + head_offs[:, None] * stride_qh + idx_odd[None, :] * stride_qd)

    q_out1 = (q1 * cos[None, :]) - (q2 * sin[None, :])
    q_out2 = (q1 * sin[None, :]) + (q2 * cos[None, :])

    oq_base = out_q_ptr + pid_t * stride_oqt
    tl.store(
        oq_base + head_offs[:, None] * stride_oqh + idx_even[None, :] * stride_oqd,
        q_out1,
    )
    tl.store(
        oq_base + head_offs[:, None] * stride_oqh + idx_odd[None, :] * stride_oqd,
        q_out2,
    )

    # ================= K (unique Hk key heads only) =================
    kh_offs = tl.arange(0, Hk)
    k_base = key_ptr + pid_t * stride_kt

    k1 = tl.load(k_base + kh_offs[:, None] * stride_kh + idx_even[None, :] * stride_kd)
    k2 = tl.load(k_base + kh_offs[:, None] * stride_kh + idx_odd[None, :] * stride_kd)

    k_out1 = (k1 * cos[None, :]) - (k2 * sin[None, :])
    k_out2 = (k1 * sin[None, :]) + (k2 * cos[None, :])

    ok_base = out_k_ptr + pid_t * stride_okt
    tl.store(
        ok_base + kh_offs[:, None] * stride_okh + idx_even[None, :] * stride_okd, k_out1
    )
    tl.store(
        ok_base + kh_offs[:, None] * stride_okh + idx_odd[None, :] * stride_okd, k_out2
    )

    # ================= pass-through (compile-time pruning) =================
    if D_HEAD > D_ROPE:
        dp = tl.arange(0, D_HEAD - D_ROPE)
        # Q pass-through
        q_pass = tl.load(
            q_base + head_offs[:, None] * stride_qh + (dp + D_ROPE)[None, :] * stride_qd
        )
        tl.store(
            oq_base
            + head_offs[:, None] * stride_oqh
            + (dp + D_ROPE)[None, :] * stride_oqd,
            q_pass,
        )
        # K pass-through
        k_pass = tl.load(
            k_base + kh_offs[:, None] * stride_kh + (dp + D_ROPE)[None, :] * stride_kd
        )
        tl.store(
            ok_base
            + kh_offs[:, None] * stride_okh
            + (dp + D_ROPE)[None, :] * stride_okd,
            k_pass,
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


def _dtype_supported(device, dtype):
    try:
        torch.zeros(1, dtype=dtype, device=device)
        return True
    except Exception:
        return False


def fused_rope_qk_mqa_native(query, key, cos_sin, rotary_dim, is_neox_style):
    T, Hq, D = query.shape
    _, Hk, _ = key.shape

    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)

    grid = (T,)

    fused_rope_qk_mqa_kernel_opt[grid](
        query,
        key,
        cos_sin,
        out_q,
        out_k,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        cos_sin.stride(0),
        cos_sin.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        Hq=Hq,
        Hk=Hk,
        D_HEAD=D,
        D_ROPE=rotary_dim,
        IS_NEOX_STYLE=is_neox_style,
    )

    return out_q, out_k


def _apply_rotary_emb(x, cos, sin, is_neox_style):
    """``x``: [T, H, D_ROPE] fp32; ``cos``/``sin``: [T, D_ROPE // 2] fp32."""
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


_DETECT_ERR = ""


try:
    _DEVICE = _detect_device()
except Exception as _e:  # pragma: no cover - only hit on headless/CI hosts
    _DEVICE = None
    _DETECT_ERR = str(_e)


pytestmark = pytest.mark.skipif(_DEVICE is None, reason=f"no cuda/npu: {_DETECT_ERR}")


_PERF_SHAPES = [
    (1, 8, 1, 128, 128, True),
    (8, 8, 1, 128, 128, True),
    (32, 8, 1, 128, 128, True),
    (128, 8, 1, 128, 128, True),
    (1024, 8, 1, 128, 128, True),
    (32, 32, 8, 128, 128, True),
    (32, 64, 1, 128, 128, True),
    (32, 8, 1, 256, 128, True),
    (32, 8, 1, 128, 64, True),
    (32, 8, 1, 128, 128, False),
]


def _make_case(shape, dtype, seed):
    T, Hq, Hk, D_HEAD, D_ROPE, is_neox = shape
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    query = torch.randn(T, Hq, D_HEAD, generator=gen, dtype=dtype, device=_DEVICE)
    key = torch.randn(T, Hk, D_HEAD, generator=gen, dtype=dtype, device=_DEVICE)
    cos = torch.randn(
        T, D_ROPE // 2, generator=gen, dtype=dtype, device=_DEVICE
    )
    sin = torch.randn(
        T, D_ROPE // 2, generator=gen, dtype=dtype, device=_DEVICE
    )
    cos_sin = torch.cat([cos, sin], dim=-1)
    return query, key, cos_sin


@pytest.mark.parametrize(
    "shape",
    _PERF_SHAPES,
    ids=lambda s: f"T{s[0]}_Hq{s[1]}_Hk{s[2]}_D{s[3]}_R{s[4]}_neox{int(s[5])}",
)
def test_fused_rope_qk_mqa_performance(shape):
    T, Hq, Hk, D_HEAD, D_ROPE, is_neox = shape
    dtype = (
        torch.float16 if _dtype_supported(_DEVICE, torch.float16) else torch.float32
    )
    query, key, cos_sin = _make_case(shape, dtype, seed=1234)

    def fn_triton():
        fused_rope_qk_mqa_native(query, key, cos_sin, D_ROPE, is_neox)

    profiling_test(
        fn_triton,
        _DEVICE.type,
        _DEVICE,
        name=(
            f"fused_rope_qk_mqa_ori_T{T}_Hq{Hq}_Hk{Hk}"
            f"_D{D_HEAD}_R{D_ROPE}_neox{int(is_neox)}"
        ),
    )


