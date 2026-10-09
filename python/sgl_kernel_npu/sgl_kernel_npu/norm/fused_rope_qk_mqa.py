import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


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
    d_all = tl.arange(0, D_ROPE)

    # cos / sin (shared across all heads for this position)
    cos = tl.load(cos_sin_ptr + pid_t * stride_ct + d * stride_cd)
    sin = tl.load(cos_sin_ptr + pid_t * stride_ct + (d + D_ROPE // 2) * stride_cd)

    head_offs = tl.arange(0, Hq)
    kh_offs = tl.arange(0, Hk)
    q_base = query_ptr + pid_t * stride_qt
    k_base = key_ptr + pid_t * stride_kt
    oq_base = out_q_ptr + pid_t * stride_oqt
    ok_base = out_k_ptr + pid_t * stride_okt

    # One load per q/k covers the whole rotary span, so each head row is a full
    # contiguous burst instead of two half-width halves, and only two per-head
    # address tensors are materialised (was eight).  The even/odd split and the
    # reassembly happen in registers.
    q_all = tl.load(q_base + head_offs[:, None] * stride_qh + d_all[None, :] * stride_qd)
    k_all = tl.load(k_base + kh_offs[:, None] * stride_kh + d_all[None, :] * stride_kd)

    if IS_NEOX_STYLE:
        q1 = al.extract_slice(q_all, (0, 0), (Hq, D_ROPE // 2), (1, 1))
        q2 = al.extract_slice(q_all, (0, D_ROPE // 2), (Hq, D_ROPE // 2), (1, 1))
        k1 = al.extract_slice(k_all, (0, 0), (Hk, D_ROPE // 2), (1, 1))
        k2 = al.extract_slice(k_all, (0, D_ROPE // 2), (Hk, D_ROPE // 2), (1, 1))
    else:
        q1 = al.extract_slice(q_all, (0, 0), (Hq, D_ROPE // 2), (1, 2))
        q2 = al.extract_slice(q_all, (0, 1), (Hq, D_ROPE // 2), (1, 2))
        k1 = al.extract_slice(k_all, (0, 0), (Hk, D_ROPE // 2), (1, 2))
        k2 = al.extract_slice(k_all, (0, 1), (Hk, D_ROPE // 2), (1, 2))

    q_out1 = (q1 * cos[None, :]) - (q2 * sin[None, :])
    q_out2 = (q1 * sin[None, :]) + (q2 * cos[None, :])
    k_out1 = (k1 * cos[None, :]) - (k2 * sin[None, :])
    k_out2 = (k1 * sin[None, :]) + (k2 * cos[None, :])

    out_ty = query_ptr.dtype.element_ty
    q_out = tl.zeros((Hq, D_ROPE), dtype=out_ty)
    k_out = tl.zeros((Hk, D_ROPE), dtype=out_ty)
    if IS_NEOX_STYLE:
        q_out = al.insert_slice(q_out, q_out1, (0, 0), (Hq, D_ROPE // 2), (1, 1))
        q_out = al.insert_slice(
            q_out, q_out2, (0, D_ROPE // 2), (Hq, D_ROPE // 2), (1, 1)
        )
        k_out = al.insert_slice(k_out, k_out1, (0, 0), (Hk, D_ROPE // 2), (1, 1))
        k_out = al.insert_slice(
            k_out, k_out2, (0, D_ROPE // 2), (Hk, D_ROPE // 2), (1, 1)
        )
    else:
        q_out = al.insert_slice(q_out, q_out1, (0, 0), (Hq, D_ROPE // 2), (1, 2))
        q_out = al.insert_slice(q_out, q_out2, (0, 1), (Hq, D_ROPE // 2), (1, 2))
        k_out = al.insert_slice(k_out, k_out1, (0, 0), (Hk, D_ROPE // 2), (1, 2))
        k_out = al.insert_slice(k_out, k_out2, (0, 1), (Hk, D_ROPE // 2), (1, 2))

    tl.store(
        oq_base + head_offs[:, None] * stride_oqh + d_all[None, :] * stride_oqd, q_out
    )
    tl.store(
        ok_base + kh_offs[:, None] * stride_okh + d_all[None, :] * stride_okd, k_out
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


def fused_rope_qk_mqa(query, key, cos_sin, rotary_dim, is_neox_style):
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
