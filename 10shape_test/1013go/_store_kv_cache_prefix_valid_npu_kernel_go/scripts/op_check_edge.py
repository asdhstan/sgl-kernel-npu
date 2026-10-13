"""Correctness edge cases for the OPERATOR: dtype variants x zero-commit."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import store_kv_cache_prefix_valid_npu_kernel as H  # noqa: E402
from sgl_kernel_npu.mem_cache.kv_cache_store import (  # noqa: E402
    store_kv_cache_prefix_valid_npu_triton as op,
)

DEV = H._DEVICE
bad = []
for dtype in (torch.float16, torch.float32):
    if not H._dtype_supported(DEV, dtype):
        print(f"  dtype {dtype} not supported -> skip")
        continue
    for zero in (False, True):
        for shape in H._PERF_SHAPES[:6]:
            r = H._make_case(shape, dtype, seed=777)
            if zero:
                r = list(r)
                r[5] = torch.zeros_like(r[5])   # commit_lens = 0
                r = tuple(r)
            exp_k, exp_v = H.store_kv_cache_prefix_valid_ref(*r)
            a = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in r)
            op(*a)
            torch.npu.synchronize()
            if not (torch.equal(a[0], exp_k) and torch.equal(a[1], exp_v)):
                bad.append((str(dtype), zero, shape,
                            int((a[0] != exp_k).sum().item()),
                            int((a[1] != exp_v).sum().item())))
print(f"[op_check_edge] {'ALL OK' if not bad else 'MISMATCH ' + str(bad[:4])}")
