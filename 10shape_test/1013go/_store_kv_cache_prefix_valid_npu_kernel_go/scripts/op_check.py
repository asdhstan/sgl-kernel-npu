"""Validate the OPERATOR itself (not the harness copy): exact correctness + perf.

Usage: python op_check.py <tag>
Writes profiler output as result_dir/skv_op_<tag>_<shape>.
"""
import csv
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import store_kv_cache_prefix_valid_npu_kernel as H  # noqa: E402  (10shape harness)
from test_common import profiling_test  # noqa: E402
from sgl_kernel_npu.mem_cache.kv_cache_store import (  # noqa: E402
    store_kv_cache_prefix_valid_npu_triton as op,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
DTYPE = torch.float16
DEV = H._DEVICE

bad = []
for shape in H._PERF_SHAPES:
    # reference inputs
    r = H._make_case(shape, DTYPE, seed=20240801 + sum(shape))
    exp_k, exp_v = H.store_kv_cache_prefix_valid_ref(*r)
    # fresh inputs for the operator
    a = H._make_case(shape, DTYPE, seed=20240801 + sum(shape))
    op(*a)
    torch.npu.synchronize()
    if not (torch.equal(a[0], exp_k) and torch.equal(a[1], exp_v)):
        bad.append((shape, int((a[0] != exp_k).sum().item()), int((a[1] != exp_v).sum().item())))
    # profile
    def fn():
        op(*a)

    b, w, h, d, c = shape
    profiling_test(fn, DEV.type, DEV, name=f"skv_op_{TAG}_B{b}_W{w}_H{h}_D{d}_C{c}")

print(f"[op_check:{TAG}] correctness: {'ALL OK' if not bad else 'MISMATCH ' + str(bad)}")
tot = n = 0
for e in os.listdir(os.path.join(ROOT, "result_dir")):
    if not e.startswith(f"skv_op_{TAG}_"):
        continue
    p = os.path.join(ROOT, "result_dir", e, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    if not os.path.isfile(p):
        continue
    v = [float(r["Duration(us)"]) for r in csv.DictReader(open(p, newline="")) if (r.get("Duration(us)") or "").strip()]
    if v:
        tot += statistics.fmean(v)
        n += 1
print(f"[op_check:{TAG}] perf: shapes={n} sum={tot:.2f} us")
