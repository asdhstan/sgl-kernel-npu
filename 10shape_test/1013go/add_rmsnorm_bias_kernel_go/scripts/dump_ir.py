"""Run add_rmsnorm_bias once per shape so Triton emits its IR into
TRITON_CACHE_DIR (for the full-IR dump)."""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "python", "sgl_kernel_npu"))

from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

def detect():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("no npu/cuda")

dev = detect()
for batch, hidden, scale in [(32, 16384, True), (512, 1024, False)]:
    dtype = torch.float16
    gen = torch.Generator(device=dev).manual_seed(7)
    x = torch.randn(batch, hidden, generator=gen, dtype=dtype, device=dev)
    r = torch.randn(batch, hidden, generator=gen, dtype=dtype, device=dev)
    w = torch.randn(hidden, generator=gen, dtype=dtype, device=dev)
    b = torch.randn(hidden, generator=gen, dtype=dtype, device=dev)
    if scale:
        qs = torch.rand(hidden, generator=gen, dtype=torch.float32, device=dev) + 0.5
        qo = torch.zeros(hidden, dtype=torch.float32, device=dev)
        add_rmsnorm_bias(x, r, w, b, 1e-6, qs, qo)
    else:
        add_rmsnorm_bias(x, r, w, b, 1e-6)
    torch.npu.synchronize()
    print(f"done B{batch}_H{hidden}_scale{int(scale)}")
