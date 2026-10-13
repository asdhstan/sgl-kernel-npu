"""Run the native _argmax_prob_kernel once so Triton emits its IR into
TRITON_CACHE_DIR, for the shape we care about."""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import argmax_prob_kernel_ori as M

B, V = 1024, 128000
gen = torch.Generator(device=M._DEVICE).manual_seed(1)
logits = torch.randn(B, V, generator=gen, dtype=torch.float16, device=M._DEVICE)
M.argmax_prob_native(logits)
torch.npu.synchronize()
print("done", logits.shape, logits.dtype)
