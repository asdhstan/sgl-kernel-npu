"""Shared profiling helper for the 10shape kernel tests.

``profiling_test`` runs ``fn_triton`` under the torch_npu (or torch CUDA)
profiler and stores the trace in ``./result_dir/<name>``.

The profiler itself always appends ``<pid>_<timestamp>_ascend_pt`` to the
folder it creates, which made ``result_dir`` unreadable.  After the profile
finishes the created folder is therefore promoted to ``<name>`` (conventionally
``<kernel_name>_<shape>``), so one kernel/shape maps to exactly one folder.
"""

import glob
import os
import shutil

import pytest
import triton
import torch
import triton.language as tl


_RESULT_ROOT = "./result_dir"


def _safe_name(name):
    """Keep the name filesystem-friendly (shape ids may contain ``/`` etc.)."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def _snapshot():
    return set(glob.glob(os.path.join(_RESULT_ROOT, "*")))


def _promote(created, target):
    """Rename the profiler-created folder to its ``kernel_name_shape`` name."""
    if not target or not created or not os.path.isdir(created):
        return
    if os.path.abspath(created) == os.path.abspath(target):
        return
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors=True)
    os.replace(created, target)
    print(f"性能 profiling 输出目录: {target}")


def profiling_test(fn_triton, device_type, device, name=None):
    """Profile ``fn_triton`` and write the trace to ``./result_dir/<name>``.

    ``name`` should identify the kernel and its shape, e.g.
    ``"causal_conv1d_update_N64_D4096_T8_W4"``.  When it is ``None`` the trace
    keeps the profiler's own (timestamped) folder name.
    """
    os.makedirs(_RESULT_ROOT, exist_ok=True)
    target = os.path.join(_RESULT_ROOT, _safe_name(name)) if name else None
    before = _snapshot()

    if device_type == 'npu':
        import torch_npu

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False
        )

        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=1,
                                                     warmup=1,
                                                     active=10,
                                                     repeat=1,
                                                     skip_first=1),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(_RESULT_ROOT)
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.npu.synchronize()
                prof.step()

        created = sorted(_snapshot() - before)
        if len(created) == 1:
            _promote(created[0], target)
        elif target:
            print(f"警告：预期生成 1 个 profiling 目录，实际 {len(created)} 个，跳过重命名")

    elif device_type == 'cuda':
        cuda_dir = target if target else _RESULT_ROOT
        os.makedirs(cuda_dir, exist_ok=True)

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=10,
                                                 repeat=1,
                                                 skip_first=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(cuda_dir)
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.cuda.synchronize()
                prof.step()

        print("\n========== GPU 性能统计 ==========")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        if target:
            print(f"性能 profiling 输出目录: {target}")

    else:
        print("警告：未检测到 GPU 或 NPU，跳过性能 profiling")

    print("性能 profiling 完成！")
