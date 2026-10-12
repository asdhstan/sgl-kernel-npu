#!/usr/bin/env python3
"""Turn ``result_dir`` profiler output into baseline / tuning reports.

Every performance test writes its trace to ``result_dir/<kernel_name>_<shape>``
(see ``test_common.profiling_test``); inside there,
``ASCEND_PROFILER_OUTPUT/kernel_details.csv`` holds one row per profiled kernel
launch with a ``Duration(us)`` column.  ``avg_time`` is the mean of those
durations over the profiled steps, which is the number every comparison in this
directory is made on.

Usage
-----
    # fresh baseline (overwrites result_baseline/*.md)
    python perf_report.py baseline --src result_dir --out result_baseline

    # record one tuning attempt on top of the existing reports
    python perf_report.py attempt --src result_dir --out result_baseline \
        --label "num_warps=4" --note "BLOCK_N=512, num_stages=1"
"""

import argparse
import csv
import datetime
import os
import statistics
import subprocess

# Longest-prefix matching, so ``split_qkv_rmsnorm_rope`` and
# ``split_qkvgate_gemma_rmsnorm_rope`` never collide.
KERNELS = {
    "mix_fused": "_mix_fused_kernel",
    "causal_conv1d_update": "_causal_conv1d_update_kernel_npu_tiled",
    "store_kv_cache_prefix_valid": "_store_kv_cache_prefix_valid_npu_kernel",
    "add_rmsnorm_bias": "add_rmsnorm_bias_kernel",
    "fused_rope_qk_mqa": "fused_rope_qk_mqa_kernel_opt",
    "split_qkv_rmsnorm_rope": "split_qkv_rmsnorm_rope_kernel",
    "split_qkvgate_gemma_rmsnorm_rope": "split_qkvgate_gemma_rmsnorm_rope_kernel",
    "argmax_prob": "_argmax_prob_kernel",
    "chain_rejection_block_sum": "_chain_rejection_block_sum_kernel",
    "chain_rejection_sample": "_chain_rejection_sample_kernel",
}
# Stable report order: exactly the order of 10shape.md.
KERNEL_ORDER = list(KERNELS)
SYMBOL_TO_KERNEL = {v: k for k, v in KERNELS.items()}

# Report key -> test script that produces the profile.
SCRIPT = {
    "mix_fused": "test_mix_fused_kernel.py",
    "causal_conv1d_update": "causal_conv1d_update_kernel_npu_tiled.py",
    "store_kv_cache_prefix_valid": "store_kv_cache_prefix_valid_npu_kernel.py",
    "add_rmsnorm_bias": "add_rmsnorm_bias_kernel.py",
    "fused_rope_qk_mqa": "fused_rope_qk_mqa_kernel_opt.py",
    "split_qkv_rmsnorm_rope": "split_qkv_rmsnorm_rope_kernel.py",
    "split_qkvgate_gemma_rmsnorm_rope":
        "split_qkvgate_gemma_rmsnorm_rope_kernel.py",
    "argmax_prob": "argmax_prob_kernel.py",
    "chain_rejection_block_sum": "chain_rejection_block_sum_kernel.py",
    "chain_rejection_sample": "chain_rejection_sample_kernel.py",
}


def _split_folder(folder):
    best = None
    for name in KERNELS:
        if folder == name or folder.startswith(name + "_"):
            if best is None or len(name) > len(best):
                best = name
    if best is None:
        return None, None
    return best, folder[len(best) + 1 :]


def _read_stats(folder, symbol):
    path = os.path.join(folder, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    if not os.path.isfile(path):
        return None
    durations = []
    names = set()
    cores = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").strip()
            names.add(name)
            if symbol is not None and name != symbol:
                continue
            raw = (row.get("Duration(us)") or "").strip()
            if not raw:
                continue
            try:
                durations.append(float(raw))
            except ValueError:
                continue
            core = (row.get("Accelerator Core") or "").strip()
            if core:
                cores.add(core)
    if not durations:
        return None
    return {
        "avg": statistics.fmean(durations),
        "min": min(durations),
        "max": max(durations),
        "n": len(durations),
        "cores": sorted(cores),
        "names": sorted(names),
    }


def collect(src):
    """{kernel: {shape: stats}} for every profiled shape under ``src``."""
    data = {k: {} for k in KERNELS}
    if not os.path.isdir(src):
        return data
    for entry in sorted(os.listdir(src)):
        folder = os.path.join(src, entry)
        if not os.path.isdir(folder):
            continue
        kernel, shape = _split_folder(entry)
        if kernel is None:
            continue
        stats = _read_stats(folder, KERNELS[kernel])
        if stats is None:
            # Fall back to every row when the expected symbol is absent.
            stats = _read_stats(folder, None)
        if stats is not None:
            stats["folder"] = entry
            data[kernel][shape] = stats
    return data


def _order_shapes(shapes):
    return sorted(shapes)


def _fmt(value):
    return f"{value:.2f}"


def _git_rev():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _kernel_table(data, kernel, attempts=None):
    lines = [
        f"# `{KERNELS[kernel]}` 性能数据",
        "",
        f"- 测试脚本: `{SCRIPT[kernel]}`",
        f"- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 "
        f"`Duration(us)` 的均值（profiler active 步）",
        "",
        "## Baseline",
        "",
        "| # | shape | avg_time(us) | min(us) | max(us) | n | core |",
        "|---|-------|--------------|---------|---------|---|------|",
    ]
    shapes = _order_shapes(data[kernel])
    for i, shape in enumerate(shapes, 1):
        s = data[kernel][shape]
        lines.append(
            f"| {i} | `{shape}` | {_fmt(s['avg'])} | {_fmt(s['min'])} | "
            f"{_fmt(s['max'])} | {s['n']} | {','.join(s['cores']) or '-'} |"
        )
    if shapes:
        total = sum(data[kernel][s]["avg"] for s in shapes)
        lines += [
            f"| | **sum** | **{_fmt(total)}** | | | | |",
            f"| | **mean** | **{_fmt(total / len(shapes))}** | | | | |",
        ]
    else:
        lines.append("| | *(no data)* | | | | | |")

    if attempts:
        lines += ["", "## 调优记录", ""]
        lines += [
            "| 尝试 | 方案 | avg_time 变化 (sum, us) | 结论 |",
            "|------|------|--------------------------|------|",
        ]
        for a in attempts:
            lines.append(
                f"| {a['label']} | {a['note']} | {a['detail']} | {a['verdict']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _summary(data, attempts=None):
    lines = [
        "# 10shape 基准性能汇总",
        "",
        f"- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git HEAD: `{_git_rev()}`",
        "- 指标: 每个 shape 的 `avg_time(us)`，取自 profiler `kernel_details.csv` 的 "
        "`Duration(us)` 均值",
        "",
        "| kernel | shapes | sum(us) | mean(us) | 明细 |",
        "|--------|--------|---------|----------|------|",
    ]
    for kernel in KERNEL_ORDER:
        shapes = _order_shapes(data[kernel])
        if not shapes:
            lines.append(f"| `{KERNELS[kernel]}` | 0 | - | - | [md]({kernel}.md) |")
            continue
        total = sum(data[kernel][s]["avg"] for s in shapes)
        lines.append(
            f"| `{KERNELS[kernel]}` | {len(shapes)} | {_fmt(total)} | "
            f"{_fmt(total / len(shapes))} | [md]({kernel}.md) |"
        )
    if attempts:
        lines += ["", "## 调优尝试索引", ""]
        lines += ["| 尝试 | 方案 | 结论 |", "|------|------|------|"]
        for a in attempts:
            lines.append(f"| {a['label']} | {a['note']} | {a['verdict']} |")
    lines.append("")
    return "\n".join(lines)


def _load_attempts(out):
    path = os.path.join(out, "_attempts.tsv")
    if not os.path.isfile(path):
        return []
    attempts = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            attempts.append(row)
    return attempts


def _save_attempts(out, attempts):
    path = os.path.join(out, "_attempts.tsv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["label", "note", "detail", "verdict"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(attempts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "attempt"])
    ap.add_argument("--src", default="result_dir")
    ap.add_argument("--out", default="result_baseline")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--verdict", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = collect(args.src)

    if args.mode == "baseline":
        attempts = []
        _save_attempts(args.out, attempts)
    else:
        attempts = _load_attempts(args.out)
        delta_bits = []
        for kernel in KERNEL_ORDER:
            shapes = _order_shapes(data[kernel])
            total = sum(data[kernel][s]["avg"] for s in shapes) if shapes else 0.0
            delta_bits.append(f"{kernel}={total:.1f}us({len(shapes)})")
        attempts.append(
            {
                "label": args.label,
                "note": args.note,
                "detail": "; ".join(delta_bits),
                "verdict": args.verdict,
            }
        )
        _save_attempts(args.out, attempts)

    for kernel in KERNEL_ORDER:
        with open(os.path.join(args.out, f"{kernel}.md"), "w") as fh:
            fh.write(_kernel_table(data, kernel, attempts))
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as fh:
        fh.write(_summary(data, attempts))

    print(f"# {args.mode} -> {args.out}")
    header = f"{'kernel':34s} {'n':>3s} {'sum(us)':>10s} {'mean(us)':>10s}"
    print(header)
    print("-" * len(header))
    for kernel in KERNEL_ORDER:
        shapes = _order_shapes(data[kernel])
        if not shapes:
            print(f"{kernel:34s}   0          -          -")
            continue
        total = sum(data[kernel][s]["avg"] for s in shapes)
        print(
            f"{kernel:34s} {len(shapes):3d} {total:10.2f} "
            f"{total / len(shapes):10.2f}"
        )


if __name__ == "__main__":
    main()
