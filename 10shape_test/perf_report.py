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
    # fresh baseline (overwrites result_baseline/*.md and _baseline.json)
    python perf_report.py baseline --src result_dir --out result_baseline

    # record one tuning attempt for one or more kernels
    python perf_report.py attempt --src result_dir --out result_baseline \
        --label "argmax autotune" --note "ARGMAX_TUNE=autotune" \
        --verdict "sum -27%" --only argmax_prob
"""

import argparse
import csv
import datetime
import json
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

BASELINE_JSON = "_baseline.json"
ATTEMPTS_JSON = "_attempts.json"


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
    cores = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").strip()
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
        "median": statistics.median(durations),
        "n": len(durations),
        "cores": sorted(cores),
    }


def _read_folder(folder, symbol):
    stats = _read_stats(folder, symbol)
    if stats is None:
        stats = _read_stats(folder, None)
    return stats


def collect(src, only=None):
    """{kernel: {shape: stats}} for every profiled shape under ``src``."""
    data = {k: {} for k in KERNELS}
    if not os.path.isdir(src):
        return data
    for entry in sorted(os.listdir(src)):
        folder = os.path.join(src, entry)
        if not os.path.isdir(folder):
            continue
        kernel, shape = _split_folder(entry)
        if kernel is None or (only and kernel not in only):
            continue
        stats = _read_folder(folder, KERNELS[kernel])
        if stats is not None:
            stats["folder"] = entry
            data[kernel][shape] = stats
    return data


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


def _kernel_md(kernel, baseline, attempts, runs=1):
    lines = [
        f"# `{KERNELS[kernel]}` 性能数据",
        "",
        f"- 测试脚本: `{SCRIPT[kernel]}`",
        "- 指标: `avg_time(us)` = `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中 "
        "`Duration(us)` 的均值（profiler active 步）",
        f"- 测量协议: baseline 取 best-of-{runs} 次运行的 `avg_time`"
        "（共享 NPU 上单次运行会被其他进程干扰，取最小值抑制）",
        "- 环境噪声提示: NPU 为共享卡，`min/max` 相差数倍说明该次测量受到干扰；"
        "跨次比较时同时看 `avg_time` 与 `min`",
        "",
        "## Baseline",
        "",
        "| # | shape | avg_time(us) | min(us) | max(us) | n | core |",
        "|---|-------|--------------|---------|---------|---|------|",
    ]
    shapes = sorted(baseline.get(kernel, {}))
    for i, shape in enumerate(shapes, 1):
        s = baseline[kernel][shape]
        lines.append(
            f"| {i} | `{shape}` | {_fmt(s['avg'])} | {_fmt(s['min'])} | "
            f"{_fmt(s['max'])} | {s['n']} | {','.join(s['cores']) or '-'} |"
        )
    if shapes:
        total = sum(baseline[kernel][s]["avg"] for s in shapes)
        lines += [
            f"| | **sum** | **{_fmt(total)}** | | | | |",
            f"| | **mean** | **{_fmt(total / len(shapes))}** | | | | |",
        ]
    else:
        lines.append("| | *(no data)* | | | | | |")

    kernel_attempts = [a for a in attempts if kernel in a["data"]]
    if kernel_attempts:
        lines += ["", "## 调优记录", ""]
        for a in kernel_attempts:
            lines += [
                f"### {a['label']}",
                "",
                f"- 时间: {a['time']}",
                f"- 方案: {a['note']}",
                f"- 结论: {a['verdict']}",
                "",
                "| # | shape | base avg(us) | 本次 avg(us) | Δ(us) | Δ% |",
                "|---|-------|--------------|--------------|-------|-----|",
            ]
            measured = a["data"][kernel]
            base_total = sum(
                baseline.get(kernel, {}).get(s, {}).get("avg", 0.0)
                for s in sorted(measured)
            )
            new_total = sum(measured[s]["avg"] for s in sorted(measured))
            for shape in sorted(measured):
                b = baseline.get(kernel, {}).get(shape, {}).get("avg")
                n = measured[shape]["avg"]
                if b is None:
                    lines.append(f"| | `{shape}` | - | {_fmt(n)} | - | - |")
                else:
                    d = n - b
                    pct = (d / b * 100.0) if b else 0.0
                    lines.append(
                        f"| | `{shape}` | {_fmt(b)} | {_fmt(n)} | "
                        f"{d:+.2f} | {pct:+.1f}% |"
                    )
            if base_total:
                dpct = (new_total - base_total) / base_total * 100.0
                lines.append(
                    f"| | **sum** | **{_fmt(base_total)}** | **{_fmt(new_total)}** "
                    f"| **{new_total - base_total:+.2f}** | **{dpct:+.1f}%** |"
                )
            lines += ["", f"- 本次 min/median: "
                      + ", ".join(
                          f"`{s}`={_fmt(measured[s]['min'])}/"
                          f"{_fmt(measured[s]['median'])}"
                          for s in sorted(measured)
                      ), ""]
    lines.append("")
    return "\n".join(lines)


def _summary_md(baseline, attempts, runs=1):
    lines = [
        "# 10shape 基准性能汇总",
        "",
        f"- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git HEAD: `{_git_rev()}`",
        "- 指标: 每个 shape 的 `avg_time(us)`，取自 profiler `kernel_details.csv` "
        "的 `Duration(us)` 均值",
        f"- 测量协议: baseline 为 best-of-{runs} 次运行的最小 `avg_time`",
        "",
        "## Baseline",
        "",
        "| kernel | shapes | sum(us) | mean(us) | 明细 |",
        "|--------|--------|---------|----------|------|",
    ]
    for kernel in KERNEL_ORDER:
        shapes = sorted(baseline.get(kernel, {}))
        if not shapes:
            lines.append(f"| `{KERNELS[kernel]}` | 0 | - | - | [{kernel}.md]({kernel}.md) |")
            continue
        total = sum(baseline[kernel][s]["avg"] for s in shapes)
        lines.append(
            f"| `{KERNELS[kernel]}` | {len(shapes)} | {_fmt(total)} | "
            f"{_fmt(total / len(shapes))} | [{kernel}.md]({kernel}.md) |"
        )
    grand = sum(
        sum(baseline[k][s]["avg"] for s in baseline.get(k, {})) for k in KERNEL_ORDER
    )
    lines.append(f"| **合计** | | **{_fmt(grand)}** | | |")

    if attempts:
        lines += [
            "",
            "## 调优尝试索引",
            "",
            "| 尝试 | 时间 | 算子 | 方案 | 结论 |",
            "|------|------|------|------|------|",
        ]
        for a in attempts:
            ks = ", ".join(f"`{k}`" for k in a["data"])
            lines.append(
                f"| {a['label']} | {a['time']} | {ks} | {a['note']} | {a['verdict']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_reports(out, baseline, attempts, runs=1):
    for kernel in KERNEL_ORDER:
        with open(os.path.join(out, f"{kernel}.md"), "w") as fh:
            fh.write(_kernel_md(kernel, baseline, attempts, runs))
    with open(os.path.join(out, "SUMMARY.md"), "w") as fh:
        fh.write(_summary_md(baseline, attempts, runs))


def _print_table(baseline, attempts, only=None):
    header = f"{'kernel':34s} {'n':>3s} {'sum(us)':>10s} {'mean(us)':>10s}"
    print(header)
    print("-" * len(header))
    for kernel in KERNEL_ORDER:
        if only and kernel not in only:
            continue
        shapes = sorted(baseline.get(kernel, {}))
        if not shapes:
            print(f"{kernel:34s}   0          -          -")
            continue
        total = sum(baseline[kernel][s]["avg"] for s in shapes)
        print(f"{kernel:34s} {len(shapes):3d} {total:10.2f} {total / len(shapes):10.2f}")
    for a in attempts[-1:]:
        print(f"\n-- {a['label']} ({a['verdict']})")
        for kernel, shapes in a["data"].items():
            total = sum(shapes[s]["avg"] for s in shapes)
            print(f"{kernel:34s} {len(shapes):3d} {total:10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "attempt", "show"])
    ap.add_argument("--src", default="result_dir")
    ap.add_argument("--out", default="result_baseline")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--merge-min",
        action="store_true",
        help="baseline mode: keep the smaller avg_time per shape across runs",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    only = set(x.strip() for x in args.only.split(",") if x.strip()) or None

    base_path = os.path.join(args.out, BASELINE_JSON)
    att_path = os.path.join(args.out, ATTEMPTS_JSON)

    if args.mode == "baseline":
        baseline = collect(args.src)
        runs = 1
        if args.merge_min and os.path.isfile(base_path):
            with open(base_path) as fh:
                old = json.load(fh)
            runs = old.get("_runs", 1) + 1
            old.pop("_runs", None)
            for kernel, shapes in old.items():
                for shape, stats in shapes.items():
                    cur = baseline.get(kernel, {}).get(shape)
                    # Best-of-N: a contended run only ever inflates avg_time.
                    if cur is None or stats["avg"] < cur["avg"]:
                        baseline.setdefault(kernel, {})[shape] = stats
        baseline["_runs"] = runs
        attempts = []
        with open(base_path, "w") as fh:
            json.dump(baseline, fh, indent=1, sort_keys=True)
        with open(att_path, "w") as fh:
            json.dump(attempts, fh, indent=1)
        _write_reports(args.out, baseline, attempts, runs)
        print(f"# baseline -> {args.out} (runs merged: {runs})")
        _print_table(baseline, attempts)
        return

    with open(base_path) as fh:
        baseline = json.load(fh)
    runs = baseline.pop("_runs", 1)
    attempts = []
    if os.path.isfile(att_path):
        with open(att_path) as fh:
            attempts = json.load(fh)

    if args.mode == "attempt":
        data = {k: v for k, v in collect(args.src, only).items() if v}
        attempts.append(
            {
                "label": args.label or f"attempt {len(attempts) + 1}",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": args.note,
                "verdict": args.verdict,
                "data": data,
            }
        )
        with open(att_path, "w") as fh:
            json.dump(attempts, fh, indent=1)
        _write_reports(args.out, baseline, attempts, runs)
        print(f"# attempt -> {args.out}")
        _print_table(baseline, attempts, only)
    else:
        _print_table(baseline, attempts, only)


if __name__ == "__main__":
    main()
