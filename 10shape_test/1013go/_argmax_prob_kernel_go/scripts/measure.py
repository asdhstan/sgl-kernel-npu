"""Run the argmax perf test N times with a fresh Triton cache and report the
per-shape minimum avg_time, compared against the stored baseline."""
import csv, json, os, statistics, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # _argmax_prob_kernel_go
ROOT = os.path.dirname(os.path.dirname(HERE))                        # 10shape_test
CACHE = os.path.join(HERE, "triton_cache")
result_dir = os.path.join(ROOT, "result_dir")

def sums():
    out = {}
    for e in os.listdir(result_dir):
        if not e.startswith("argmax_prob_") or e.startswith("argmax_prob_ori_"): continue
        p = os.path.join(result_dir, e, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        if not os.path.isfile(p): continue
        v = [float(r["Duration(us)"]) for r in csv.DictReader(open(p, newline=""))
             if (r.get("Name") or "").strip() == "_argmax_prob_kernel" and (r.get("Duration(us)") or "").strip()]
        if v: out[e] = statistics.fmean(v)
    return out

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
env = dict(os.environ, TRITON_CACHE_DIR=CACHE)
acc = {}
for i in range(runs):
    subprocess.run([sys.executable, "-m", "pytest", "argmax_prob_kernel.py",
                    "-k", "performance", "-q", "-p", "no:cacheprovider"],
                   cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k, v in sums().items():
        acc.setdefault(k, []).append(v)
    print(f"  run {i+1}: sum={sum(v[-1] for v in acc.values()):.2f}")

base = json.load(open(os.path.join(ROOT, "result_baseline", "_baseline.json")))
base.pop("_runs", None)
print(f"\n{'shape':22s} {'baseline':>9s} {'patched':>9s} {'delta':>9s}")
b_tot = p_tot = 0.0
for k in sorted(acc, key=lambda x: -min(acc[x])):
    shape = k[len("argmax_prob_"):]
    b = base["argmax_prob"].get(shape, {}).get("avg")
    p = min(acc[k])
    b_tot += b or 0; p_tot += p
    print(f"{shape:22s} {b:9.2f} {p:9.2f} {(p/b-1)*100:+8.1f}%")
print(f"{'SUM':22s} {b_tot:9.2f} {p_tot:9.2f} {(p_tot/b_tot-1)*100:+8.1f}%")
json.dump({k: min(v) for k, v in acc.items()}, open(os.path.join(HERE, "patched_measure.json"), "w"), indent=1)
