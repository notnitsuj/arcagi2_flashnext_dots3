"""Nsight Compute probe (if the container has ncu): speed-of-light, memory, occupancy and warp-stall sections for a few
launches of the weight-streaming / attention kernels, driven by the existing microbenchmarks.
  python dots3/ncu_probe.py
Prints the ncu reports (or 'ncu not found')."""
import glob, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
cands = [shutil.which("ncu")] + sorted(glob.glob("/usr/local/cuda*/bin/ncu")) + sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
ncu = next((c for c in cands if c and os.path.exists(c)), None)
print(f"ncu_probe ncu={ncu}", flush=True)
if not ncu:
    sys.exit(0)
SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "WarpStateStats", "LaunchStats"]
JOBS = [("_lowbit_moe_gemm_tp", [os.path.join(HERE, "bench_lowbit_pf.py"), "192"], {"ARC_LOWBIT_PAIRS": "1"}),
        ("^offload_split_kernel", [os.path.join(HERE, "bench_offload_cuda.py"), "48", "4"], {}),
        ("_sparse_mla_offload_split_kernel", [os.path.join(HERE, "bench_offload_cuda.py"), "48", "4"], {}),
        ("_w8a8_triton_block_scaled_mm", [os.path.join(HERE, "bench_fp8_gemm.py"), "192"], {})]
for kern, cmd, env in JOBS:
    if not os.path.exists(cmd[0]):
        continue
    args = [ncu, "--target-processes", "all", "-k", f"regex:{kern}", "-s", "20", "-c", "2"]
    for s in SECTIONS:
        args += ["--section", s]
    r = subprocess.run(args + [sys.executable] + cmd, env=dict(os.environ, **env), capture_output=True, text=True,
                       timeout=1500)
    keep = [l for l in r.stdout.splitlines() if not l.startswith(("oc ", "lbpf ", "abl "))]
    print(f"ncu_probe ==== {kern} rc={r.returncode}", flush=True)
    print("\n".join(keep[-400:]), flush=True)
    if r.returncode:
        print(r.stderr[-1500:], flush=True)
