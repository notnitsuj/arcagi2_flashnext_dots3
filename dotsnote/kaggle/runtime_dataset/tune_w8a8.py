"""Tile sweep for vLLM's Triton W8A8 block-FP8 GEMM on the per-rank Dots3 dense shapes at decode batch sizes; writes
vLLM config JSONs (N=..,K=..,device_name=..,dtype=fp8_w8a8,block_shape=[128,128].json) that get_w8a8_block_fp8_configs
picks up.   python dots3/tune_w8a8.py OUTDIR [M ...]   (ARC_TRITON_SM89=1 ARC_EMULATE_L4_OCC=1 for L4-like codegen)"""
import os, sys, time, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("ARC_TRITON_SM89") == "1":
    import triton_sm89  # noqa
import torch
import triton
from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
dev = torch.device("cuda"); torch.manual_seed(0)
OUT = sys.argv[1]
Ms = [int(a) for a in sys.argv[2:]] or [160]
SHAPES = [(2112, 5120), (6144, 1024), (5120, 4096), (768, 5120), (5120, 384), (4096, 1024)]
if os.environ.get("ARC_W8A8_SHAPES"):   # "N:K[:count],..."
    SHAPES = [tuple(int(v) for v in e.split(":"))[:2] for e in os.environ["ARC_W8A8_SHAPES"].split(",")]
DEVNAME = os.environ.get("ARC_W8A8_DEVNAME") or fu.get_device_name_as_file_name() if hasattr(fu, "get_device_name_as_file_name") else "dev"
GRID = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128, GROUP_SIZE_M=g, num_warps=w, num_stages=st)
        for bm, bn, w, st, g in itertools.product([16, 32, 64], [32, 64, 128], [4, 8], [2, 3, 4], [1, 16])
        if not (bm == 64 and bn == 128 and w == 4 and st == 4)]


def timeit(f, n=20, reps=5):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        f()
    torch.cuda.current_stream().wait_stream(st)
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / (n * reps) * 1e6


os.makedirs(OUT, exist_ok=True)
total_best = {}
for N, K in SHAPES:
    w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.rand((N + 127) // 128, (K + 127) // 128, device=dev) * 0.02 + 0.005
    res = {}
    for M in Ms:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        qx, xs = fu.per_token_group_quant_fp8(x, 128)
        ref = None
        best = None
        for cfg in GRID:
            fu.get_w8a8_block_fp8_configs = lambda *a, cfg=cfg: {M: cfg}
            try:
                f = lambda: fu.w8a8_triton_block_scaled_mm(qx, w, xs, s, [128, 128], torch.bfloat16)
                o = f().float()
                if ref is None:
                    ref = o
                if float((o - ref).abs().max() / ref.abs().max()) > 1e-2:
                    continue
                t = timeit(f)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        res[str(M)] = best[1]
        total_best[M] = total_best.get(M, 0) + best[0]
        print(f"N={N} K={K} M={M} best {best[0]:.1f} us {best[1]}", flush=True)
    json.dump(res, open(os.path.join(OUT, f"N={N},K={K},device_name={DEVNAME},dtype=fp8_w8a8,block_shape=[128,128].json"), "w"))
for M, t in total_best.items():
    print(f"M={M} total best {t:.1f} us over {len(SHAPES)} shapes")
