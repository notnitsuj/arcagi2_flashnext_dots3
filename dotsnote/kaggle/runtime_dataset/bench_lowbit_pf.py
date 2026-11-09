"""Lowbit MoE pair kernel: K-group prefetch (ARC_LOWBIT_PF) vs baseline on the target GPU, per-rank TP4 Dots3 shapes,
checkpoint bit mix (96 x 3-bit + 48 x 2-bit experts), random codes. Checks PF=1 == PF=0 bitwise; CUDA-graph timing.
  ARC_LOWBIT_PAIRS=1 python dots3/bench_lowbit_pf.py [M=160,192]
Prints 'lbpf <proj> M=<m> cfg <bm,bn,w,st> base <ms> pf <ms> GB/s <base> <pf> exact <bool>'."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
import torch
import arcstats.lowbit_moe as lm
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
Ms = [int(m) for m in (sys.argv[1] if len(sys.argv) > 1 else "160,192").split(",")]
E, N3 = 144, 96
bits = [3 if e < N3 else 2 for e in range(E)]


def timeit(f, n=20, reps=5):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / (n * reps) * 1e3


for name, (n_out, K, x_per_tok) in {"gate_up": (768, 5120, True), "down": (5120, 384, False)}.items():
    codes = [torch.randint(0, 2 ** b, (n_out, K), dtype=torch.uint8) for b in bits]
    sc = [torch.rand(n_out, K // 128) * .01 for _ in range(E)]
    zp = [torch.randint(0, 4, (n_out, K // 128), dtype=torch.uint8) for _ in range(E)]
    P = lm.LowbitProj(bits, codes, sc, zp, dev)
    wbytes = P.nbytes()
    for M in Ms:
        tw, tid = torch.topk(torch.rand(M, E, device=dev), 8, dim=-1)
        x = torch.randn(M if x_per_tok else M * 8, K, device=dev, dtype=torch.bfloat16).half()
        for cfg in ([16, 64, 4, 2], [16, 64, 4, 3], [16, 128, 4, 2], [16, 128, 4, 3], [16, 128, 8, 2], [32, 128, 4, 2]):
            sid, eid, npp = moe_align_block_size(tid.int(), cfg[0], E)
            outs = {}
            ts = {}
            for pf in (0, 1):
                lm._PF = pf
                out = torch.empty(M * 8, n_out, device=dev, dtype=torch.bfloat16)
                f = lambda: P(x, out, tw.reshape(-1).float(), sid, eid, npp, 8, x_per_tok, False, *cfg)
                try:
                    ts[pf] = timeit(f)
                    f(); torch.cuda.synchronize(); outs[pf] = out.clone()
                except Exception as ex:
                    ts[pf] = float("nan"); print(f"lbpf {name} M={M} cfg {cfg} pf={pf} FAILED {repr(ex)[:160]}", flush=True)
            ex = (0 in outs and 1 in outs and torch.equal(outs[0], outs[1]))
            gbs = {k: wbytes / (v * 1e-3) / 1e9 for k, v in ts.items()}
            print(f"lbpf {name} M={M} cfg {cfg} base {ts[0]:.3f} pf {ts[1]:.3f} GB/s {gbs[0]:.0f} {gbs[1]:.0f} "
                  f"exact {ex}", flush=True)
lm._PF = 0
