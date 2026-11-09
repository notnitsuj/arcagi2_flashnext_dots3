"""Tile autotune for arcstats.lowbit_moe on the target GPU (run at notebook startup, before the server).

  python dots3/tune_lowbit.py OUT.json [--ms 64,128,160,192,256,288,320,352,384,1024] [--experts 144] [--n3 96] [--full]
ARC_LOWBIT_PAIRS=1 tunes the pair-layout kernel (what the server will run with the same env). Default grid is the
reduced one (num_warps 4, stages 2-3: every best config on L4 and sm89-emulated H100 so far); --full sweeps all.
Per-rank TP4 shapes of Dots3 lb checkpoints: gate_up N=768 (2 x 1536/4) K=5120; down N=5120 K=384. Random codes of the
checkpoint's bit mix (timing only). Writes {proj: {M: [block_m, block_n, warps, stages]}} for ARC_LOWBIT_TUNE.
"""
import json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":   # dev hosts: L4 code generation on an H100 (dots3/triton_sm89.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_sm89  # noqa: F401
import arcstats.lowbit_moe as lm  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402

OUT = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
Ms = [int(m) for m in arg("--ms", "64,128,160,192,256,288,320,352,384,1024").split(",")]
E = int(arg("--experts", "144")); N3 = int(arg("--n3", "96"))
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
bits = [3 if e < N3 else 2 for e in range(E)]
res = {}
t_all = time.time()
for name, (n_out, K, x_per_tok) in {"gate_up": (768, 5120, True), "down": (5120, 384, False)}.items():
    codes = [torch.randint(0, 2 ** b, (n_out, K), dtype=torch.uint8) for b in bits]
    sc = [torch.rand(n_out, K // 128) * .01 for _ in range(E)]
    zp = [torch.randint(0, 4, (n_out, K // 128), dtype=torch.uint8) for _ in range(E)]
    P = lm.LowbitProj(bits, codes, sc, zp, dev)
    res[name] = {}
    for M in Ms:
        tw, tid = torch.topk(torch.rand(M, E, device=dev), 8, dim=-1)
        x = torch.randn(M if x_per_tok else M * 8, K, device=dev, dtype=torch.bfloat16)
        out = torch.empty(M * 8, n_out, device=dev, dtype=torch.bfloat16)
        best, base = None, None
        full = "--full" in sys.argv
        for bm in ([16, 32] if M <= 512 else ([16, 32, 64] if full else [32, 64])):   # decode up to 128 seqs x 4
            sid, eid, npp = moe_align_block_size(tid.int(), bm, E)
            for bn in (64, 128, 256):
                for nw in ((4, 8) if full else (4,)):
                    for ns in ((2, 3, 4) if full else (2, 3)):
                        f = lambda: P(x, out, tw.reshape(-1).float(), sid, eid, npp, 8, x_per_tok, False, bm, bn, nw, ns)
                        try:
                            f(); torch.cuda.synchronize(); t0 = time.time()
                            for _ in range(10):
                                f()
                            torch.cuda.synchronize(); dt = (time.time() - t0) / 10
                        except Exception:
                            continue
                        if (bm, bn, nw, ns) == (16, 64, 4, 2):
                            base = dt
                        if best is None or dt < best[0]:
                            best = (dt, [bm, bn, nw, ns])
        res[name][str(M)] = best[1]
        print(f"{name} M={M}: best {best[1]} {best[0]*1e3:.3f} ms (default {base*1e3 if base else -1:.3f} ms)", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"tuned in {time.time() - t_all:.0f}s -> {OUT}", flush=True)
