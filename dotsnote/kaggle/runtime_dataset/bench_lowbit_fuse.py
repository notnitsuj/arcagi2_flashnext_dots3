"""Fused MoE small kernels (arcstats.lowbit_fuse) vs the current torch / vLLM ops on the target GPU: per-rank TP4 Dots3
shapes, real expert weights when found (BENCH_REAL=1 / --real=FILE), skewed synthetic router logits, CUDA graphs,
interleaved rounds with medians (L4 clocks drift).
  ARC_LOWBIT_PAIRS=1 python dots3/bench_lowbit_fuse.py [Ms=256] [--real=FILE]
Prints per-part times (routing, x prep, silu prep, combine; old vs fused), the full MoE layer (routing included) old vs
fused, and the accuracy of the fused layer vs the current one (rel. L2 / max error, top-k id agreement). Last lines:
  'FUSE_GATE M=<m> aux_old <ms> aux_new <ms> layer_old <ms> layer_new <ms> rel <err> ids <agree> ok <bool>'"""
import glob, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_sm89  # noqa: F401
import torch
import arcstats.lowbit_moe as lm
import arcstats.lowbit_fuse as lf
from arcstats.lowbit2 import combine
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

T0 = time.time()
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
Ms = [int(m) for m in (args[0] if args else "256").split(",")]
E, N3, TOPK, SCALE = 144, 96, 8, 2.5
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "9"))
assert lm.PAIRS, "set ARC_LOWBIT_PAIRS=1"


def log(*a):
    print("lbf", *a, flush=True)


def build():
    real = next((a for a in sys.argv if a.startswith("--real=")), None)
    path = real.split("=", 1)[1] if real else None
    if path is None and os.environ.get("BENCH_REAL") == "1":
        c = sorted(glob.glob(os.environ.get("ARC_INPUT", "/kaggle/input") + "/**/lowbit-L10.pt", recursive=True))
        path = c[0] if c else None
    if path:
        d = torch.load(path, map_location="cpu", weights_only=False)
        ex = [d[e] for e in d["experts"]]
        Is = 1536 // 4
        gu_rows = torch.cat([torch.arange(0, Is), 1536 + torch.arange(0, Is)])
        H = ex[0]["d"]["p2" if "p2" in ex[0]["d"] else "p4"].shape[0]
        gpr = ex[0]["d_s"].shape[1] // 4
        gu = lm.proj_from_packed(ex, "gu", gu_rows, slice(None), dev)
        dn = lm.proj_from_packed(ex, "d", torch.arange(H), slice(0, gpr), dev)
        log("weights real", os.path.basename(path))
    else:
        bits = [3 if e < N3 else 2 for e in range(E)]
        ps = []
        for n_out, K in ((768, 5120), (5120, 384)):
            codes = [torch.randint(0, 2 ** b, (n_out, K), dtype=torch.uint8) for b in bits]
            sc = [torch.rand(n_out, K // 128) * .004 + .001 for _ in range(E)]
            zp = [torch.randint(0, 2 ** b, (n_out, K // 128), dtype=torch.uint8) for b in bits]
            ps.append(lm.LowbitProj(bits, codes, sc, zp, dev))
        gu, dn = ps
        log("weights random")
    return gu, dn


def torch_route(logits, bias):   # == ArcLowbitExperts.forward
    scores = torch.sigmoid(logits.float())
    sel = scores + bias.float()
    ids = torch.topk(sel, TOPK, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True)
    return w * SCALE, ids.to(torch.int32)


def graphs_median(fns, n=10):
    gs = []
    for f in fns:
        f(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                f()
        gs.append(g)
    ts = [[] for _ in gs]
    for rnd in range(ROUNDS):
        order = list(range(len(gs)))
        order = order[rnd % len(order):] + order[:rnd % len(order)]
        for i in order:
            gs[i].replay(); torch.cuda.synchronize(); t0 = time.time()
            gs[i].replay(); gs[i].replay(); torch.cuda.synchronize()
            ts[i].append((time.time() - t0) / (2 * n) * 1e3)
    return [sorted(t)[len(t) // 2] for t in ts]


gu, dn = build()
moe = lm.LowbitMoE(gu, dn, E)
g = torch.Generator().manual_seed(7)
pop = torch.randn(E, generator=g) * 0.7
bias = (torch.randn(E, generator=g) * 0.05).to(dev)
out_lines = []
for M in Ms:
    pref = torch.randn(max(1, M // 4), E, generator=g) * 0.8
    lg = (pop[None] + pref.repeat_interleave(4, 0)[:M] + torch.randn(M, E, generator=g)) - 3.0
    logits = lg.to(dev).bfloat16()
    x = (torch.randn(M, 5120, device=dev) * 0.5).bfloat16()
    lm._TUNE = None   # tiles from ARC_LOWBIT_TUNE as in serving (defaults if unset)
    w0, i0 = torch_route(logits, bias)
    w1, i1 = lf.route(logits, bias, TOPK, True, SCALE)
    same = (torch.sort(i0, 1).values == torch.sort(i1, 1).values).all(1).float().mean().item()
    w0s = torch.gather(w0, 1, torch.argsort(i0.long(), 1)); w1s = torch.gather(w1, 1, torch.argsort(i1.long(), 1))
    wd = float(((w0s - w1s).abs() / w0s.abs()).max())
    # parts
    xh = x.half(); c1 = torch.empty(M * TOPK, gu.N, dtype=torch.bfloat16, device=dev)
    sid, eid, npp = moe_align_block_size(i0, 16, E)
    gu(xh, c1, w0.reshape(-1), sid, eid, npp, TOPK, True, False, 16, 128, 4, 2)
    c2 = torch.empty(M * TOPK, gu.N // 2, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(M * TOPK, dn.N, dtype=torch.bfloat16, device=dev)

    def silu_old():
        torch.ops._C.silu_and_mul(c2, c1)
        h = c2.to(torch.float16)
        return h, h.view(M * TOPK, -1, 128).float().sum(-1)
    torch.ops._C.silu_and_mul(c2, c1)
    c2n, xs2n = lf.silu_prep(c1)
    silu_eq = float((c2n.float() != c2.half().float()).float().mean())
    xs_o = xh.view(M, -1, 128).float().sum(-1); xh_n, xs_n = lf.prep_x(x)
    xs_rel = float(((xs_o - xs_n).abs() / xs_o.abs().clamp_min(1e-3)).max())
    parts = {"route": (lambda: torch_route(logits, bias), lambda: lf.route(logits, bias, TOPK, True, SCALE)),
             "prep_x": (lambda: (x.to(torch.float16), x.to(torch.float16).view(M, -1, 128).float().sum(-1)),
                        lambda: lf.prep_x(x)),
             "silu": (silu_old, lambda: lf.silu_prep(c1)),
             "combine": (lambda: c3.view(M, TOPK, dn.N).sum(1), lambda: combine(c3, M, TOPK))}
    fns = [f for p in parts.values() for f in p]
    ts = graphs_median(fns)
    pt = {k: (ts[2 * i], ts[2 * i + 1]) for i, k in enumerate(parts)}
    log(f"M={M} parts us old/new " + " ".join(f"{k} {a * 1e3:.1f}/{b * 1e3:.1f}" for k, (a, b) in pt.items()))
    log(f"M={M} route ids same {same:.4f} max w rel diff {wd:.2e} | silu_prep c2 != ref frac {silu_eq:.2e} | "
        f"xs max rel diff {xs_rel:.2e}")
    aux_old, aux_new = sum(a for a, _ in pt.values()), sum(b for _, b in pt.values())
    # full layer incl. routing
    lay_old = lambda: moe(x, *torch_route(logits, bias))
    lay_new = lambda: lf.moe_forward(moe, x, *lf.route(logits, bias, TOPK, True, SCALE))
    y0, y1 = lay_old(), lay_new()
    rel = float((y1.float() - y0.float()).norm() / y0.float().norm())
    mx = float((y1.float() - y0.float()).abs().max() / y0.float().abs().max())
    # bf16 noise scale for comparison: the same layer with x perturbed by one bf16 ulp-ish rounding (fp16 round trip)
    t_old, t_new = graphs_median([lay_old, lay_new])
    ok = same >= 0.999 and rel < 5e-3 and aux_new < 0.95 * aux_old and t_new < t_old
    log(f"M={M} layer old {t_old:.3f} ms new {t_new:.3f} ms ({(1 - t_new / t_old) * 100:+.1f}%) | aux old "
        f"{aux_old * 1e3:.1f} us new {aux_new * 1e3:.1f} us | out rel L2 {rel:.2e} max {mx:.2e}")
    out_lines.append(f"FUSE_GATE M={M} aux_old {aux_old:.4f} aux_new {aux_new:.4f} layer_old {t_old:.4f} "
                     f"layer_new {t_new:.4f} rel {rel:.2e} ids {same:.4f} ok {ok}")
for l in out_lines:
    print(l, flush=True)
log(f"done in {time.time() - T0:.0f} s")
