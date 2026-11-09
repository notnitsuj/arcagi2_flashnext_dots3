"""Low-bit MoE on the target GPU: pair kernel (current) vs expert-stationary lowbit2 kernel, per-rank TP4 Dots3 shapes,
skewed synthetic routing (calibrated to the lb144r gates on captured MoE inputs: ~205 16-token blocks, ~135 experts
touched, hottest expert ~100 tokens at M=256), CUDA-graph timing. Checks lowbit2 == pair kernel bitwise.
  ARC_LOWBIT_PAIRS=1 python dots3/bench_lowbit2.py [Ms=256,320,384] [--quick] [--gate] [--real LOWBIT_FILE]
Weights: random codes with the checkpoint bit mix (96 x 3-bit + 48 x 2-bit), or --real: one layer of the lowbit
checkpoint (rank 0 shard; also found via ARC_INPUT/**/lowbit-L10.pt when BENCH_REAL=1).
Output lines: 'lb2 ...'. With --gate the last line is
  'LB2_GATE M=<m> old <ms> new <ms> exact <bool> cfg_gu <...> cfg_dn <...>' per M (full MoE layer, ms)
and the tuned table is written to the json given by BENCH_OUT."""
import glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_sm89  # noqa: F401
import torch
import arcstats.lowbit_moe as lm
import arcstats.lowbit2 as l2
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

T0 = time.time()
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
Ms = [int(m) for m in (args[0] if args else "256,320,384").split(",")]
QUICK, GATE = "--quick" in sys.argv, "--gate" in sys.argv
DEADLINE = float(os.environ.get("BENCH_DEADLINE_S", "1e9"))
E, N3, TOPK = 144, 96, 8
assert lm.PAIRS, "set ARC_LOWBIT_PAIRS=1"


def log(*a):
    print("lb2", *a, flush=True)


def timeit(f, n=10, reps=5):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        t0 = time.time(); g.replay(); torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / n * 1e3)
    del g
    return best


def routing(M, seed, sp=0.7, sq=0.8):
    """Skewed top-8 routing: expert popularity + per-sequence preference shared by its 4 MTP tokens + noise."""
    g = torch.Generator().manual_seed(seed)
    pop = torch.randn(E, generator=g) * sp
    pref = torch.randn(max(1, M // 4), E, generator=g) * sq
    sc = pop[None] + pref.repeat_interleave(4, 0)[:M] + torch.randn(M, E, generator=g)
    ids = torch.topk(sc, TOPK, -1).indices
    w = torch.rand(M, TOPK, generator=g) + 0.1
    w = w / w.sum(-1, keepdim=True) * 2.5
    return w.to(dev).float(), ids.to(dev).int()


def build():
    real = next((a for a in sys.argv if a.startswith("--real=")), None)
    path = real.split("=", 1)[1] if real else None
    if path is None and os.environ.get("BENCH_REAL") == "1":
        c = sorted(glob.glob(os.environ.get("ARC_INPUT", "/kaggle/input") + "/**/lowbit-L10.pt", recursive=True))
        path = c[0] if c else None
    if path:
        d = torch.load(path, map_location="cpu", weights_only=False)
        ex = [d[e] for e in d["experts"]]
        I, tp, r = 1536, 4, 0
        Is = I // tp
        gu_rows = torch.cat([torch.arange(r * Is, (r + 1) * Is), I + torch.arange(r * Is, (r + 1) * Is)])
        H = ex[0]["d"]["p2" if "p2" in ex[0]["d"] else "p4"].shape[0]
        gpr = ex[0]["d_s"].shape[1] // tp
        gu = lm.proj_from_packed(ex, "gu", gu_rows, slice(None), dev)
        dn = lm.proj_from_packed(ex, "d", torch.arange(H), slice(r * gpr, (r + 1) * gpr), dev)
        log("weights real", os.path.basename(path), [int(e["bits"]) for e in ex].count(3), "x 3-bit")
    else:
        bits = [3 if e < N3 else 2 for e in range(E)]
        ps = []
        for n_out, K in ((768, 5120), (5120, 384)):
            codes = [torch.randint(0, 2 ** b, (n_out, K), dtype=torch.uint8) for b in bits]
            sc = [torch.rand(n_out, K // 128) * .004 + .001 for _ in range(E)]
            zp = [torch.randint(0, 2 ** b, (n_out, K // 128), dtype=torch.uint8) for b in bits]
            ps.append(lm.LowbitProj(bits, codes, sc, zp, dev))
        gu, dn = ps
        log("weights random", N3, "x 3-bit")
    assert gu.pairs and dn.pairs
    return gu, dn


gu, dn = build()
moe = lm.LowbitMoE(gu, dn, E)
wb = {"gate_up": gu.nbytes(), "down": dn.nbytes()}
log("weight MB/layer", {k: round(v / 1e6, 1) for k, v in wb.items()})
OLD_CFGS = [(bm, bn, 4, ns, ev) for bm in (16, 32) for bn in (64, 128, 256) for ns in (2, 3) for ev in (0, 1)
            if not (bm == 32 and (bn == 256 or ns == 3))] + [(64, 128, 4, 2, 0)]   # (64,128): today's M>256 tile
NEW_CFGS = [(bn, cm, 4, 2, ev) for bn in (64, 128, 256) for cm in (2, 4) for ev in (0, 1) if not (bn == 256 and cm == 4)]
if QUICK:
    OLD_CFGS = [(16, 64, 4, 2, 0), (16, 128, 4, 2, 0), (16, 128, 4, 2, 1)]
    NEW_CFGS = [(256, 2, 4, 2, 0), (128, 4, 4, 2, 1)]
fmt = lambda c: "/".join(map(str, c))
table = {"moe2": {}, "gate_up": {}, "down": {}}
gate_lines = []
for M in Ms:
    if time.time() - T0 > DEADLINE:
        log("deadline reached, skipping M", M); continue
    tw, tid = routing(M, 1000 + M)
    cnt = torch.bincount(tid.flatten().long(), minlength=E)
    log(f"M={M} routing blocks16 {int(((cnt + 15) // 16).sum())} touched {int((cnt > 0).sum())} max {int(cnt.max())}")
    x = (torch.randn(M, 5120, device=dev) * 0.5).bfloat16()
    xh = x.half()
    xs_gu = xh.view(M, -1, 128).float().sum(-1)
    sid, eid, npp = moe_align_block_size(tid, 16, E)
    c1 = torch.empty(M * TOPK, gu.N, dtype=torch.bfloat16, device=dev)
    gu(xh, c1, tw.reshape(-1), sid, eid, npp, TOPK, True, False, 16, 128, 4, 2, xs=xs_gu)   # real down inputs
    c2 = torch.empty(M * TOPK, gu.N // 2, dtype=torch.bfloat16, device=dev)
    torch.ops._C.silu_and_mul(c2, c1)
    c2h = c2.half()
    xs_dn = c2h.view(M * TOPK, -1, 128).float().sum(-1)
    c3 = torch.empty(M * TOPK, dn.N, dtype=torch.bfloat16, device=dev)
    dn(c2h, c3, tw.reshape(-1), sid, eid, npp, 1, False, True, 16, 128, 4, 2, xs=xs_dn)
    # --- aux ops of the current path (per layer; activations hot in L2 here, colder in serving)
    aux = {"x_to_f16": lambda: x.to(torch.float16), "xs_gu": lambda: xh.view(M, -1, 128).float().sum(-1),
           "align16": lambda: moe_align_block_size(tid, 16, E), "items": lambda: l2.make_items(eid, npp, 2),
           "silu_mul": lambda: torch.ops._C.silu_and_mul(c2, c1), "c2_to_f16": lambda: c2.to(torch.float16),
           "xs_dn": lambda: c2h.view(M * TOPK, -1, 128).float().sum(-1),
           "combine_sum": lambda: c3.view(M, TOPK, dn.N).sum(1), "combine_kern": lambda: l2.combine(c3, M, TOPK)}
    log(f"M={M} aux_us", {k: round(timeit(f) * 1e3, 1) for k, f in aux.items()})
    ce = torch.equal(l2.combine(c3, M, TOPK), c3.view(M, TOPK, dn.N).sum(1))
    log(f"M={M} combine kernel bitwise == torch sum: {ce}")
    # --- per projection: old pair kernel (block_m tiles, +evict) vs expert-stationary kernel
    al = {bm: moe_align_block_size(tid, bm, E) for bm in (16, 32, 64)}
    items = {cm: l2.make_items(eid, npp, cm) for cm in (2, 4)}
    best = {}
    for name, P, xin, xs, tk, xpt, mw in (("gate_up", gu, xh, xs_gu, TOPK, True, False),
                                         ("down", dn, c2h, xs_dn, 1, False, True)):
        ref = torch.empty(M * TOPK, P.N, dtype=torch.bfloat16, device=dev)
        P(xin, ref, tw.reshape(-1), *al[16], tk, xpt, mw, 16, 128, 4, 2, xs=xs)
        res = []
        for kind, cfgs in (("old", OLD_CFGS), ("new", NEW_CFGS)):
            for cfg in cfgs:
                o = torch.zeros_like(ref)
                if kind == "old":
                    f = lambda: P(xin, o, tw.reshape(-1), *al[cfg[0]], tk, xpt, mw, *cfg[:4], xs=xs, evict=cfg[4])
                else:
                    f = lambda: l2.gemm2(P, xin, o, tw.reshape(-1), sid, eid, *items[cfg[1]], tk, xpt, mw, cfg[:4],
                                         xs=xs, evict=cfg[4])
                try:
                    t = timeit(f)
                    o.zero_(); f(); torch.cuda.synchronize()
                    ok = bool(torch.equal(o, ref))
                except Exception as ex:
                    log(name, M, kind, cfg, "FAILED", repr(ex)[:160]); continue
                res.append((t, kind, cfg, ok))
        for kind in ("old", "new"):
            r = sorted(x for x in res if x[1] == kind)
            if r:
                log(f"{name} M={M} {kind} best {fmt(r[0][2])} {r[0][0]:.3f} ms {wb[name] / r[0][0] / 1e6:.0f} GB/s exact_all "
                    f"{all(x[3] for x in r)} | " + " ".join(f"{fmt(c)}:{t:.3f}{'' if ok else 'X'}" for t, _, c, ok in r))
        ok_res = sorted(x for x in res if x[3])
        best[name] = ok_res
        oldr = sorted(x for x in ok_res if x[1] == "old")
        table[name][str(M)] = list(oldr[0][2][:4]) if oldr else [16, 128, 4, 2]
        # memory probe: loads of the expert-stationary kernel only (no dequant / MMA) = streaming floor
        o = torch.zeros_like(ref)
        tp = timeit(lambda: l2.gemm2(P, xin, o, tw.reshape(-1), sid, eid, *items[2], tk, xpt, mw, (256, 2, 4, 2),
                                     xs=xs, evict=1, probe=1))
        log(f"{name} M={M} probe(loads only) {tp:.3f} ms {wb[name] / tp / 1e6:.0f} GB/s")
    # --- full layer: current LowbitMoE (best old tiles, no evict, torch sum) vs pipeline variants, bitwise
    ob = lambda name: next(x for x in best[name] if x[1] == "old" and x[2][4] == 0)
    lm._TUNE = {"gate_up": [(M, ob("gate_up")[2][:4])], "down": [(M, ob("down")[2][:4])]}
    y_ref = moe(x, tw, tid)
    as_cfg = lambda r, ev=None: [r[1]] + list(r[2][:4]) + [r[2][4] if ev is None else ev]
    gu_c = [as_cfg(best["gate_up"][0])]
    dn_c = {json.dumps(as_cfg(r, ev)) for r in (best["down"][0], ob("down")) for ev in (1, 2)}
    dn_c |= {json.dumps(as_cfg(ob("down")))}
    variants = [{"gu": g, "dn": json.loads(d), "comb": cb} for g in gu_c + [as_cfg(ob("gate_up"))]
                for d in sorted(dn_c) for cb in (0, 1, 2)]
    variants = [json.loads(v) for v in dict.fromkeys(json.dumps(v) for v in variants)]
    # interleaved rounds (L4 clocks drift by several % between back-to-back measurements): per-variant median
    graphs, vres = [], []
    for v in variants:
        try:
            y = l2.moe_forward(moe, x, tw, tid, v)
            ex = bool(torch.equal(y, y_ref))
            f = (lambda v=v: l2.moe_forward(moe, x, tw, tid, v))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    f()
            graphs.append((g, ex, v))
        except Exception as e_:
            log(f"M={M} layer variant {json.dumps(v)} FAILED {repr(e_)[:160]}")
    gb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb):
        for _ in range(10):
            moe(x, tw, tid)
    allg = [(gb, True, None)] + graphs
    times = [[] for _ in allg]
    for rnd in range(int(os.environ.get("BENCH_ROUNDS", "9"))):
        order = list(range(len(allg)))
        order = order[rnd % len(order):] + order[:rnd % len(order)]
        for i in order:
            allg[i][0].replay(); torch.cuda.synchronize(); t0 = time.time()
            allg[i][0].replay(); allg[i][0].replay(); torch.cuda.synchronize()
            times[i].append((time.time() - t0) / 20 * 1e3)
    med = lambda a: sorted(a)[len(a) // 2]
    t_base = med(times[0])
    for (g, ex, v), ts in zip(allg[1:], times[1:]):
        t = med(ts)
        vres.append((t, ex, v))
        log(f"M={M} layer {t:.3f} ms ({(1 - t / t_base) * 100:+.1f}%) exact {ex} {json.dumps(v)}")
    del allg, graphs, gb
    good = sorted((r for r in vres if r[1]), key=lambda r: r[0])
    bt, _, bv = good[0] if good else (t_base, True, None)
    log(f"M={M} layer base {t_base:.3f} ms best {bt:.3f} ms ({(1 - bt / t_base) * 100:+.1f}%) {json.dumps(bv)}")
    table["moe2"][str(M)] = bv if bt < t_base else None
    gate_lines.append(f"LB2_GATE M={M} old {t_base:.4f} new {min(bt, t_base):.4f} exact {bool(vres) and all(r[1] for r in vres)} "
                      f"cfg {json.dumps(bv)}")
for l in gate_lines:
    print(l, flush=True)
if os.environ.get("BENCH_OUT"):
    json.dump(table, open(os.environ["BENCH_OUT"], "w"), indent=1)
log(f"done in {time.time() - T0:.0f} s")
