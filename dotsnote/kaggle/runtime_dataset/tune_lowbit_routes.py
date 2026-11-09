"""Low-bit MoE tile retune at MTP-5 decode sizes with REAL routing.

tune_lowbit.py times uniform random top-8 routing (~21 tokens per expert at M=384); decode routing is skewed (hot
experts with ~100 tokens, cold ones with 1-2), which changes padding per block_m and the work per CTA. This bench
replays per-call top-k ids recorded on a real MTP-5 decode (ARC_ROUTE_DUMP, lowbit_moe.ArcLowbitExperts) through the
pair kernel with the real low-bit weights of the same layers (TP4 rank-0 shard), for every candidate tile, in
interleaved rounds (L4 clocks drift by several % between back-to-back runs: per-candidate medians), and checks every
candidate's output bitwise against the current table's (tiles must be bit-exact).

  ARC_LOWBIT_PAIRS=1 python dots3/tune_lowbit_routes.py ROUTES.npz OUT.json [--ms 288,320,352,384] [--layers 10,30]
      [--calls 6] [--rounds 5] [--cur CURRENT_TABLE.json] [--full]
ROUTES.npz: {f"L{layer}_ids": int16 [sum M, 8], f"L{layer}_m": int32 [calls]} (dots3/tune_lowbit_routes.py --pack).
Weights: ARC_INPUT/**/lowbit-L<layer>.pt (BENCH_REAL layout), else --ckpt DIR/lowbit-L<layer>.pt.
A call with m tokens is used for size M if M-32 < m <= M, padded to M with copies of its first row (vLLM pads a
CUDA-graph batch with identical rows, which route identically).
Output: {"gate_up": {M: [bm, bn, warps, stages]}, "down": {...}} best per M by summed median over layers and calls,
plus lines 'LBR ...' with every candidate's time and exactness.
  python dots3/tune_lowbit_routes.py --pack DUMP_PREFIX OUT.npz [--layers 10,30] [--keep 40]   # from ARC_ROUTE_DUMP files
"""
import glob, json, os, sys, time

import numpy as np

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d

if "--pack" in sys.argv:   # dump files (one per layer) -> one small npz with decode calls only (M % 6 == 0, M <= 384)
    pre, out = sys.argv[sys.argv.index("--pack") + 1: sys.argv.index("--pack") + 3]
    keep = int(arg("--keep", "40"))
    res = {}
    for L in [int(x) for x in arg("--layers", "10,30").split(",")]:
        d = np.load(f"{pre}_L{L:02d}.npz")
        ids, m = d["ids"], d["m"]
        off = np.concatenate([[0], np.cumsum(m)])
        sel_ids, sel_m = [], []
        by = {}
        for i, mm in enumerate(m):
            if mm % 6 == 0 and 256 < mm <= 384:
                by.setdefault((mm + 31) // 32 * 32, []).append(i)
        for M, idx in sorted(by.items()):
            idx = idx[-keep:]                                 # latest calls (longest contexts)
            for i in idx:
                sel_ids.append(ids[off[i]:off[i + 1]]); sel_m.append(m[i])
            print(f"L{L} M<={M}: {len(by[M])} decode calls, kept {len(idx)}")
        res[f"L{L}_ids"] = np.concatenate(sel_ids).astype(np.int16)
        res[f"L{L}_m"] = np.array(sel_m, dtype=np.int32)
    np.savez_compressed(out, **res)
    print("wrote", out, {k: v.shape for k, v in res.items()})
    sys.exit(0)

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_sm89  # noqa: F401
import arcstats.lowbit_moe as lm  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402

T0 = time.time()
ROUTES, OUT = sys.argv[1], sys.argv[2]
Ms = [int(m) for m in arg("--ms", "288,320,352,384").split(",")]
LAYERS = [int(x) for x in arg("--layers", "10,30").split(",")]
NCALLS, ROUNDS = int(arg("--calls", "6")), int(arg("--rounds", "5"))
DEADLINE = float(os.environ.get("BENCH_DEADLINE_S", "1e9"))
FULL = "--full" in sys.argv
E, TOPK = 144, 8
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
assert lm.PAIRS, "set ARC_LOWBIT_PAIRS=1"
log = lambda *a: print("LBR", *a, flush=True)
CUR = {"gate_up": {"288": [16, 128, 4, 2], "320": [16, 64, 4, 2], "352": [32, 64, 4, 2], "384": [32, 64, 4, 2]},
       "down": {"288": [16, 128, 4, 2], "320": [32, 128, 4, 3], "352": [32, 128, 4, 2], "384": [32, 128, 4, 3]}}
if arg("--cur", None):
    CUR = json.load(open(arg("--cur", None)))
cur_tile = lambda proj, M: tuple(next(c for m, c in sorted((int(k), v) for k, v in CUR[proj].items()) if m >= M)
                                 if any(int(k) >= M for k in CUR[proj]) else
                                 sorted((int(k), v) for k, v in CUR[proj].items())[-1][1])


def weights(L):
    c = sorted(glob.glob(os.environ.get("ARC_INPUT", "/kaggle/input") + f"/**/lowbit-L{L:02d}.pt", recursive=True))
    path = c[0] if c else os.path.join(arg("--ckpt", "dots3/ckpt/lb144r/lowbit"), f"lowbit-L{L:02d}.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    if os.environ.get("ARC_DELTA_FILE"):
        from arcstats.delta_overlay import apply_lowbit
        apply_lowbit(L, d)
    ex = [d[e] for e in d["experts"]]
    I, tp, r = 1536, 4, 0
    Is = I // tp
    gu_rows = torch.cat([torch.arange(r * Is, (r + 1) * Is), I + torch.arange(r * Is, (r + 1) * Is)])
    H = ex[0]["d"]["p2" if "p2" in ex[0]["d"] else "p4"].shape[0]
    gpr = ex[0]["d_s"].shape[1] // tp
    gu = lm.proj_from_packed(ex, "gu", gu_rows, slice(None), dev)
    dn = lm.proj_from_packed(ex, "d", torch.arange(H), slice(r * gpr, (r + 1) * gpr), dev)
    log(f"L{L} weights {os.path.basename(path)} 3-bit experts {[int(e['bits']) for e in ex].count(3)} "
        f"MB gate_up {gu.nbytes() / 1e6:.0f} down {dn.nbytes() / 1e6:.0f}")
    return gu, dn


def calls_for(R, L, M):
    """NCALLS recorded decode calls for size M: calls with M-32 < m <= M, else the first n x 6 rows of larger calls
    (rows of one request are consecutive, so this is a batch of n whole requests; n = mid-bucket), padded to M."""
    ids, m = R[f"L{L}_ids"], R[f"L{L}_m"]
    off = np.concatenate([[0], np.cumsum(m)])
    exact_ = [i for i in range(len(m) - 1, -1, -1) if M - 32 < m[i] <= M]
    take = None
    if not exact_:
        take = 6 * ((M - 16) // 6)
        exact_ = [i for i in range(len(m) - 1, -1, -1) if m[i] >= take]
    out = []
    for i in exact_[:NCALLS]:
        x = torch.from_numpy(ids[off[i]:off[i + 1]].astype(np.int32))
        if take:
            x = x[:take]
        if x.shape[0] < M:
            x = torch.cat([x, x[:1].repeat(M - x.shape[0], 1)])
        out.append(x.to(dev))
    return out


def cands(M):
    bms = (16, 32, 64) if FULL else (16, 32)
    return [(bm, bn, nw, ns) for bm in bms for bn in (64, 128, 256) for nw in ((4, 8) if FULL else (4,))
            for ns in ((2, 3, 4) if FULL else (2, 3))]


R = np.load(ROUTES)
tot = {p: {M: {} for M in Ms} for p in ("gate_up", "down")}   # proj -> M -> cfg -> [summed median ms]
exact = {p: {M: {} for M in Ms} for p in ("gate_up", "down")}
for L in LAYERS:
    gu, dn = weights(L)
    for M in Ms:
        if time.time() - T0 > DEADLINE:
            log("deadline reached, skipping", L, M); continue
        calls = calls_for(R, L, M)
        if not calls:
            log(f"L{L} M={M}: no recorded calls"); continue
        cnt = torch.stack([torch.bincount(t.flatten().long(), minlength=E) for t in calls]).float()
        log(f"L{L} M={M} calls {len(calls)} experts touched {float((cnt > 0).sum(1).float().mean()):.0f} max tokens "
            f"{float(cnt.max(1).values.mean()):.0f} blocks16 {float(((cnt + 15) // 16).sum(1).mean()):.0f} "
            f"blocks32 {float(((cnt + 31) // 32).sum(1).mean()):.0f}")
        for name, P in (("gate_up", gu), ("down", dn)):
            xpt, tk, mw = (True, TOPK, False) if name == "gate_up" else (False, 1, True)
            rows = M if xpt else M * TOPK
            g = torch.Generator(device=dev).manual_seed(L * 1000 + M)
            xin = (torch.randn(rows, P.K, device=dev, generator=g) * 0.5).half()
            xs = xin.view(rows, -1, 128).float().sum(-1)
            tws = [(torch.rand(M, TOPK, device=dev, generator=g) * 0.3 + 0.05).float().reshape(-1) for _ in calls]
            als = {bm: [moe_align_block_size(t, bm, E) for t in calls] for bm in (16, 32, 64)}
            cfgs = cands(M)
            c0 = cur_tile(name, M)
            if c0 not in cfgs:
                cfgs.append(c0)
            outs, graphs = {}, {}
            for cfg in cfgs:   # per cfg: one graph replaying every call once (x 3), output of the last call kept
                o = [torch.empty(M * TOPK, P.N, dtype=torch.bfloat16, device=dev) for _ in calls]
                f = lambda cfg=cfg, o=o: [P(xin, o[i], tws[i], *als[cfg[0]][i], tk, xpt, mw, *cfg, xs=xs)
                                          for i in range(len(calls))]
                try:
                    f(); torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        for _ in range(3):
                            f()
                    gr.replay(); torch.cuda.synchronize()
                except Exception as ex_:
                    log(f"L{L} {name} M={M} {cfg} FAILED {repr(ex_)[:120]}"); continue
                graphs[cfg], outs[cfg] = gr, o
            ref = outs.get(c0)
            times = {cfg: [] for cfg in graphs}
            for _ in range(ROUNDS):
                for cfg, gr in graphs.items():
                    torch.cuda.synchronize(); t0 = time.perf_counter(); gr.replay(); torch.cuda.synchronize()
                    times[cfg].append((time.perf_counter() - t0) / 3 / len(calls) * 1e3)
            res = sorted((float(np.median(v)), cfg) for cfg, v in times.items())
            for t, cfg in res:
                ok = ref is not None and all(torch.equal(a, b) for a, b in zip(outs[cfg], ref))
                tot[name][M][cfg] = tot[name][M].get(cfg, 0.0) + t
                exact[name][M][cfg] = exact[name][M].get(cfg, True) and ok
            tc = float(np.median(times[c0])) if c0 in times else float("nan")
            wbytes = P.nbytes()
            log(f"L{L} {name} M={M} cur {'/'.join(map(str, c0))} {tc:.3f} ms ({wbytes / tc / 1e6:.0f} GB/s) best "
                f"{'/'.join(map(str, res[0][1]))} {res[0][0]:.3f} ms ({wbytes / res[0][0] / 1e6:.0f} GB/s) | " +
                " ".join(f"{'/'.join(map(str, c))}:{t:.3f}{'' if exact[name][M][c] else 'X'}" for t, c in res))
            del graphs, outs
            torch.cuda.empty_cache()
    del gu, dn
    torch.cuda.empty_cache()

table = {}
for name in ("gate_up", "down"):
    table[name] = {}
    for M in Ms:
        ok = {c: t for c, t in tot[name][M].items() if exact[name][M].get(c)}
        if not ok:
            continue
        best = min(ok, key=ok.get)
        c0 = cur_tile(name, M)
        table[name][str(M)] = list(best)
        log(f"SUMMARY {name} M={M} cur {'/'.join(map(str, c0))} {tot[name][M].get(c0, float('nan')):.3f} -> best "
            f"{'/'.join(map(str, best))} {ok[best]:.3f} ms (sum over layers; "
            f"{(ok[best] / tot[name][M].get(c0, float('nan')) - 1) * 100:+.1f}%) exact {exact[name][M][best]} "
            f"inexact cfgs {sum(1 for v in exact[name][M].values() if not v)}")
json.dump(table, open(OUT, "w"), indent=1)
log(f"done in {time.time() - T0:.0f} s -> {OUT}: {json.dumps(table)}")
