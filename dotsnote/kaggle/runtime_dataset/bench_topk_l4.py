"""Sharded-indexer top-k on one GPU at production shape (arcstats.fast_topk vs the torch.topk path of
shard_indexer.sharded_topk): NREQ requests x 4 MTP rows, decode contexts sampled from ctx_lengths.json, P=4 ranks
simulated on this GPU (one key store per rank), Lc = max_model_len / P like the decode path.
  python dots3/bench_topk_l4.py [NREQ] [BS] [MAX_MODEL_LEN]     (env ARC_SHARD_KLOC, BENCH_TOPK_QUICK=1)
Exactness: the full per-step pipeline (P local selects -> rank-major payload -> merge) must give the identical index
set per topk_indices_buffer row as the torch path (and each rank's local selection the same set as torch.topk), on the
indexer logits and on tie-heavy / short-row / fp16 stress inputs; bitwise row differences (order of exactly equal
values only) are reported for information.
Timing (CUDA graphs) per rank and DSA layer: logits kernel (with / without the -inf tail) + local top-k + merge.
Prints 'topk exact1 <0|1> exact2 <0|1>' and 'topk orig <ms> fast1 <ms> fast2 <ms>' (the startup gate parses these)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, HERE); import triton_sm89  # noqa
import torch
from arcstats import shard_indexer as si, shard_logits_v2 as v2, fast_topk as ft

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 63
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
MML = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
CF = next(p for p in (os.path.join(HERE, "ctx_lengths.json"), os.path.join(HERE, "..", "kaggle", "l4run", "ctx_lengths.json"))
          if os.path.exists(p))
QUICK = os.environ.get("BENCH_TOPK_QUICK") == "1"
G, H, D, P = 4, 64, 128, 4
KLOC = int(os.environ.get("ARC_SHARD_KLOC", "768"))
TOPK = 2048
cl = json.load(open(CF))["ctx"]
g = torch.Generator().manual_seed(1)
ctx_req = torch.tensor([min(cl[i], MML - 8) for i in torch.randint(len(cl), (NREQ,), generator=g).tolist()], device=dev)
pages = int((ctx_req.max() + BS - 1) // BS) + 1
NB = NREQ * pages + 4
RPB = BS // P
stores = []
for rk in range(P):
    rows_b = torch.empty(NB * RPB, si.ROWB, dtype=torch.uint8, device=dev)
    rows_b[:, :D] = (torch.randn(NB * RPB, D, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    rows_b[:, D:] = (torch.rand(NB * RPB, 1, device=dev) * 0.9 + 0.1).view(torch.uint8)
    stores.append(si.ShardStore(rows_b.view(NB, BS, si.ROWB // P), P, rk))
bt = torch.randperm(NB - 4, device=dev)[:NREQ * pages].view(NREQ, pages).int()
rows = NREQ * G
req = (torch.arange(rows, device=dev) // G).int()
ctx = (ctx_req.repeat_interleave(G) - (G - 1 - torch.arange(rows, device=dev) % G)).int()
q = (torch.randn(rows, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
w = torch.randn(rows, H, device=dev) * 0.1
Lc = (MML + P - 1) // P
kl = min(KLOC, Lc)
kk = min(TOPK, P * kl)
print(f"topk rows {rows} ctx_mean {int(ctx.float().mean())} max {int(ctx.max())} Lc {Lc} kl {kl} kk {kk} "
      f"on {torch.cuda.get_device_name()}", flush=True)
V2A = dict(G=G, BN=128, KB=2, warps=4, f8=False)


def logits(rk, out, noinf=False):
    return v2.logits("grp", stores[rk], q, w, bt, req, ctx, out, Lc, noinf=noinf, **V2A)


# ---- reference / fast pipelines (shard_indexer.sharded_topk logic with the all-gather replaced by a stack)
def local_torch(lg, rk, kl_):
    v, j = torch.topk(lg, kl_, dim=1)
    return torch.cat([v.float(), (j * P + rk).to(torch.int32).view(torch.float32)], 1)


def merge_torch(allp, n, kl_, out):
    kk_ = min(TOPK, P * kl_)
    a = allp.view(P, n, 2 * kl_).transpose(0, 1)
    vals = a[:, :, :kl_].reshape(n, P * kl_)
    poss = a[:, :, kl_:].contiguous().view(torch.int32).reshape(n, P * kl_)
    tv, ti = torch.topk(vals, kk_, dim=1)
    sel = torch.gather(poss, 1, ti)
    out[:, :kk_] = torch.where(torch.isinf(tv), torch.full_like(sel, -1), sel)
    if kk_ < TOPK:
        out[:, kk_:] = -1
    return out


def pipeline(lgs, ctx_, level, kl_):
    n = lgs[0].shape[0]
    if level == 0:
        allp = torch.cat([local_torch(lgs[rk], rk, kl_) for rk in range(P)], 0)
    else:
        allp = torch.cat([ft.local(lgs[rk], ctx_, rk, P, kl_) for rk in range(P)], 0)
    out = torch.full((n, TOPK), -7, dtype=torch.int32, device=dev)
    if level >= 2:
        return ft.merge(allp, n, P, kl_, min(TOPK, P * kl_), out), allp
    return merge_torch(allp, n, kl_, out), allp


def local_sets_equal(lgs, ctx_, kl_):
    """per-rank local selection vs torch.topk on the same row (finite values; exact index set)."""
    bad = 0
    for rk in range(P):
        a = local_torch(lgs[rk], rk, kl_)
        b = ft.local(lgs[rk], ctx_, rk, P, kl_)
        va, vb = a[:, :kl_], b[:, :kl_]
        pa, pb = a[:, kl_:].view(torch.int32), b[:, kl_:].view(torch.int32)
        fa, fb = torch.isfinite(va), torch.isfinite(vb)
        ka = torch.where(fa, pa.long(), torch.full_like(pa.long(), 1 << 40)).sort(1).values
        kb = torch.where(fb, pb.long(), torch.full_like(pb.long(), 1 << 40)).sort(1).values
        bad += int((ka != kb).any(1).sum())
    return bad


ok1 = ok2 = True


def check(name, lgs, ctx_, kl_=kl):
    global ok1, ok2
    ref, _ = pipeline(lgs, ctx_, 0, kl_)
    o1, _ = pipeline(lgs, ctx_, 1, kl_)
    o2, _ = pipeline(lgs, ctx_, 2, kl_)
    torch.cuda.synchronize()
    m1, m2 = int((o1 != ref).any(1).sum()), int((o2 != ref).any(1).sum())
    rs = ref.sort(1).values
    s1, s2 = int((o1.sort(1).values != rs).any(1).sum()), int((o2.sort(1).values != rs).any(1).sum())
    ls = local_sets_equal(lgs, ctx_, kl_)
    ok1 &= s1 == 0 and ls == 0
    ok2 &= s2 == 0 and ls == 0
    print(f"topk check {name}: rows {ref.shape[0]} local-set-mismatch {ls} set-mismatch fast1 {s1} fast2 {s2} | "
          f"bitwise-diff rows fast1 {m1} fast2 {m2} | valid/row {float((ref >= 0).sum(1).float().mean()):.0f}",
          flush=True)


lg_inf = [logits(rk, torch.empty(rows, Lc, device=dev)) for rk in range(P)]
torch.cuda.synchronize()
check("indexer logits fp32", lg_inf, ctx)
# noinf logits: the tail stays garbage (NaN-filled here); fast path must not read it
lg_noinf = [logits(rk, torch.full((rows, Lc), float("nan"), device=dev), noinf=True) for rk in range(P)]
torch.cuda.synchronize()
o_ref, _ = pipeline(lg_inf, ctx, 0, kl)
for lvl in (1, 2):
    o, _ = pipeline(lg_noinf, ctx, lvl, kl)
    torch.cuda.synchronize()
    mm = int((o.sort(1).values != o_ref.sort(1).values).any(1).sum())
    print(f"topk check noinf-logits fast{lvl} set-mismatch {mm} bitwise-diff rows {int((o != o_ref).any(1).sum())}",
          flush=True)
    ok1 &= lvl != 1 or mm == 0
    ok2 &= lvl != 2 or mm == 0
# fp16 logits (ARC_SHARD_F16OUT)
check("indexer logits fp16", [l.half() for l in lg_inf], ctx)
# ties: logits quantized to 1/8 and to a handful of levels, plus zeros / -0.0
check("ties q1/8", [(l * 8).round() / 8 for l in lg_inf], ctx)
check("ties 5 levels", [torch.where(torch.isfinite(l), (l.abs() * 3).floor().clamp(max=4), l) for l in lg_inf], ctx)
check("ties signed zeros", [torch.where(torch.isfinite(l), torch.where(l > 0, torch.zeros_like(l), -torch.zeros_like(l)), l)
                            for l in lg_inf], ctx)
check("ties 5 levels fp16", [torch.where(torch.isfinite(l), (l.abs() * 3).floor().clamp(max=4), l).half() for l in lg_inf],
      ctx)
# short / empty rows, -inf holes inside the valid range (dummy blocks), small Lc (prefill chunks)
ctx_s = ctx.clone(); ctx_s[::7] = torch.randint(0, 3000, (ctx_s[::7].numel(),), device=dev, dtype=torch.int32)
ctx_s[1] = 0; ctx_s[2] = 1; ctx_s[3] = P * kl; ctx_s[5] = P * kl + 1
lg_s = []
for rk in range(P):
    l = logits(rk, torch.empty(rows, Lc, device=dev))
    jj = torch.arange(Lc, device=dev)[None] * P + rk
    l = torch.where(jj < ctx_s[:, None], l, torch.full_like(l, float("-inf")))
    l[::5, 100:900] = float("-inf")
    lg_s.append(l)
check("short rows + -inf holes", lg_s, ctx_s)
for Lc2 in (3001, 1024, 700, 300, 5):   # prefill chunks: small Lc, odd Lc (scalar loads), Lc < K_LOC
    c2 = torch.randint(0, Lc2 * P + 1, (rows,), device=dev, dtype=torch.int32)
    ls2 = []
    for rk in range(P):
        l = torch.randn(rows, Lc2, device=dev)
        jj = torch.arange(Lc2, device=dev)[None] * P + rk
        ls2.append(torch.where(jj < c2[:, None], l, torch.full_like(l, float("-inf"))))
    check(f"random Lc={Lc2}", ls2, c2, min(KLOC, Lc2))
print(f"topk exact1 {int(ok1)} exact2 {int(ok2)}", flush=True)


def timeit(f, n=8, reps=5):
    f(); torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        f()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(n):
            f()
    gr.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        gr.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / (n * reps) * 1e3


out_l = torch.empty(rows, Lc, device=dev)
L0 = lg_inf[0]
allp = pipeline(lg_inf, ctx, 0, kl)[1]
tout = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
t_lg = timeit(lambda: logits(0, out_l))
t_lgn = timeit(lambda: logits(0, out_l, noinf=True))
t_lt = timeit(lambda: local_torch(L0, 0, kl))
t_mt = timeit(lambda: merge_torch(allp, rows, kl, tout))
t_mf, mnt = min((timeit(lambda: ft.merge(allp, rows, P, kl, kk, tout, nt=nt)), nt) for nt in (512, 1024))
print(f"topk logits {t_lg:.3f} ms noinf {t_lgn:.3f} ms | local torch {t_lt:.3f} ms | merge torch {t_mt:.3f} ms "
      f"fast {t_mf:.3f} ms (nt={mnt})", flush=True)
best = None
for nt in ((512, 1024) if QUICK else (256, 512, 1024)):
    for agg in ((0,) if QUICK else (0, 1)):
        t = timeit(lambda: ft.local(L0, ctx, 0, P, kl, nt=nt, agg=agg))
        print(f"topk local fast nt={nt} agg={agg} {t:.3f} ms", flush=True)
        if best is None or t < best[0]:
            best = (t, nt, agg)
L16 = L0.half()
print(f"topk local fp16 torch {timeit(lambda: local_torch(L16, 0, kl)):.3f} ms fast "
      f"{timeit(lambda: ft.local(L16, ctx, 0, P, kl, nt=best[1], agg=best[2])):.3f} ms", flush=True)
t_o = t_lg + t_lt + t_mt
t_1 = t_lgn + best[0] + t_mt
t_2 = t_lgn + best[0] + t_mf
print(f"topk best nt={best[1]} agg={best[2]} mnt={mnt}", flush=True)
print(f"topk orig {t_o:.3f} ms fast1 {t_1:.3f} ms fast2 {t_2:.3f} ms (per rank per F layer: logits + local + merge)",
      flush=True)
print(f"topk sel orig {t_lt + t_mt:.3f} ms fast1 {best[0] + t_mt:.3f} ms fast2 {best[0] + t_mf:.3f} ms "
      f"(local + merge only)", flush=True)
