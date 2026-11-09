"""Sharded indexer decode logits on one GPU at production shape: NREQ requests x 4 MTP rows, decode contexts sampled
from ctx_lengths.json, this rank's keys (P=4). Variants: original per-row kernel, shard_logits_pk, shard_logits_v2
(row8 / grp, FP8 or fp16 MMA, tile sweep); also the local top-K_LOC that follows. CUDA-graph timing.
  python dots3/bench_idx_l4.py [NREQ] [BS] [ctx_lengths.json]
Prints one line per variant: 'idx <name> <ms> maxrel <err> infmis <n>'."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, HERE); import triton_sm89  # noqa
import torch
from arcstats import shard_indexer as si, shard_logits_pk as pk, shard_logits_v2 as v2

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
CF = sys.argv[3] if len(sys.argv) > 3 else next(p for p in (os.path.join(HERE, "ctx_lengths.json"),
                                                            os.path.join(HERE, "..", "kaggle", "l4run", "ctx_lengths.json"))
                                                if os.path.exists(p))
G, H, D, P, RANK = 4, 64, 128, 4, 0
cl = json.load(open(CF))["ctx"]
g = torch.Generator().manual_seed(1)
ctx_req = torch.tensor([cl[i] for i in torch.randint(len(cl), (NREQ,), generator=g).tolist()], device=dev)
pages = int((ctx_req.max() + BS - 1) // BS) + 1
NB = NREQ * pages + 4
RPB = BS // P
rows_b = torch.empty(NB * RPB, si.ROWB, dtype=torch.uint8, device=dev)
rows_b[:, :D] = (torch.randn(NB * RPB, D, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
rows_b[:, D:] = (torch.rand(NB * RPB, 1, device=dev) * 0.9 + 0.1).view(torch.uint8)
kc = rows_b.view(NB, BS, si.ROWB // P)
store = si.ShardStore(kc, P, RANK)
bt = torch.randperm(NB - 4, device=dev)[:NREQ * pages].view(NREQ, pages).int()
rows = NREQ * G
req = (torch.arange(rows, device=dev) // G).int()
ctx = (ctx_req.repeat_interleave(G) - (G - 1 - torch.arange(rows, device=dev) % G)).int()
q = (torch.randn(rows, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
w = torch.rand(rows, H, device=dev) * 0.1
maxc = int(ctx.max())
Lc = (maxc + P - 1) // P
print(f"idx rows {rows} ctx_mean {int(ctx.float().mean())} max {maxc} Lc {Lc} BS {BS} "
      f"keys/rank {int((ctx_req // P).sum())} on {torch.cuda.get_device_name()}", flush=True)


def ref():
    si._PK = False
    return store.logits(q, w, bt, req, ctx, maxc, group=G)


QUICK = os.environ.get("BENCH_IDX_QUICK") == "1"
R = ref()
fR = torch.isfinite(R)
scale = float(R[fR].abs().max())


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


def report(name, f):
    try:
        o = f(); torch.cuda.synchronize()
        fo = torch.isfinite(o)
        mis = int((fo != fR).sum())
        both = fo & fR
        err = float((o - R).abs()[both].max()) / scale if both.any() else 0.0
        print(f"idx {name} {timeit(f):.3f} ms maxrel {err:.2e} infmis {mis}", flush=True)
    except Exception as ex:
        print(f"idx {name} FAILED {repr(ex)[:160]}", flush=True)


report("orig_row", ref)
if not QUICK:
    si._PK = True
    report("pk_default", lambda: store.logits(q, w, bt, req, ctx, maxc, group=G))
    si._PK = False
out = torch.empty(rows, Lc, device=dev, dtype=torch.float32)
QUICK = os.environ.get("BENCH_IDX_QUICK") == "1"   # startup gate: the plausible configs only
for f8 in (True, False):
    for BN in ((128,) if QUICK else (32, 64, 128)):
        for wp in ((4,) if QUICK else (2, 4, 8)):
            for stg in ((2,) if QUICK else (1, 2)):
                report(f"row8 f8={int(f8)} BN={BN} w={wp} st={stg}",
                       lambda: v2.logits("row8", store, q, w, bt, req, ctx, out, Lc, BN=BN, warps=wp, stages=stg, f8=f8))
    for BN in ((64, 128) if QUICK else (32, 64, 128)):
        for KB in (1, 2, 4):
            for wp in ((4,) if QUICK else (4, 8)):
                report(f"grp f8={int(f8)} BN={BN} KB={KB} w={wp}",
                       lambda: v2.logits("grp", store, q, w, bt, req, ctx, out, Lc, G=G, BN=BN, KB=KB, warps=wp, f8=f8))
kl = min(si.K_LOC, Lc)
print(f"idx topk{kl} {timeit(lambda: torch.topk(R, kl, dim=1)):.3f} ms", flush=True)
out16 = torch.empty(rows, Lc, device=dev, dtype=torch.float16)
for f8 in (False, True):
    for KB in (1, 2):
        report(f"grp16 f8={int(f8)} BN=128 KB={KB} w=4",
               lambda: v2.logits("grp", store, q, w, bt, req, ctx, out16, Lc, G=G, BN=128, KB=KB, warps=4, f8=f8).float())
R16 = R.half()
print(f"idx topk{kl}_f16 {timeit(lambda: torch.topk(R16, kl, dim=1)):.3f} ms", flush=True)
