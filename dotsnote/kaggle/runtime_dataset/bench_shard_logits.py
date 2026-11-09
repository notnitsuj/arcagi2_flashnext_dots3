"""Sharded indexer decode logits: per-row kernel vs request-packed FP8 kernel (arcstats.shard_logits_pk), correctness +
CUDA-graph timing at long contexts.  python dots3/bench_shard_logits.py [NREQ] [CTX_MEAN]
Prints 'row X ms  pk Y ms  maxdiff Z  inf_mismatch N'."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
from arcstats import shard_indexer as si, shard_logits_pk as pk

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 24
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 90000
G, H, D, P, RANK, BS = 4, 64, 128, 4, 0, 64
ctx_req = torch.randint(CTX // 3, CTX * 5 // 3, (NREQ,), device=dev)
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


def run(packed):
    si._PK = packed
    return store.logits(q, w, bt, req, ctx, maxc, group=G)


a, b = run(False), run(True)
fa, fb = torch.isfinite(a), torch.isfinite(b)
mism = int((fa != fb).sum())
both = fa & fb
md = float(((a - b).abs()[both]).max() / a.abs()[both].max()) if both.any() else 0.0


def timeit(packed, n=10, reps=5):
    si._PK = packed
    f = lambda: store.logits(q, w, bt, req, ctx, maxc, group=G)
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


print(f"rows {rows} ctx_mean {int(ctx.float().mean())} Lc {(maxc + P - 1) // P}")
print(f"row {timeit(False):.3f} ms  pk {timeit(True):.3f} ms  maxdiff {md:.2e}  inf_mismatch {mism}")
