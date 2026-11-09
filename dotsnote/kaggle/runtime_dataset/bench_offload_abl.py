"""Offload decode attention (split kernel, production config) ablation on one GPU, kernel time only (CUDA graph):
what bounds it -- row loads, the Q.K MMA or the P.V MMA? Also Q.K on FP8 tensor cores (timing).
  python dots3/bench_offload_abl.py [NREQ=48]
Prints 'abl <name> <ms> ms' per variant."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl
from arcstats import sparse_offload as so

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 48
Q, TOPK, H, R, CTX = 4, 2048, 32, 4_000_000, 60_000
so.CACHE_ROWS, so._WAYS, so._RESOLVE = 131072, 8, True
off = so.LayerOffload(R, dev)
base = torch.randperm(R // CTX, device=dev)[:NREQ] * CTX
hot = torch.stack([torch.randperm(CTX, device=dev)[:2400] for _ in range(NREQ)])   # ~2.4k distinct rows per request
pick = torch.rand(NREQ, Q, 2400, device=dev).argsort(-1)[..., :TOPK]
idx = (base[:, None, None] + torch.gather(hot[:, None, :].expand(-1, Q, -1), 2, pick)).reshape(-1, TOPK).int()
T = NREQ * Q
q = torch.randn(T, H, 576, device=dev, dtype=torch.bfloat16)
off.attend(q, idx, 0.04, 1.0, H, ql=Q)
rb = off._sa_prefill(idx.reshape(-1))      # all hits now: the resolved locations the kernel reads
SPLIT, BH, BN, W, ST = 2, 32, 32, 4, 3
HP = BH
ws = so._split_ws(T * SPLIT * HP, dev)
print(f"abl T={T} H={H} topk={TOPK} on {torch.cuda.get_device_name()}", flush=True)


def launch(abl=0, qk8=0):
    so._sparse_mla_offload_split_kernel[(T, 1, SPLIT)](
        q, off.host, off.cache, off.tag, idx, ws[0], ws[1], ws[2], off.C, off.R,
        q.stride(0), q.stride(1), idx.stride(0), H, 0.04 * 1.4426950408889634,
        rb["res"], rb["spill"], TOPK=TOPK, SPLIT=SPLIT, BLOCK_H=BH, BLOCK_N=BN, HP=HP, DN=so.D_NOPE, DR=so.D_ROPE,
        W=off.W, ROWMAJOR=2, RES=1, DVS=1, CD=tl.float16, ABL=abl, QK8=qk8, num_warps=W, num_stages=ST)


def timeit(f, n=10, reps=5):
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


for name, kw in (("full", {}), ("no_pv", dict(abl=1)), ("no_qk", dict(abl=2)), ("loads_only", dict(abl=3)),
                 ("qk_fp8", dict(qk8=1))):
    try:
        print(f"abl {name} {timeit(lambda: launch(**kw)):.3f} ms", flush=True)
    except Exception as ex:
        print(f"abl {name} FAILED {repr(ex)[:200]}", flush=True)

# ---- request-packed path, piece by piece (same inputs): union build, fill/resolve on union rows, pk kernel, combine
import itertools
flat = idx.reshape(-1)
print(f"abl split_prefill(all entries) {timeit(lambda: off._sa_prefill(flat)):.3f} ms", flush=True)
print(f"abl split_combine {timeit(lambda: so._combine_kernel[(T, 1)](ws[0], ws[1], ws[2], torch.empty(T, H, so.D_NOPE, device=dev, dtype=torch.bfloat16), H * so.D_NOPE, so.D_NOPE, H, 1.0, SPLIT=SPLIT, BLOCK_H=BH, HP=HP, DN=so.D_NOPE, num_warps=4)):.3f} ms", flush=True)
N = Q * TOPK
HG = min(H, 64 // Q); NHB = H // HG
BM = max(16, triton.next_power_of_2(Q * HG))
st = {}


def union():
    x = idx.reshape(NREQ, N)
    x = torch.where((x >= 0) & (x < off.R), x, 2147483647)
    srt, perm = torch.sort(x, dim=1)
    b = so._pk_bufs(NREQ, N, BM * NHB, dev)
    u, mk, cnt = b["u"][:NREQ * N], b["mk"][:NREQ * N], b["cnt"]
    mk.zero_()
    so._union_kernel[(NREQ,)](srt, perm, u, mk, cnt, N=N, TOPK=TOPK, num_warps=8)
    st.update(b=b, u=u, mk=mk, cnt=cnt)


print(f"abl pk_union(sort+kernel) {timeit(union):.3f} ms", flush=True)
union(); torch.cuda.synchronize()
print(f"abl pk_distinct_rows/req {float(st['cnt'][:NREQ].float().mean()):.0f}", flush=True)
print(f"abl pk_prefill(union rows) {timeit(lambda: off._sa_prefill(st['u'])):.3f} ms", flush=True)
rbu = off._sa_prefill(st["u"]); torch.cuda.synchronize()
b = st["b"]
best = None
for DV, BNp, Wp, KS, STp in itertools.product((512, 256, 128), (32, 64), (4, 8), (4, 8), (1, 2)):
    NDV = so.D_NOPE // DV

    def pk():
        so._dsa_pk_kernel[(NREQ, NDV * NHB, KS)](
            q, off.host, off.cache, rbu["spill"], st["u"], st["mk"], rbu["res"], st["cnt"], b["m"], b["l"], b["acc"],
            q.stride(0), q.stride(1), H, off.R, 0.04 * 1.4426950408889634,
            QL=Q, N=N, BM=BM, BN=BNp, DN=so.D_NOPE, DR=so.D_ROPE, DV=DV, KS=KS, QCHUNK=1, BD=128, HG=HG, NHB=NHB,
            num_warps=Wp, num_stages=STp)
    try:
        t = timeit(pk)
        print(f"abl pk_kernel DV={DV} BN={BNp} w={Wp} KS={KS} st={STp} {t:.3f} ms", flush=True)
        if best is None or t < best[0]:
            best = (t, DV, BNp, Wp, KS, STp)
    except Exception as ex:
        print(f"abl pk_kernel DV={DV} BN={BNp} w={Wp} KS={KS} st={STp} FAILED {repr(ex)[:120]}", flush=True)
print(f"abl pk_best {best}", flush=True)
