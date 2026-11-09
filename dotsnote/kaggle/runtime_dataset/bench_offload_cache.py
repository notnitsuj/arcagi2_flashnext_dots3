"""Timing of one offload decode-attention call (attention + cache fill) at L4-like load: 40 requests x 4 MTP tokens x
top-2048, per-request hot sets drifting slowly across steps.   python dots3/bench_offload_cache.py C [steps]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
from arcstats import sparse_offload as so

C = int(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
dev = torch.device("cuda"); torch.manual_seed(0)
NREQ, Q, TOPK, H, R, CTX = 40, 4, 2048, int(os.environ.get("BENCH_H", "32")), 4_000_000, 60_000   # 32 heads/rank (TP4)
so.CACHE_ROWS = C
off = so.LayerOffload(R, dev)
if os.environ.get("BENCH_RANDHOST"):   # random fp8 rows so outputs can be compared across variants
    g = torch.Generator(device=dev).manual_seed(1)
    for i in range(0, R, 1 << 18):
        off.host[i:i + (1 << 18)].copy_((torch.randn(min(1 << 18, R - i), so.D_ROW, device=dev, generator=g) * 0.5).to(so.FP8))
base = torch.randperm(R // CTX, device=dev)[:NREQ] * CTX                 # each request owns a CTX-row slot range
perm_ = torch.stack([torch.randperm(CTX, device=dev) for _ in range(NREQ)])
hot = perm_[:, :3000].clone()                                             # ~3k hot rows per request
q = torch.randn(NREQ * Q, H, 576, device=dev, dtype=torch.bfloat16)
ts, hits, after = [], [], []
for step in range(STEPS):
    if step:   # drift: 5% of each hot set replaced per step
        k = torch.randint(0, 3000, (NREQ, 150), device=dev)
        if os.environ.get("BENCH_DISTINCT"):   # swap in rows from outside the hot set: hot rows stay distinct
            k = torch.rand(NREQ, 3000, device=dev).argsort(-1)[:, :150]
            k2 = 3000 + torch.rand(NREQ, CTX - 3000, device=dev).argsort(-1)[:, :150]
            a_, b_ = perm_.gather(1, k), perm_.gather(1, k2)
            perm_.scatter_(1, k, b_); perm_.scatter_(1, k2, a_)
            hot = perm_[:, :3000].clone()
        else:
            hot.scatter_(1, k, torch.randint(0, CTX, (NREQ, 150), device=dev))
    if os.environ.get("BENCH_DISTINCT"):   # distinct rows per token, like a real top-k
        pick = torch.rand(NREQ, Q, 3000, device=dev).argsort(-1)[..., :TOPK]
    else:
        pick = torch.randint(0, 3000, (NREQ, 1 if os.environ.get("BENCH_SAME") else Q, TOPK), device=dev).expand(NREQ, Q, TOPK)
    idx = (base[:, None, None] + torch.gather(hot[:, None, :].expand(-1, Q, -1), 2, pick)).reshape(-1, TOPK).to(torch.int32)
    f = idx.reshape(-1).long(); hits.append(float(off._cached(f).float().mean()))
    torch.cuda.synchronize(); t0 = time.time()
    o = off.attend(q, idx, 0.04, 1.0, H, ql=Q)
    torch.cuda.synchronize(); ts.append(time.time() - t0)
    after.append(float(off._cached(f).float().mean()))   # rows still uncached after the fill = host reads in attn
if os.environ.get("BENCH_RANDHOST"):   # dense reference from the host rows for 16 tokens of the last step
    tt = torch.arange(0, NREQ * Q, NREQ * Q // 16, device=dev)
    K = off.host[idx[tt].long().cpu()].to(dev).float()                 # [16, TOPK, 576]
    sc = torch.einsum("thd,tkd->thk", q[tt].float(), K) * 0.04
    ref = torch.einsum("thk,tkd->thd", sc.softmax(-1), K[..., :512])
    err = ((o[tt].float() - ref).abs().max() / ref.abs().max()).item()
    print(f"ref check: max rel err {err:.2e}")
    if so._PACKED:   # same final state through the per-token path
        so._PACKED = False
        o2 = off.attend(q, idx, 0.04, 1.0, H, ql=Q)
        e2 = ((o2[tt].float() - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2)))
        e1 = ((o[tt].float() - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2)))
        print("per-token err packed", [round(float(x), 3) for x in e1], "split", [round(float(x), 3) for x in e2])
        so._PACKED = True
    if so._RESOLVE:
        rb = so._RS[dev] if dev in so._RS else next(iter(so._RS.values()))
        r = rb["res"][:idx.numel()]
        print(f"spill rows last call {int(rb['scnt'])}  entries: cache {float((r >= 0).float().mean()):.4f} "
              f"spill {float((r <= -2).float().mean()):.4f} host {float(((r == -1) & (idx.reshape(-1) >= 0)).float().mean()):.4f}")
w = STEPS // 2
print(f"C={C} prefill={so._PREFILL}: {1e3 * sum(ts[w:]) / len(ts[w:]):.2f} ms/call  hit_before={sum(hits[w:]) / len(hits[w:]):.3f}  hit_after_fill={sum(after[w:]) / len(after[w:]):.4f}  uniq/call={int(torch.unique(f).numel())}  resolve={so._RESOLVE} checksum={float(o.float().abs().sum()):.6e}")
