"""CUDA offload split kernel (arcstats/offload_cuda.py) vs the Triton split kernel: correctness (resolved locations
mixing cache lines, spill rows, host rows and invalid entries) and CUDA-graph kernel time at production shapes.
  python dots3/bench_offload_cuda.py [NREQ=48] [QL=4]
Prints 'oc check ...' and 'oc <variant> T=<T> <ms> ms'."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
import torch
import triton.language as tl
from arcstats import sparse_offload as so
from arcstats import offload_cuda as oc

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 48
QL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOPK, H, R, CTX = 2048, 32, 4_000_000, 60_000
T = NREQ * QL
print(f"oc T={T} H={H} topk={TOPK} on {torch.cuda.get_device_name()} cuda_lib={oc.available()}", flush=True)
so.CACHE_ROWS, so._WAYS, so._RESOLVE = 131072, 8, True
off = so.LayerOffload(R, dev)
base = torch.randperm(R // CTX, device=dev)[:NREQ] * CTX
hot = torch.stack([torch.randperm(CTX, device=dev)[:2400] for _ in range(NREQ)])   # ~2.4k distinct rows per request
pick = torch.rand(NREQ, QL, 2400, device=dev).argsort(-1)[..., :TOPK]
idx = (base[:, None, None] + torch.gather(hot[:, None, :].expand(-1, QL, -1), 2, pick)).reshape(-1, TOPK).int()
q = torch.randn(T, H, 576, device=dev, dtype=torch.bfloat16)
off.attend(q, idx, 0.04, 1.0, H, ql=QL)
rb = off._sa_prefill(idx.reshape(-1))      # all hits now: the resolved locations the kernels read
SC = 0.04 * 1.4426950408889634
# random FP8 rows wherever the kernels read (cache lines, the first host rows, spill rows)
off.cache.copy_((torch.randn(off.cache.shape, device=dev) * 0.5).to(so.FP8))
off.host[: 4096].copy_((torch.randn(4096, 576) * 0.5).to(so.FP8))


def triton_split(res, idx_, spill, split, ws, out, bh=32, warps=8, st=2, bn=32):
    HB = 1
    so._sparse_mla_offload_split_kernel[(T, HB, split)](
        q, off.host, off.cache, off.tag, idx_, ws[0], ws[1], ws[2], off.C, off.R,
        q.stride(0), q.stride(1), idx_.stride(0), H, SC, res, spill,
        TOPK=TOPK, SPLIT=split, BLOCK_H=bh, BLOCK_N=bn, HP=32, DN=so.D_NOPE, DR=so.D_ROPE,
        W=off.W, ROWMAJOR=2, RES=1, DVS=1, CD=tl.float16, num_warps=warps, num_stages=st)
    so._combine_kernel[(T, HB)](ws[0], ws[1], ws[2], out, out.stride(0), out.stride(1), H, 1.0,
                                SPLIT=split, BLOCK_H=32, HP=32, DN=so.D_NOPE, num_warps=4)


def cuda_split(res, idx_, spill, split, ws, out, stages=3):
    oc.split_partials(q, off.host, off.cache, spill, idx_, res, ws, split, SC, off.R, stages=stages)
    so._combine_kernel[(T, 1)](ws[0], ws[1], ws[2], out, out.stride(0), out.stride(1), H, 1.0,
                               SPLIT=split, BLOCK_H=32, HP=32, DN=so.D_NOPE, num_warps=4)


def ws_for(split):
    n = T * split * 32
    return (torch.empty(n, device=dev), torch.empty(n, device=dev), torch.empty(n * so.D_NOPE, device=dev))


# ---- correctness: production resolution, then a mixed one (spill rows, host rows, invalid entries)
spill = torch.empty((so._SPILL, 576), dtype=so.FP8, device=dev)   # production entries may already use any spill row
spill.copy_((torch.randn(spill.shape, device=dev) * 0.5).to(so.FP8))
res_mix = rb["res"].clone()
u = torch.rand(res_mix.shape, device=dev)
res_mix = torch.where(u < 0.1, -2 - torch.randint(0, 4096, res_mix.shape, device=dev, dtype=torch.int32), res_mix)
idx_mix = idx.clone()
hostsel = (u >= 0.1) & (u < 0.13)
res_mix = torch.where(hostsel, torch.full_like(res_mix, -1), res_mix)
idx_mix.view(-1)[hostsel] = torch.randint(0, 4096, (int(hostsel.sum()),), device=dev, dtype=torch.int32)
idx_mix.view(-1)[(u >= 0.13) & (u < 0.15)] = -1
idx_mix[5, :] = -1                               # a token with no valid rows
for name, res, idx_, sp in (("prod", rb["res"], idx, rb["spill"]), ("mixed", res_mix, idx_mix, spill)):
    for split in (2, 4, 8):
        o1 = torch.empty(T, H, so.D_NOPE, device=dev, dtype=torch.bfloat16)
        o2 = torch.empty_like(o1)
        triton_split(res, idx_, sp, split, ws_for(split), o1)
        for stages in (2, 3, 12, 13):
            cuda_split(res, idx_, sp, split, ws_for(split), o2, stages)
            torch.cuda.synchronize()
            ok = ~(o1.float().isnan().any(-1).any(-1))
            d = (o1.float() - o2.float()).abs()[ok]
            print(f"oc check {name} split={split} st={stages} max_abs {d.max().item():.3e} "
                  f"rel {(d.norm() / o1.float()[ok].norm()).item():.3e} |o| {o1.float()[ok].abs().mean().item():.3e} "
                  f"nan tokens triton {int((~ok).sum())} cuda {int(o2.float().isnan().any(-1).any(-1).sum())}", flush=True)


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


out = torch.empty(T, H, so.D_NOPE, device=dev, dtype=torch.bfloat16)
ref = torch.empty_like(out)
triton_split(rb['res'], idx, rb['spill'], 4, ws_for(4), ref)
for cfg in (2, 3, 12, 13):   # split 1: normalized bf16 written by the kernel (no partials, no combine)
    f = lambda: oc.split_partials(q, off.host, off.cache, rb['spill'], idx, rb['res'], None, 1, SC, off.R, cfg, out=out)
    f(); torch.cuda.synchronize()
    d = (ref.float() - out.float()).abs()
    print(f"oc cuda_direct split=1 st={cfg} T={T} {timeit(f):.3f} ms  rel {(d.norm() / ref.float().norm()).item():.2e}",
          flush=True)
for split in (2, 4, 8):
    ws = ws_for(split)
    for bh, w, st in ((32, 8, 2), (16, 4, 2)):
        if bh == 16:
            continue
        print(f"oc triton split={split} bh{bh} w{w} st{st} T={T} "
              f"{timeit(lambda: triton_split(rb['res'], idx, rb['spill'], split, ws, out, bh, w, st)):.3f} ms", flush=True)
    for stages in (2, 3, 12, 13):
        print(f"oc cuda split={split} st={stages} T={T} "
              f"{timeit(lambda: cuda_split(rb['res'], idx, rb['spill'], split, ws, out, stages)):.3f} ms", flush=True)
        print(f"oc cuda_kernel_only split={split} st={stages} T={T} "
              f"{timeit(lambda: oc.split_partials(q, off.host, off.cache, rb['spill'], idx, rb['res'], ws, split, SC, off.R, stages)):.3f} ms",
              flush=True)

# ---- prefill form: staged rows + per-token indices into them (dsa_sm89 _forward_offload prefill chunks)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
for TP, NR in ((512, 8192), (2048, 16384)):
    rows = (torch.randn(NR, 576, device=dev) * 0.5).to(so.FP8)
    qp = torch.randn(TP, H, 576, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(TP, device=dev) + NR - TP                     # causal: token i sees rows <= its position
    sel = (torch.rand(TP, NR, device=dev) * (torch.arange(NR, device=dev)[None, :] <= pos[:, None])).topk(TOPK, -1)
    sidx = torch.where(sel.values > 0, sel.indices, -1).int().contiguous()
    o_ref, _, _ = triton_bf16_mla_sparse_interface(qp, rows.view(NR, 1, 576), sidx.view(TP, 1, -1), sm_scale=0.04,
                                                   kv_scale=1.0)
    o_c = torch.empty(TP, H, 512, device=dev, dtype=torch.bfloat16)
    for split, cfg in ((1, 3), (1, 13), (2, 3)):
        oc.attend_rows(qp, rows, sidx, 0.04, 1.0, o_c, split=split, stages=cfg)
        torch.cuda.synchronize()
        d = (o_ref[:, :H].float() - o_c.float()).abs()
        print(f"oc prefill check T={TP} split={split} cfg={cfg} max_abs {d.max().item():.3e} rel "
              f"{(d.norm() / o_ref[:, :H].float().norm()).item():.3e}", flush=True)
        print(f"oc prefill cuda T={TP} split={split} cfg={cfg} "
              f"{timeit(lambda: oc.attend_rows(qp, rows, sidx, 0.04, 1.0, o_c, split=split, stages=cfg), n=3, reps=3):.3f} ms", flush=True)
    print(f"oc prefill triton_bf16 T={TP} {timeit(lambda: triton_bf16_mla_sparse_interface(qp, rows.view(NR, 1, 576), sidx.view(TP, 1, -1), sm_scale=0.04, kv_scale=1.0), n=3, reps=3):.3f} ms", flush=True)
