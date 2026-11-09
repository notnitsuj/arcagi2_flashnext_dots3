"""Offload decode attention v2 (arcstats/offload_cuda2.py) vs the shipping CUDA kernel (offload_cuda.split_partials,
split 1): accuracy vs an FP32 reference and kernel time at production shapes, plus L4 microbenchmarks (tensor-core
rates, cvt rate, SM clock) and ablations of a copy of the v1 kernel.
  python dots3/bench_offload_attn2.py [NREQ=64] [QL=4] [what=all|mb|abl|v2]
Rows: C cache lines of random e4m3 rows with a per-request hot set (~2400 distinct rows over QL tokens, like real
MTP decode); 'mixed' check adds spill rows, host rows (compact e2m3 host rows when ARC_OFFLOAD_ROW_FMT is set),
invalid entries and a token without valid rows."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
import torch
from arcstats import offload_cuda as oc
from arcstats import offload_cuda2 as oc2
from arcstats import offload_rowfmt as rf

dev = torch.device("cuda"); torch.manual_seed(0)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 64
QL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
WHAT = sys.argv[3] if len(sys.argv) > 3 else "all"
TOPK, H, C, R, HOT = 2048, 32, 214000, 4_000_000, 2400
T = NREQ * QL
FP8 = torch.float8_e4m3fn
st = torch.cuda.current_stream().cuda_stream
print(f"oa2 T={T} topk={TOPK} C={C} on {torch.cuda.get_device_name()} v1={oc.available()} v2={oc2.available()} "
      f"rowfmt={rf.FMT}", flush=True)
assert oc2.available(), oc2._ERR
L = oc2.lib()


def timeit(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(3):
        e0.record()
        for _ in range(reps):
            fn()
        e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / reps)
    return min(ts)


if WHAT in ("all", "mb"):
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    o = torch.zeros(128, device=dev)
    clk = torch.zeros(2, dtype=torch.int64, device=dev)
    flop = {0: 2 * 16 * 8 * 16, 1: 2 * 16 * 8 * 16, 2: 2 * 16 * 8 * 32, 3: 0, 4: 2 * 16 * 8 * 16, 5: 0}
    names = {0: "mma f16->f32 m16n8k16", 1: "mma f16->f16 m16n8k16", 2: "mma e4m3->f32 m16n8k32",
             3: "cvt e4m3x2->f16x2 loop", 5: "same loop without cvt", 4: "mma f16->f32 + 2 cvt (v1 K ratio)"}
    for kind in (0, 1, 2, 3, 5, 4):
        for wpsm in (4, 8):   # warps per SM (CTAs of 4 warps)
            blocks, iters = nsm * wpsm // 4, 2048
            fn = lambda: L.oa2_mb_launch(kind, iters, blocks, o.data_ptr(), clk.data_ptr(), st)
            ms = timeit(fn, reps=5)
            c = clk.tolist()
            mhz = c[0] / max(c[1], 1) * 1e3
            n = blocks * 4 * iters * 8   # warp-instructions (per kind: 8 per iteration per warp)
            rate = f"{n * flop[kind] / ms / 1e9:.1f} TFLOPS" if flop[kind] else ""
            print(f"oa2 mb {names[kind]} warps/SM={wpsm}: {ms:.3f} ms {rate} "
                  f"({n / nsm / 4 / (ms * 1e-3) / (mhz * 1e6):.3f} warp-instr/clk/SMSP) clk {mhz:.0f} MHz", flush=True)

# ---- data: production-like resolved rows
cache = (torch.randn(C, 576, device=dev) * 0.6).to(FP8)
hot = torch.stack([torch.randperm(C, device=dev)[:HOT] for _ in range(NREQ)]).int()
pick = torch.rand(NREQ, QL, HOT, device=dev).argsort(-1)[..., :TOPK]
res = torch.gather(hot[:, None, :].expand(-1, QL, -1), 2, pick).reshape(T, TOPK).contiguous()
idx = torch.randint(0, R, (T, TOPK), device=dev, dtype=torch.int32)
q = (torch.randn(T, H, 576, device=dev) * 1.5).to(torch.bfloat16)
spill = (torch.randn(8192, 576, device=dev) * 0.6).to(FP8)
NH = 8192
host8 = (torch.randn(NH, 576, device=dev) * 0.6).to(FP8)     # plain fp8 host rows (fmt 0)
SMS, KVS = 0.0417, 1.3
sm2 = SMS * 1.4426950408889634 * KVS


def ref(qq, idx_, res_, host_rows):
    """fp32: rows as the kernels read them (cache line / spill / host), base-2 softmax"""
    rr = res_.view(T, TOPK).long()
    rows = torch.empty(T, TOPK, 576, device=dev)
    for t0 in range(0, T, 32):
        r = rr[t0:t0 + 32]
        x = cache[r.clamp(0, C - 1)].float()
        x = torch.where((r <= -2)[..., None], spill[(-r - 2).clamp(0, 8191)].float(), x)
        x = torch.where((r == -1)[..., None], host_rows[idx_[t0:t0 + 32].long().clamp(0, NH - 1)], x)
        rows[t0:t0 + 32] = x
    valid = (idx_ >= 0) & (idx_ < R)
    s = torch.einsum("thd,tkd->thk", qq.float(), rows) * sm2
    s = torch.where(valid[:, None, :], s, torch.tensor(-1e30, device=dev))
    p = torch.exp2(s - s.amax(-1, keepdim=True))
    p = torch.where(valid[:, None, :], p, torch.zeros((), device=dev))
    o = torch.einsum("thk,tkd->thd", p, rows[..., :512]) / p.sum(-1, keepdim=True).clamp(min=1e-30)
    return o * KVS


def v1(qq, idx_, res_, out, hostbuf, fmt):
    oc.split_partials(qq, hostbuf, cache, spill, idx_, res_.view(-1), None, 1, sm2, R, stages=2, out=out,
                      kv_scale=KVS, hfmt=fmt)


def v2(qq, idx_, res_, out, hostbuf, fmt, f8, stg=3, v3=False):
    oc2.attend(qq, hostbuf, cache, spill, idx_, res_.view(-1), sm2, R, out, KVS, hfmt=fmt, pvf8=f8, st=stg, kern=v3)


KERNS = [int(x) for x in os.environ.get("OA2_KERNS", "2,3,4").split(",")]
VARS = [(f8, stg, v3) for v3 in KERNS for f8 in (False, True)
        for stg in ((3, 4) if v3 >= 4 else (2, 3))]
vname = lambda f8, stg, v3: f"v{v3} pv{'8' if f8 else '16'} st{stg}"


if WHAT in ("all", "v2"):
    # mixed resolution: spill rows, host rows, invalid entries, one token with no valid rows
    fmt = rf.FMT
    if fmt:
        xs = torch.randn(NH, 512, device=dev) * 0.6
        code, k = rf.ref_quant(xs)
        rope = (torch.randn(NH, 64, device=dev) * 0.6).to(FP8)
        hostbuf = rf.ref_pack(code, k, rope)
        host_rows = torch.cat([rf.ref_dequant(code, k).to(FP8).float(), rope.float()], 1)
    else:
        hostbuf, host_rows = host8, host8.float()
    res_m, idx_m = res.clone(), idx.clone()
    u = torch.rand(res_m.shape, device=dev)
    res_m = torch.where(u < 0.1, -2 - torch.randint(0, 8192, res_m.shape, device=dev, dtype=torch.int32), res_m)
    hs = (u >= 0.1) & (u < 0.13)
    res_m[hs] = -1
    idx_m[hs] = torch.randint(0, NH, (int(hs.sum()),), device=dev, dtype=torch.int32)
    idx_m[(u >= 0.13) & (u < 0.15)] = -1
    idx_m[5, :] = -1
    for name, rs_, ix_ in (("prod", res, idx), ("mixed", res_m, idx_m)):
        o_ref = ref(q, ix_, rs_, host_rows)
        o1 = torch.empty(T, H, 512, device=dev, dtype=torch.bfloat16)
        v1(q, ix_, rs_, o1, hostbuf, fmt)
        torch.cuda.synchronize()
        den = o_ref.norm().item()
        e1 = (o1.float() - o_ref)
        print(f"oa2 check {name} v1: max_abs {e1.abs().max().item():.3e} rel {e1.norm().item() / den:.3e} "
              f"|o| {o_ref.abs().mean().item():.3e}", flush=True)
        for f8, stg, v3 in VARS:
            if True:
                o2 = torch.full_like(o1, float("nan"))
                v2(q, ix_, rs_, o2, hostbuf, fmt, f8, stg, v3)
                torch.cuda.synchronize()
                e2 = (o2.float() - o_ref)
                d12 = (o2.float() - o1.float())
                print(f"oa2 check {name} {vname(f8, stg, v3)}: vs ref max_abs {e2.abs().max().item():.3e} "
                      f"rel {e2.norm().item() / den:.3e} | vs v1 max_abs {d12.abs().max().item():.3e} "
                      f"mean_abs {d12.abs().mean().item():.3e} nan {int(o2.isnan().sum())}", flush=True)
    # timing (all-hit production resolution; fmt-instantiated kernels as in serving)
    o1 = torch.empty(T, H, 512, device=dev, dtype=torch.bfloat16)
    hb = hostbuf
    t1 = timeit(lambda: v1(q, idx, res, o1, hb, fmt))
    print(f"oa2 time v1 cuda_s1_c2 T={T}: {t1:.4f} ms", flush=True)
    for f8, stg, v3 in VARS:
        t2 = timeit(lambda: v2(q, idx, res, o1, hb, fmt, f8, stg, v3))
        print(f"oa2 time {vname(f8, stg, v3)} T={T}: {t2:.4f} ms ({t2 / t1:.3f}x v1)", flush=True)

if WHAT in ("all", "abl"):
    o1 = torch.empty(T, H, 512, device=dev, dtype=torch.bfloat16)
    names = {0: "base", 1: "no row loads", 2: "no idx/res loads", 3: "no loads at all", 4: "no cvt", 8: "no QK mma",
             16: "no PV mma", 24: "no mma", 28: "no mma no cvt", 32: "meta in smem"}
    for cfg in (0x42, 0x82):
        for abl in (0, 1, 2, 3, 4, 8, 16, 24, 28, 32):
            fn = lambda: L.oa2_abl_launch(q.data_ptr(), cache.data_ptr(), idx.data_ptr(), res.data_ptr(), T, R,
                                          q.stride(0), q.stride(1), idx.stride(0), TOPK, sm2, o1.data_ptr(),
                                          o1.stride(0), o1.stride(1), KVS, cfg, abl, st)
            rc = fn()
            if rc:
                print(f"oa2 abl cfg {cfg:#x} {abl} rc {rc}", flush=True)
                continue
            print(f"oa2 abl {'4w' if cfg == 0x42 else '8w'} {names[abl]:18s} T={T}: {timeit(fn):.4f} ms", flush=True)

if WHAT in ("all", "abl", "v2abl"):
    o1 = torch.empty(T, H, 512, device=dev, dtype=torch.bfloat16)
    names = {1: "no row loads", 8: "no QK mma", 16: "no PV mma", 24: "no mma"}
    for f8, v3, abl in [(f8, v3, abl) for v3 in (False, True) for f8 in (False, True) for abl in (1, 8, 16, 24)
                        if (3 if v3 else 2) in KERNS]:
        if v3 and abl == 24:
            continue
        if True:
            cfg = (2 if v3 else 3) | (16 if f8 else 0) | (32 if v3 else 0) | (abl << 8)
            fn = lambda: L.oa2_v2_launch(q.data_ptr(), host8.data_ptr(), cache.data_ptr(), spill.data_ptr(),
                                         idx.data_ptr(), res.data_ptr(), T, R, q.stride(0), q.stride(1), idx.stride(0),
                                         TOPK, sm2, o1.data_ptr(), o1.stride(0), o1.stride(1), KVS, None, None, 0, cfg, st)
            rc = fn()
            if rc:
                print(f"oa2 v2abl {cfg:#x} rc {rc}", flush=True)
                continue
            print(f"oa2 v2abl {'v3' if v3 else 'v2'} pv{'8' if f8 else '16'} {names[abl]:14s} T={T}: {timeit(fn):.4f} ms",
                  flush=True)
    # phase profile (thread 0 of every CTA, clock64): where a tile's time goes
    import numpy as np
    pbuf = np.zeros(16, dtype=np.uint64)
    PH = ["wait+bar", "issue", "QK", "bar2", "softmax", "bar3", "PV"]
    PH4 = ["wait full", "turn", "QK+PV", "gbar1", "softmax", "gbar2", "-"]
    for f8, kv in [(f8, kv) for kv in KERNS for f8 in (False, True)]:
        v3 = kv == 3
        cfg = (2 if v3 else (4 if kv >= 4 else 3)) | (16 if f8 else 0) | {2: 0, 3: 32, 4: 64, 5: 128}[kv] | (64 << 8)
        fn = lambda: L.oa2_v2_launch(q.data_ptr(), host8.data_ptr(), cache.data_ptr(), spill.data_ptr(),
                                     idx.data_ptr(), res.data_ptr(), T, R, q.stride(0), q.stride(1), idx.stride(0),
                                     TOPK, sm2, o1.data_ptr(), o1.stride(0), o1.stride(1), KVS, None, None, 0, cfg, st)
        tm = timeit(fn, reps=10)
        L.oa2_prof(pbuf.ctypes.data, 1)
        fn(); torch.cuda.synchronize()
        L.oa2_prof(pbuf.ctypes.data, 1)
        n = max(int(pbuf[7]), 1)
        tot = sum(int(x) for x in pbuf[:7])
        mhz = int(pbuf[8]) / max(int(pbuf[9]), 1) * 1e3
        ntile = TOPK // 32
        print(f"oa2 prof v{kv} pv{'8' if f8 else '16'} T={T} {tm:.4f} ms, clk {mhz:.0f} MHz, cycles/tile "
              + " ".join(f"{(PH4 if kv >= 4 else PH)[i]} {int(pbuf[i]) / n / ntile:.0f}" for i in range(7)) + f" | total {tot / n / ntile:.0f}",
              flush=True)
