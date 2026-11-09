"""Offload decode attention tile configs vs L4 occupancy: compiles each config with the current codegen (ARC_TRITON_SM89=1
for L4's), reads registers / shared memory from the compiled kernel and reports CTAs and warps per L4 SM (64K regs,
100 KB smem, 48 warps, 24 CTAs) next to the measured time here.   python dots3/tune_offload_l4occ.py [split|packed]"""
import os, sys, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
from arcstats import sparse_offload as so

L4_REGS, L4_SMEM, L4_WARPS, L4_CTAS = 65536, 100 * 1024, 48, 24
WHICH = sys.argv[1] if len(sys.argv) > 1 else "split"
dev = torch.device("cuda"); torch.manual_seed(0)
NREQ, Q, TOPK, H, R, CTX = 40, 4, 2048, 32, 4_000_000, 60_000
so.CACHE_ROWS = 131072
off = so.LayerOffload(R, dev)
base = torch.randperm(R // CTX, device=dev)[:NREQ] * CTX
hot = torch.stack([torch.randperm(CTX, device=dev)[:3000] for _ in range(NREQ)])
q = torch.randn(NREQ * Q, H, 576, device=dev, dtype=torch.bfloat16)
pick = torch.rand(NREQ, Q, 3000, device=dev).argsort(-1)[..., :TOPK]
idx = (base[:, None, None] + torch.gather(hot[:, None, :].expand(-1, Q, -1), 2, pick)).reshape(-1, TOPK).int()
so._PACKED = WHICH == "packed"
so._RESOLVE = os.environ.get("ARC_OFFLOAD_RESOLVE") == "1"   # production path (set ARC_OFFLOAD_WAYS=8 too)
kern = so._dsa_pk_kernel if so._PACKED else so._sparse_mla_offload_split_kernel


def occ(regs, smem, warps):
    thr = 32 * warps
    regs_cta = ((regs + 7) // 8 * 8) * thr
    c = min(L4_REGS // regs_cta, (L4_SMEM // max(smem + 1024, 1)) if smem else L4_CTAS, L4_WARPS // warps, L4_CTAS)
    return c, c * warps


ref = None
if so._PACKED:
    grid = [dict(_PK_BN=bn, _PK_DV=dv, _PK_WARPS=w, _PK_STAGES=st, _PK_KS=ks, _PK_BD=bd)
            for bn, dv, w, st, ks, bd in itertools.product([16, 32, 64], [128, 256, 512], [4, 8], [1, 2], [8], [64])]
else:
    BHS = [int(x) for x in os.environ.get("TUNE_BH", "16,32").split(",")]
    SPS = [int(x) for x in os.environ.get("TUNE_SPLIT", "1,2,4,8").split(",")]
    grid = [dict(_SPLIT_BH=bh, _SPLIT_BN=bn, _SPLIT_WARPS=w, _SPLIT_STAGES=st, _SPLIT=sp)
            for bh, bn, w, st, sp in itertools.product(BHS, [16, 32, 64], [4, 8], [1, 2, 3], SPS)]
for cfg in grid:
    for k, v in cfg.items():
        setattr(so, k, v)
    try:
        kern.device_caches.clear() if hasattr(kern, "device_caches") else None
        o = off.attend(q, idx, 0.04, 1.0, H, ql=Q)
        torch.cuda.synchronize()
        if ref is None:
            ref = o.float()
        err = float((o.float() - ref).abs().max() / ref.abs().max())
        for _ in range(3):
            off.attend(q, idx, 0.04, 1.0, H, ql=Q)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            off.attend(q, idx, 0.04, 1.0, H, ql=Q)
        torch.cuda.synchronize(); t = (time.time() - t0) / 10 * 1e3
        ks = [k for dc in kern.device_caches.values() for k in dc[0].values()]
        kk = ks[-1]
        c, w = occ(kk.n_regs, kk.metadata.shared, cfg.get("_SPLIT_WARPS", cfg.get("_PK_WARPS")))
        cfg["ms"] = round(t, 3)
        print(f"{cfg} regs={kk.n_regs} spills={kk.n_spills} smem={kk.metadata.shared} L4: {c} CTA/SM {w} warps/SM | "
              f"call {t:.3f} ms err {err:.1e}", flush=True)
    except Exception as ex:
        print(cfg, "FAIL", repr(ex)[:100], flush=True)
