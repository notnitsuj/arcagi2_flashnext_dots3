"""ARC_MLA_FP8 check + timing: synthetic block-FP8 kv_b_proj (Dots3 SWA: 16 heads/rank, P=192, V=128, L=1024; DSA:
32 heads, P=128, V=128, L=512), vLLM-style bf16 W_UK_T / W_UV, conversion round trip, bmm_fp8 vs torch.bmm.
  python dots3/bench_mla_fp8.py
Prints 'mlafp8 <layer> B=<b> <uk|uv> bf16 <ms> fp8 <ms> maxdiff <x>'."""
import os, sys, time, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
import torch
from arcstats import mla_fp8

dev = torch.device("cuda"); torch.manual_seed(0)
CFGS = [(64, 64, 4, 3), (128, 64, 4, 3), (128, 128, 8, 3), (64, 128, 4, 2), (32, 128, 4, 3), (128, 64, 8, 4)]


def timeit(f, n=4):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / (5 * n) * 1e3


NL = 12   # distinct layers cycled per timed call: weights come from DRAM, not L2 (as in serving)
for name, N, P, V, L in (("swa", 16, 192, 128, 1024), ("dsa", 32, 128, 128, 512)):
    rows = N * (P + V)
    code = (torch.randn(rows, L, device=dev) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    si = torch.rand((rows + 127) // 128, (L + 127) // 128, device=dev) * 1e-3 + 1e-4
    w = (code.float() * si.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :L]).to(torch.bfloat16)
    kvb = w.t().contiguous().view(L, N, P + V)               # vLLM: kv_b_proj_weight [L, N, P + V]
    W_UK, W_UV = kvb.split([P, V], dim=-1)
    layer = types.SimpleNamespace(kv_b_proj=types.SimpleNamespace(weight_scale_inv=si),
                                  W_UK_T=W_UK.permute(1, 2, 0).contiguous(), W_UV=W_UV.transpose(0, 1).contiguous())
    ref_uk, ref_uv = layer.W_UK_T.clone(), layer.W_UV.clone()
    ok = mla_fp8._convert(layer)
    print(f"mlafp8 {name} convert ok={ok}", flush=True)
    uks = [ref_uk.clone() for _ in range(NL)]; uvs = [ref_uv.clone() for _ in range(NL)]
    f8k = [tuple(t.clone() if torch.is_tensor(t) else t for t in layer._arc_uk) for _ in range(NL)]
    f8v = [tuple(t.clone() if torch.is_tensor(t) else t for t in layer._arc_uv) for _ in range(NL)]
    for B in (48, 192, 384, 1024):
        q = torch.randn(B, N, P, device=dev, dtype=torch.bfloat16).transpose(0, 1)        # [N, B, P] view
        o1 = torch.empty(B, N, L, device=dev, dtype=torch.bfloat16); o2 = torch.empty_like(o1)
        mla_fp8._TorchProxy.bmm(q, layer.W_UK_T, out=o2.transpose(0, 1)); torch.bmm(q, ref_uk, out=o1.transpose(0, 1))
        f1 = lambda: [torch.bmm(q, w_, out=o1.transpose(0, 1)) for w_ in uks]
        t1 = timeit(f1) / NL
        t2 = {c: timeit(lambda: [mla_fp8.bmm_fp8(q, f_, o2.transpose(0, 1), c) for f_ in f8k]) / NL for c in CFGS}
        best = min(t2, key=t2.get); t2s = " ".join(f"{c}:{v:.4f}" for c, v in t2.items())
        d = ((o1.float() - o2.float()).abs().max() / o1.float().abs().max()).item()
        t3 = timeit(lambda: [mla_fp8.bmm_fp8(q, f_, o2.transpose(0, 1)) for f_ in f8k]) / NL
        d3 = ((o1.float() - o2.float()).abs().max() / o1.float().abs().max()).item()
        print(f"mlafp8 {name} B={B} uk bf16 {t1:.4f} fp8 best {t2[best]:.4f} {best} ms rel_maxdiff {d:.2e} default {t3:.4f} ms "
              f"diff {d3:.2e} | {t2s}", flush=True)
        x = torch.randn(B, N, L, device=dev, dtype=torch.bfloat16).transpose(0, 1)        # [N, B, L] view
        u1 = torch.empty(B, N, V, device=dev, dtype=torch.bfloat16); u2 = torch.empty_like(u1)
        mla_fp8._TorchProxy.bmm(x, layer.W_UV, out=u2.transpose(0, 1)); torch.bmm(x, ref_uv, out=u1.transpose(0, 1))
        t1 = timeit(lambda: [torch.bmm(x, w_, out=u1.transpose(0, 1)) for w_ in uvs]) / NL
        t2 = {c: timeit(lambda: [mla_fp8.bmm_fp8(x, f_, u2.transpose(0, 1), c) for f_ in f8v]) / NL for c in CFGS}
        best = min(t2, key=t2.get); t2s = " ".join(f"{c}:{v:.4f}" for c, v in t2.items())
        d = ((u1.float() - u2.float()).abs().max() / u1.float().abs().max()).item()
        t3 = timeit(lambda: [mla_fp8.bmm_fp8(x, f_, u2.transpose(0, 1)) for f_ in f8v]) / NL
        d3 = ((u1.float() - u2.float()).abs().max() / u1.float().abs().max()).item()
        print(f"mlafp8 {name} B={B} uv bf16 {t1:.4f} fp8 best {t2[best]:.4f} {best} ms rel_maxdiff {d:.2e} default {t3:.4f} ms "
              f"diff {d3:.2e} | {t2s}", flush=True)
