"""Bit-exactness and speed of arcstats.gmerge (ARC_GATE_MERGE): g_proj rows appended (after zero padding to a 128-row
scale block) to fused_qkv_a_proj, one W8A8 block-FP8 GEMM with the fused shape's tile config, vs the two separate
GEMMs (each with its own activation quantization), for the SWA (2112 + 64) and DSA (1600 + 128) layer shapes.
  python dots3/test_gmerge.py [--sm89] [M,M,...]      (--sm89: L4 code generation on an H100)
Last line (run_l4 gate_merge_gate): GMERGE_GATE exact <bool> sep <us> merged <us> speedup <x> ok <bool>
  (ok = bit-exact at every M and the step-weighted merged path >= 5% faster at the decode M's: GMERGE_DECODE_M,
   default 256, and M / 4 for the MTP drafts)
Tile configs: w8a8_configs.json next to this script or in kaggle/l4run/ (or files in ARC_W8A8_CONFIG_DIR).
"""
import glob, json, os, sys

if "--sm89" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_sm89  # noqa: F401

import torch
from vllm.model_executor.layers.quantization.utils import fp8_utils as fu

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = {}
for p in (os.path.join(HERE, "w8a8_configs.json"), os.path.join(HERE, "..", "kaggle", "l4run", "w8a8_configs.json")):
    if os.path.exists(p):
        for fn, c in json.load(open(p)).items():
            b = fn.split(",")
            TAB[(int(b[0][2:]), int(b[1][2:]))] = {int(k): v for k, v in c.items()}
        break
for f in glob.glob(os.path.join(os.environ.get("ARC_W8A8_CONFIG_DIR", "/nonexistent"), "N=*,K=*,*.json")):
    b = os.path.basename(f).split(",")
    TAB[(int(b[0][2:]), int(b[1][2:]))] = {int(k): v for k, v in json.load(open(f)).items()}
ALIAS = {}
_orig = fu.get_w8a8_block_fp8_configs
fu.get_w8a8_block_fp8_configs = lambda N, K, bn, bk: TAB.get(ALIAS.get((N, K), (N, K))) or _orig(N, K, bn, bk)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MS = [int(m) for m in args[0].split(",")] if args else [1, 7, 48, 64, 160, 192, 256, 320, 1024]
DEC = int(os.environ.get("GMERGE_DECODE_M", "256"))   # main-model decode batch (64 seqs x MTP-4); drafts run M = seqs
WEIGHT = {(2112, DEC): 34, (1600, DEC): 13, (2112, DEC // 4): 2}   # calls per decode step (SWA + MTP, DSA, drafts 2-3)
torch.manual_seed(0)


def timeit(f, it=200):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            f()
    ts = []
    for _ in range(it // 20):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / 20)
    ts.sort()
    return ts[len(ts) // 2]


bad = 0
t_sep = t_mrg = 0.0
for nq, ng, K in ((2112, 64, 5120), (1600, 128, 5120)):
    goff = -(-nq // 128) * 128
    ALIAS[(goff + ng, K)] = (nq, K)
    wq = (torch.randn(nq, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    wg = (torch.randn(ng, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sq = torch.rand(-(-nq // 128), K // 128, device="cuda") * 1e-2
    sg = torch.rand(1, K // 128, device="cuda") * 1e-2
    w = torch.zeros(goff + ng, K, dtype=wq.dtype, device="cuda")
    w[:nq] = wq
    w[goff:] = wg
    s = torch.cat([sq, sg])
    for M in MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 3

        def sep():
            xq, xs = fu.per_token_group_quant_fp8(x, 128)
            yq = fu.w8a8_triton_block_scaled_mm(xq, wq, xs, sq, [128, 128], output_dtype=torch.bfloat16)
            xq2, xs2 = fu.per_token_group_quant_fp8(x, 128)
            yg = fu.w8a8_triton_block_scaled_mm(xq2, wg, xs2, sg, [128, 128], output_dtype=torch.bfloat16)
            return yq, yg

        def mrg():
            xq, xs = fu.per_token_group_quant_fp8(x, 128)
            return fu.w8a8_triton_block_scaled_mm(xq, w, xs, s, [128, 128], output_dtype=torch.bfloat16)
        yq, yg = sep()
        ym = mrg()
        ok = torch.equal(ym[:, :nq], yq) and torch.equal(ym[:, goff:], yg)
        bad += not ok
        line = f"gmerge nq={nq} ng={ng} M={M}: qkv {'==' if torch.equal(ym[:, :nq], yq) else '!='} gate " \
               f"{'==' if torch.equal(ym[:, goff:], yg) else '!= (max diff %.3g)' % (ym[:, goff:] - yg).abs().max()}"
        if (nq, M) in WEIGHT:
            a, b = timeit(sep), timeit(mrg)
            t_sep += a * WEIGHT[(nq, M)]
            t_mrg += b * WEIGHT[(nq, M)]
            line += f"  sep {a:.1f} us merged {b:.1f} us"
        print(line, flush=True)
sp = t_sep / max(t_mrg, 1e-9)
print(f"GMERGE_GATE exact {bad == 0} sep {t_sep:.0f} merged {t_mrg:.0f} speedup {sp:.3f} ok {bad == 0 and sp >= 1.05}")
