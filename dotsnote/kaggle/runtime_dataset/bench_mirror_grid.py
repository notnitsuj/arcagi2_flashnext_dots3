"""Cost of a background (MIRROR=2) row-fill copy on the co-running decode kernels, per copy grid
(ARC_OFFLOAD_MIRROR_GRID).

Main stream: a decode-like loop of serving kernels at T=256 (MTP-3 verify, TP4 rank shapes): low-bit MoE layer (random
3/2-bit codes, skewed top-8 routing, serving tiles 16/64/4/2 and 16/128/4/2) and block-FP8 W8A8 GEMMs (w8a8_v2 tiles of
the v22 table), CUDA-graph captured. Side stream (priority -1): csrc fill_copy of M random compact host rows
(ARC_OFFLOAD_ROW_FMT, default e2m2) from a cudaHostRegister'ed buffer into cache lines, as the mirror copy does.
For each grid (CTAs x rows per warp): main loop time with the copy running vs alone, copy time with vs without the main
loop, and the extra main-loop time per copied row. Rounds interleave the grids (L4 clocks drift).
  ARC_LOWBIT_PAIRS=1 python dots3/bench_mirror_grid.py [M=50000] [grids=58x2,32x4,16x4,8x4,4x4]
Prints 'mg ...' lines; the last one per grid: 'MG grid=<g>x<r> main_alone <ms> main_with <ms> (+x%) copy_alone <ms>
copy_with <ms> extra_us_per_krow <us>'."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "arcstats_pkg"))
os.environ.setdefault("ARC_OFFLOAD_ROW_FMT", "e2m2")
os.environ.setdefault("ARC_LOWBIT_PAIRS", "1")
import torch
import arcstats.lowbit_moe as lm
from arcstats import offload_cuda as oc, offload_rowfmt as rf, sparse_offload as so, w8a8_v2
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
torch.manual_seed(0)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
M_ROWS = int(args[0]) if args else 50000
NSM = torch.cuda.get_device_properties(dev).multi_processor_count
GRIDS = [tuple(int(v) for v in g.split("x")) for g in (args[1] if len(args) > 1 else f"{NSM}x2,32x4,16x4,8x4,4x4").split(",")]
E, N3, TOPK, T = 144, 96, 8, 256


def log(*a):
    print("mg", *a, flush=True)


# ---- main-stream work: MoE layer + dense W8A8 GEMMs of one decoder layer (TP4 rank shapes)
bits = [3 if e < N3 else 2 for e in range(E)]
ps = []
for n_out, K in ((768, 5120), (5120, 384)):
    codes = [torch.randint(0, 2 ** b, (n_out, K), dtype=torch.uint8) for b in bits]
    sc = [torch.rand(n_out, K // 128) * .004 + .001 for _ in range(E)]
    zp = [torch.randint(0, 2 ** b, (n_out, K // 128), dtype=torch.uint8) for b in bits]
    ps.append(lm.LowbitProj(bits, codes, sc, zp, dev))
moe = lm.LowbitMoE(ps[0], ps[1], E)
lm._TUNE = {"gate_up": [(T, [16, 64, 4, 2])], "down": [(T, [16, 128, 4, 2])]}
g = torch.Generator().manual_seed(1)
pop = torch.randn(E, generator=g) * 0.7
sc_ = pop[None] + (torch.randn(T // 4, E, generator=g) * 0.8).repeat_interleave(4, 0) + torch.randn(T, E, generator=g)
tid = torch.topk(sc_, TOPK, -1).indices.int().to(dev)
tw = (torch.rand(T, TOPK, generator=g) + 0.1).to(dev).float()
x = (torch.randn(T, 5120, device=dev) * 0.5).bfloat16()
dense = []   # (N, K, cfg) per v22 table at M=256 (q/kv proj, o proj, shared expert, ...)
for N, K, cfg in ((2112, 5120, dict(BM=64, BN=64, num_warps=4, num_stages=3)),
                  (5120, 1024, dict(BM=64, BN=64, num_warps=4, num_stages=3)),
                  (4096, 1024, dict(BM=64, BN=64, num_warps=4, num_stages=3)),
                  (768, 5120, dict(BM=64, BN=64, num_warps=4, num_stages=4))):
    A = torch.randn(T, K, device=dev).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    As = torch.rand(T, K // 128, device=dev) + 0.5
    Bs = torch.rand((N + 127) // 128, K // 128, device=dev) + 0.5
    dense.append((A, B, As, Bs, cfg))


def layer():
    moe(x, tw, tid)
    for A, B, As, Bs, cfg in dense:
        w8a8_v2.v2_mm(A, B, As, Bs, cfg)


NL = 6
layer()
torch.cuda.synchronize()
gm = torch.cuda.CUDAGraph()
with torch.cuda.graph(gm):
    for _ in range(NL):
        layer()

# ---- side copy: M random compact host rows -> cache lines (no flags), as the MIRROR=2 S-layer copy
R = 4 << 20
host = so._registered_host((R, rf.RB), torch.uint8 if rf.ON else so.FP8)
hv = host.view(torch.uint8)
for c in range(0, R, 1 << 18):
    blk = torch.randint(0, 256, (min(1 << 18, R - c), rf.RB), dtype=torch.uint8)
    if rf.ON:
        blk[:, rf.SOFF:rf.ROFF] = torch.randint(-10, 2, (blk.shape[0], rf.ROFF - rf.SOFF), dtype=torch.int8).view(torch.uint8)
    hv[c:c + blk.shape[0]] = blk
C = 1 << 18
cache = torch.zeros((C, 576), dtype=so.FP8, device=dev)
spill = torch.zeros((16, 576), dtype=so.FP8, device=dev)
sets = [torch.randperm(R)[:M_ROWS].to(dev) for _ in range(4)]
mcnt = torch.tensor([M_ROWS], dtype=torch.int32, device=dev)
cpd = torch.randperm(C, device=dev)[:M_ROWS].long()
side = torch.cuda.Stream(dev, priority=-1)
log(f"device {torch.cuda.get_device_name(dev)} nsm {NSM} rows {M_ROWS} x {rf.RB} B ({os.environ['ARC_OFFLOAD_ROW_FMT']}), "
    f"main = {NL} x (MoE T{T} + {len(dense)} W8A8)")


def run(grid, with_main, with_copy, k):
    e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda.synchronize()
    e[0].record()
    if with_copy:
        side.wait_event(e[0])
        e[2].record(side)
        oc.fill_copy(sets[k % len(sets)], mcnt, cpd, host, cache, spill, None, None, grid[0] + 1000 * grid[1], side,
                     hfmt=rf.FMT)
        e[3].record(side)
    if with_main:
        gm.replay()
    e[1].record()
    torch.cuda.synchronize()
    return e[0].elapsed_time(e[1]) if with_main else None, e[2].elapsed_time(e[3]) if with_copy else None


for _ in range(3):
    run(GRIDS[0], True, True, 0)
res = {gr: dict(ma=[], mw=[], ca=[], cw=[]) for gr in GRIDS}
for r in range(int(os.environ.get("MG_ROUNDS", "9"))):
    for gr in GRIDS:
        res[gr]["ma"].append(run(gr, True, False, r)[0])
        res[gr]["ca"].append(run(gr, False, True, r)[1])
        t, c = run(gr, True, True, r)
        res[gr]["mw"].append(t)
        res[gr]["cw"].append(c)
med = lambda v: sorted(v)[len(v) // 2]
for gr in GRIDS:
    d = {k: med(v) for k, v in res[gr].items()}
    # main-loop time added per 1000 copied rows, counted only while the copy overlaps the main loop
    ov = min(d["cw"], d["mw"])
    rows_ov = M_ROWS * ov / d["cw"]
    extra = (d["mw"] - d["ma"]) * 1e3 / max(1.0, rows_ov / 1e3)
    print(f"MG grid={gr[0]}x{gr[1]} main_alone {d['ma']:.3f} main_with {d['mw']:.3f} ({100 * (d['mw'] / d['ma'] - 1):+.1f}%) "
          f"copy_alone {d['ca']:.3f} copy_with {d['cw']:.3f} rows/us {M_ROWS / d['ca'] / 1e3:.1f} "
          f"extra_us_per_krow {extra:.1f}", flush=True)
