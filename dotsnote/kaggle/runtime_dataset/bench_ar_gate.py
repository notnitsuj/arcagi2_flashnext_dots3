"""ARC_OFFLOAD_AR_GATE bench: does gating the background row copy off the all-reduce
windows shorten an all-reduce-like host transfer that overlaps it?

Main stream, repeated: [gate word := 1] AR-like kernel [gate word := 0] then a compute gap (bf16 matmuls, ~0.5 ms).
The AR-like kernel moves an all-reduce's host traffic at T=256 (TP4 rank, 5120 hidden): 32 CTAs x 512 threads
store BYTES to a registered host buffer, then load BYTES from another one (the q8/int6 host AR moves ~1.9 MB each way
per call). Side stream (priority -1): csrc fill_copy of M random compact host rows into cache lines, as the MIRROR=2
copy, ungated vs gated (offload_fill_copy_launch_gated). Reports the median AR-like time alone / with the copy /
with the gated copy, the copy duration, and checks the copied rows are identical in both modes.
  python dots3/bench_ar_gate.py [M=50000] [grid=16x4] [BYTES=1900000] [ITERS=40]   ('ag ...' / 'AG ...' lines)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "arcstats_pkg"))
os.environ.setdefault("ARC_OFFLOAD_ROW_FMT", "e2m2")
import torch
import triton
import triton.language as tl
from arcstats import offload_cuda as oc, offload_rowfmt as rf, sparse_offload as so

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
a = sys.argv[1:]
M = int(a[0]) if len(a) > 0 else 50000
GR = tuple(int(v) for v in (a[1] if len(a) > 1 else "16x4").split("x"))
BYTES = int(a[2]) if len(a) > 2 else 1_900_000
ITERS = int(a[3]) if len(a) > 3 else 40


@triton.jit
def _arlike(x_ptr, hout_ptr, hin_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """phase 1: device -> host (n int32 words); phase 2: host -> device (n words); per-CTA strided chunks"""
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for s in range(pid * BLOCK, n, npg * BLOCK):
        o = s + tl.arange(0, BLOCK)
        m = o < n
        tl.store(hout_ptr + o, tl.load(x_ptr + o, mask=m), mask=m)
    tl.debug_barrier()
    for s in range(pid * BLOCK, n, npg * BLOCK):
        o = s + tl.arange(0, BLOCK)
        m = o < n
        tl.store(y_ptr + o, tl.load(hin_ptr + o, mask=m, cache_modifier=".cv"), mask=m)


def log(*x):
    print("ag", *x, flush=True)


nw = BYTES // 4
hout = so._registered_host((nw,), torch.int32)
hin = so._registered_host((nw,), torch.int32)
hin.copy_(torch.randint(0, 1 << 30, (nw,), dtype=torch.int32))
x = torch.randint(0, 1 << 30, (nw,), dtype=torch.int32, device=dev)
y = torch.empty_like(x)
A = torch.randn(256, 5120, device=dev, dtype=torch.bfloat16)
B = torch.randn(5120, 3072, device=dev, dtype=torch.bfloat16)
gate = torch.zeros(1, dtype=torch.int32, device=dev)

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
rows = torch.randperm(R)[:M].to(dev)
mcnt = torch.tensor([M], dtype=torch.int32, device=dev)
cpd = torch.randperm(C, device=dev)[:M].long()
side = torch.cuda.Stream(dev, priority=-1)
assert oc.gate_available(), "offload_attn.so without offload_fill_copy_launch_gated"
log(f"device {torch.cuda.get_device_name(dev)} AR-like {BYTES / 1e6:.2f} MB out + in, copy {M} x {rf.RB} B rows, "
    f"grid {GR[0]}x{GR[1]}, {ITERS} AR+gap iterations")


def run(mode):
    """mode: 0 = no copy, 1 = copy ungated, 2 = copy gated -> (median AR us, AR us while the copy runs, copy ms)"""
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    c0, c1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = torch.cuda.Event(enable_timing=True)
    t0.record()
    if mode:
        side.wait_event(t0)
        c0.record(side)
        oc.fill_copy(rows, mcnt, cpd, host, cache, spill, None, None, GR[0] + 1000 * GR[1], side, hfmt=rf.FMT,
                     gate=gate if mode == 2 else None)
        c1.record(side)
    for i in range(ITERS):
        gate.fill_(1)
        ev[i][0].record()
        _arlike[(32,)](x, hout, hin, y, nw, BLOCK=2048, num_warps=16)
        ev[i][1].record()
        gate.fill_(0)
        for _ in range(3):
            A @ B
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) * 1e3 for s, e in ev]
    if not mode:
        return ts, None, None
    cs, ce = t0.elapsed_time(c0), t0.elapsed_time(c1)
    during = [i for i, (s, e) in enumerate(ev) if t0.elapsed_time(s) < ce and t0.elapsed_time(e) > cs]
    return ts, during, c0.elapsed_time(c1)


for _ in range(2):
    run(0); run(1); run(2)
ref = None
res = {0: [], 1: [], 2: []}
for r in range(5):
    for mode in (0, 1, 2):
        cache.zero_()
        res[mode].append(run(mode))
        if mode:
            snap = cache[cpd].view(torch.uint8).clone()
            if ref is None:
                ref = snap
            if not torch.equal(snap, ref):
                bad = (snap.view(torch.uint8) != ref.view(torch.uint8)).any(1)
                log(f"MISMATCH mode {mode} round {r}: {int(bad.sum())} rows differ")
                raise SystemExit(1)
med = lambda v: sorted(v)[len(v) // 2]
# index-matched: the AR-like calls that overlapped the copy vs the SAME iteration indices of the no-copy run of that
# round (the first calls after a sync run at a different speed than the steady state on L4)
out = {}
for mode, name in ((1, "ungated"), (2, "gated")):
    w, a_, cp = [], [], []
    for r in range(len(res[mode])):
        ts, during, c = res[mode][r]
        base = res[0][r][0]
        if during:
            w.append(sum(ts[i] for i in during) / len(during))
            a_.append(sum(base[i] for i in during) / len(during))
        cp.append(c)
    out[name] = (med(w), med(a_), med(cp))
    log(f"{name}: AR-like overlapping the copy {out[name][0]:.0f} us vs same calls without a copy {out[name][1]:.0f} us "
        f"({100 * (out[name][0] / out[name][1] - 1):+.0f}%), copy {out[name][2]:.2f} ms")
print(f"AG ungated_us {out['ungated'][0]:.0f} (alone {out['ungated'][1]:.0f}) gated_us {out['gated'][0]:.0f} "
      f"(alone {out['gated'][1]:.0f}) copy_ms_ungated {out['ungated'][2]:.2f} copy_ms_gated {out['gated'][2]:.2f} "
      f"rows_identical True", flush=True)
