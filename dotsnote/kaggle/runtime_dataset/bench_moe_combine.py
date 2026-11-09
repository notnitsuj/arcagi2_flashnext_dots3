"""One low-bit MoE layer (144 experts, top-8, per-rank TP4 shapes) at decode M: separate top-k sum vs fused fp32-atomic
combine in the down GEMM (ARC_LOWBIT_COMBINE). CUDA-graph timed.  ARC_TRITON_SM89=1 [ARC_EMULATE_L4_OCC=1]
  python dots3/bench_moe_combine.py [M]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
import arcstats.lowbit_moe as lm
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 160
H, I, TOPK, E = 5120, 384, 8, 144
bits = [3 if e < 96 else 2 for e in range(E)]
gu = lm.LowbitProj(bits, [torch.randint(0, 2 ** b, (2 * I, H), dtype=torch.uint8) for b in bits],
                   [torch.rand(2 * I, H // 128) * .01 for _ in bits], [torch.randint(0, 4, (2 * I, H // 128), dtype=torch.uint8) for _ in bits], dev)
dn = lm.LowbitProj(bits, [torch.randint(0, 2 ** b, (H, I), dtype=torch.uint8) for b in bits],
                   [torch.rand(H, I // 128) * .01 for _ in bits], [torch.randint(0, 4, (H, I // 128), dtype=torch.uint8) for _ in bits], dev)
moe = lm.LowbitMoE(gu, dn, E)
x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
logits = torch.randn(M, E, device=dev)
w, ids = torch.topk(logits.softmax(-1), TOPK)
w = (w / w.sum(-1, keepdim=True)).float()
ids = ids.int()


def timeit(f, n=10, reps=5):
    f(); torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        f()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / (n * reps) * 1e3


res = {}
for c in (False, True):
    lm._COMBINE = c
    o = moe(x, w, ids).float()
    res[c] = (o, timeit(lambda: moe(x, w, ids)))
err = float((res[True][0] - res[False][0]).abs().max() / res[False][0].abs().max())
print(f"M={M} sum {res[False][1]:.3f} ms  combine {res[True][1]:.3f} ms  rel err {err:.1e}")
