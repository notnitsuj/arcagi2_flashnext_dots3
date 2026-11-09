"""LM head at decode batch sizes on one GPU (TP4 shard: 38016 x 5120): bf16 F.linear vs block-FP8 W8A8 (arcstats.lmhead_fp8).
  python dots3/bench_lmhead.py   -> 'lmhead M=<m> bf16 <ms> fp8 <ms> relerr <e> top1 <agree>' per M"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, HERE); import triton_sm89  # noqa
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
from arcstats.lmhead_fp8 import quantize, BLK
if os.environ.get("ARC_W8A8_CONFIG_DIR"):   # tuned tile configs (dots3/tune_w8a8.py output, device_name=ANY)
    import arcstats
    arcstats._w8a8_configs()

torch.manual_seed(0)
N, K = 38016, 5120
w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
q, s = quantize(w)


def graph_ms(f, n=10, reps=5):
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


def fp8(x):
    xq, xs = fu.per_token_group_quant_fp8(x, BLK)
    return fu.w8a8_triton_block_scaled_mm(xq, q, xs, s, [BLK, BLK], output_dtype=x.dtype)


for M in (40, 160):
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    a, b = F.linear(x, w), fp8(x)
    err = float((a.float() - b.float()).norm() / a.float().norm())
    top1 = float((a.argmax(1) == b.argmax(1)).float().mean())
    print(f"lmhead M={M} bf16 {graph_ms(lambda: F.linear(x, w)):.3f} ms fp8 {graph_ms(lambda: fp8(x)):.3f} ms "
          f"relerr {err:.2e} top1 {top1:.3f}", flush=True)
