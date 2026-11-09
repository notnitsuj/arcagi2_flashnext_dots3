"""Dense block-FP8 linears at decode batch sizes on the per-rank TP4 Dots3 shapes: FP8 Marlin (W8A16, what L4 runs now)
vs vLLM's Triton W8A8 block GEMM (activations quantized per 1x128 group, FP8 tensor-core dot; the checkpoint's
activation_scheme is dynamic W8A8). CUDA-graph timed. ARC_TRITON_SM89=1 (+ ARC_EMULATE_L4_OCC=1) for L4 codegen.
  python dots3/bench_fp8_gemm.py [T ...]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, \
    apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8, \
    w8a8_triton_block_scaled_mm
dev = torch.device("cuda"); torch.manual_seed(0)
SHAPES = [(2112, 5120), (6144, 1024), (5120, 4096), (768, 5120), (5120, 384), (4096, 1024)]   # (N, K) per rank
if os.environ.get("ARC_W8A8_SHAPES"):   # "N:K:count,..." per-step counts weight the totals
    SHAPES = [tuple(int(v) for v in e.split(":")) for e in os.environ["ARC_W8A8_SHAPES"].split(",")]
if os.environ.get("ARC_W8A8_CONFIG_DIR"):
    import arcstats
    arcstats._w8a8_configs()
Ts = [int(a) for a in sys.argv[1:]] or [160]


def timeit(f, n=20, reps=10):
    for _ in range(3):
        f()
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


tot = {}
for shp in SHAPES:
    N, K = shp[:2]
    cnt = shp[2] if len(shp) > 2 else 1
    w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.rand((N + 127) // 128, (K + 127) // 128, device=dev) * 0.02 + 0.005
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s.clone(), requires_grad=False)
    layer.weight_block_size = [128, 128]
    layer.input_size_per_partition, layer.output_size_per_partition = K, N
    layer.orig_dtype = torch.bfloat16
    layer.input_scale = None
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    wf = (w.float() * s.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K])
    for T in Ts:
        x = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ wf.T
        mar = lambda: apply_fp8_marlin_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale_inv,
                                              workspace=layer.workspace, size_n=N, size_k=K, bias=None)

        def w8a8():
            qx, xs = per_token_group_quant_fp8(x, 128)
            return w8a8_triton_block_scaled_mm(qx, w, xs, s, [128, 128], torch.bfloat16)
        em = float((mar().float() - ref).abs().max() / ref.abs().max())
        et = float((w8a8().float() - ref).abs().max() / ref.abs().max())
        tm, tt = timeit(mar), timeit(w8a8)
        tot.setdefault(T, [0, 0]); tot[T][0] += tm * cnt; tot[T][1] += tt * cnt
        print(f"N={N:5d} K={K:5d} T={T:4d}  marlin {tm*1e3:7.1f} us (err {em:.1e})  w8a8 {tt*1e3:7.1f} us (err {et:.1e})")
for T, (a, b) in tot.items():
    print(f"T={T} total marlin {a*1e3:.1f} us  w8a8 {b*1e3:.1f} us")
