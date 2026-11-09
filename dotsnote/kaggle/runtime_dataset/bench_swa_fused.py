"""Fused SWA decode attention (arcstats.swa_fused) vs vLLM's Dots3 gather+bmm path: correctness and timing at L4-like
decode load (48 requests x 4 MTP tokens, window 513, 1088-wide FP8 rows, 32-row pages).  ARC_TRITON_SM89=1 for L4
codegen on an H100.
  python dots3/bench_swa_fused.py [ctx]   (prints orig / fused / packed / cuda<cfg> timings and errors)
Timing cycles over ARC_SWA_BENCH_LAYERS (6) KV caches so the windows come from DRAM as in the server (a single cache's
windows, ~27 MB, stay in the 48 MB L2 of an L4 and flatter the two-pass kernels). CUDA candidates (csrc/swa_attn.cu,
ARC_SWA_FUSED=2 + ARC_SWA_CUDA=cfg) from ARC_SWA_CUDA_CANDS (default "1,11"); skipped if swa_attn.so is missing."""
import os, sys, time, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import triton_sm89  # noqa
import torch
from vllm.models.dots3_note.nvidia import attention as da
orig = da.Dots3NoteTritonMLAImpl._forward_swa_mqa
from arcstats import swa_fused  # noqa: E402
swa_fused.install()
fused = da.Dots3NoteTritonMLAImpl._forward_swa_mqa
dev = torch.device("cuda"); torch.manual_seed(0)
NREQ, QL, H, D, DN, PAGE, WIN = int(os.environ.get("ARC_SWA_BENCH_NREQ", "48")), 4, 16, 1088, 1024, 32, 513
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
NL = int(os.environ.get("ARC_SWA_BENCH_LAYERS", "6"))
CANDS = [int(c) for c in os.environ.get("ARC_SWA_CUDA_CANDS", "1,11").split(",") if c]
pages = (CTX + PAGE - 1) // PAGE + 1
nblk = NREQ * pages + 8
kvs = [(torch.randn(nblk, PAGE, D, device=dev) * 0.5).to(torch.float8_e4m3fn) for _ in range(NL)]
kv = kvs[0]
bt = torch.randperm(nblk - 8, device=dev)[:NREQ * pages].view(NREQ, pages).int()
sl = torch.randint(CTX // 2, CTX, (NREQ,), device=dev).int()
sl[0] = 3                                                # short request: window clipped at 0 (token 0 has none)
sl[1:6] = torch.tensor([514, 515, 516, 517, 100], dtype=torch.int32)   # windows starting at / near position 0
dec = types.SimpleNamespace(query_len=QL, block_table=bt, seq_lens=sl)
self = types.SimpleNamespace(head_size=D, sliding_window=WIN, scale=D ** -0.5, kv_lora_rank=DN)
q = torch.randn(NREQ * QL, H, D, device=dev, dtype=torch.bfloat16)
ks = torch.tensor(1.0, device=dev)
a = orig(self, q, kv, dec, ks)[0]


def rel(b):
    e = (a.float() - b.float()).abs().amax((1, 2)) / a.float().abs().amax((1, 2))
    return e


cuda_ok = bool(CANDS) and swa_fused.cuda_lib() is not None
ok = {}
for nm, pk in (("", False), ("packed ", True)):   # ARC_SWA_FUSED=1 split-K per token vs =2 request-packed
    swa_fused.PACKED, swa_fused.CUDA = pk, 0
    try:   # a variant that cannot compile on this GPU (e.g. shared memory) is skipped, not fatal
        b = fused(self, q, kv, dec, ks)[0]
    except Exception as ex:
        print(nm + "variant failed:", repr(ex)[:160]); continue
    ok[pk] = True
    e = rel(b)
    print(nm + "rel err (excluding the qpos<0 token)", float(e[1:].max()), "per-token first 6", [round(float(x), 4) for x in e[:6]])
if cuda_ok:
    swa_fused.PACKED = True
    for c in CANDS:
        swa_fused.CUDA = c
        try:
            b = fused(self, q, kv, dec, ks)[0]
        except Exception as ex:
            print(f"cuda{c} variant failed:", repr(ex)[:160]); continue
        ok[f"cuda{c}"] = True
        e = rel(b)
        print(f"cuda{c} rel err (excluding the qpos<0 token)", float(e[1:].max()), "empty-window token zero",
              bool(b[0].float().abs().max() == 0))


def timed(f):
    """us per call over NL caches round-robin: CUDA graph of 4 x NL calls (eager fallback)"""
    for x in kvs:
        f(x)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(4):
                for x in kvs:
                    f(x)
        run, n = g.replay, 4 * NL
    except Exception:
        torch.cuda.synchronize()
        run, n = (lambda: [f(x) for _ in range(4) for x in kvs]), 4 * NL
    run(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        t0 = time.time(); run(); torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / n)
    return best * 1e3


runs = [("orig", False, 0), ("fused", False, 0), ("packed", True, 0)] + [(f"cuda{c}", True, c) for c in CANDS]
for nm, pk, c in runs:
    if nm != "orig" and (pk if not c else nm) not in ok:
        continue
    swa_fused.PACKED, swa_fused.CUDA = pk, c
    f = (lambda x: orig(self, q, x, dec, ks)) if nm == "orig" else (lambda x: fused(self, q, x, dec, ks))
    print(nm, f"{timed(f):.3f} ms/call")
