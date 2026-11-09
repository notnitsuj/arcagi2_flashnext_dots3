"""Hardware microbenchmarks for the Dots3 L4 build (runs on any CUDA box; meant for Kaggle 4xL4).

  python dots3/l4_microbench.py OUT.json [--gpus 4]
1. routed-expert MoE per layer (TP4 shard shapes): arcstats.lowbit_moe mixed 3/2-bit (144 experts) vs Marlin INT4
   (128 experts), decode M in {8,32,64,128} tokens, plus a 1024-token prefill chunk
2. UVA host-read bandwidth (offload row gathers from cudaHostRegister'ed memory)
3. NCCL all-reduce latency/bandwidth over the TP group (no P2P on Kaggle L4)
"""
import json, os, sys, time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "microbench.json"
NG = int(sys.argv[sys.argv.index("--gpus") + 1]) if "--gpus" in sys.argv else torch.cuda.device_count()


def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def bench_moe(res):
    from arcstats.lowbit_moe import LowbitProj, LowbitMoE
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    H, I, TOPK = 5120, 384, 8
    E = 144
    bits = [3 if e < 96 else 2 for e in range(E)]
    gu = LowbitProj(bits, [torch.randint(0, 2 ** b, (2 * I, H), dtype=torch.uint8) for b in bits],
                    [torch.rand(2 * I, H // 128) * .01 for _ in bits], [torch.randint(0, 4, (2 * I, H // 128), dtype=torch.uint8) for _ in bits], dev)
    dn = LowbitProj(bits, [torch.randint(0, 2 ** b, (H, I), dtype=torch.uint8) for b in bits],
                    [torch.rand(H, I // 128) * .01 for _ in bits], [torch.randint(0, 4, (H, I // 128), dtype=torch.uint8) for _ in bits], dev)
    moe = LowbitMoE(gu, dn, E)
    res["lowbit_bytes"] = gu.nbytes() + dn.nbytes()
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types
        qt = scalar_types.uint4b8
        Em = 128

        def mk(n_out, k):
            out = marlin_quantize(torch.randn(k, n_out, device=dev, dtype=torch.bfloat16) * 0.02, qt, 128)
            return out[1].unsqueeze(0).repeat(Em, 1, 1).contiguous(), out[2].unsqueeze(0).repeat(Em, 1, 1).contiguous()
        w1, s1 = mk(2 * I, H); w2, s2 = mk(H, I)
        res["marlin_bytes"] = (w1.numel() + w2.numel()) * 4
    except Exception as ex:
        res["marlin_error"] = repr(ex)[:200]; w1 = None
    for M in (8, 32, 64, 128, 1024):
        x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        tw, tid = torch.topk(torch.rand(M, E, device=dev), TOPK, dim=-1)
        res[f"lowbit_M{M}_ms"] = timeit(lambda: moe(x, tw.float(), tid.int())) * 1e3
        if w1 is not None:
            twm, tidm = torch.topk(torch.rand(M, Em, device=dev), TOPK, dim=-1)
            res[f"marlin_M{M}_ms"] = timeit(lambda: fused_marlin_moe(x, w1, w2, None, None, s1, s2, twm.float(), tidm,
                                                                     qt.id, global_num_experts=Em)) * 1e3
        print(f"M={M}: lowbit {res[f'lowbit_M{M}_ms']:.3f} ms, marlin {res.get(f'marlin_M{M}_ms', -1):.3f} ms", flush=True)


def bench_uva(res):
    from arcstats.sparse_offload import _registered_host, LayerOffload
    dev = torch.device("cuda:0")
    R = 4_000_000
    off = LayerOffload(R, dev)
    off.host.view(torch.uint8).random_()
    rows = torch.randint(0, R, (2048 * 64,), device=dev, dtype=torch.int32)
    out = torch.empty(rows.shape[0], 576, device=dev, dtype=off.host.dtype)
    t = timeit(lambda: off.gather(rows, out))
    res["uva_gather_GBps"] = rows.shape[0] * 576 / t / 1e9
    print(f"UVA random-row gather: {res['uva_gather_GBps']:.1f} GB/s", flush=True)
    h = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
    d = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    res["h2d_GBps"] = (1 << 30) / timeit(lambda: d.copy_(h, non_blocking=True), 5) / 1e9
    print(f"pinned H2D: {res['h2d_GBps']:.1f} GB/s", flush=True)


def _ar_worker(rank, n, q):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    dist.init_process_group("nccl", rank=rank, world_size=n)
    r = {}
    for nbytes in (16 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20):
        x = torch.ones(nbytes // 2, device=f"cuda:{rank}", dtype=torch.bfloat16)
        t = timeit(lambda: dist.all_reduce(x), 50)
        r[f"allreduce_{nbytes >> 10}KB_us"] = t * 1e6
    if rank == 0:
        q.put(r)
    dist.destroy_process_group()


def bench_allreduce(res):
    if NG < 2:
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_ar_worker, args=(i, NG, q)) for i in range(NG)]
    [p.start() for p in ps]
    r = q.get(timeout=600)
    [p.join() for p in ps]
    res.update(r)
    print("all-reduce:", {k: round(v, 1) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    res = {"gpu": torch.cuda.get_device_name(0), "n_gpu": NG}
    for f in (bench_moe, bench_uva, bench_allreduce):
        try:
            f(res)
        except Exception as ex:
            res[f.__name__ + "_error"] = repr(ex)[:300]
            print(f.__name__, "failed", ex, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
