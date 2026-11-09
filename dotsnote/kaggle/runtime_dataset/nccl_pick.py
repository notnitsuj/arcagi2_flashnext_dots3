"""Time NCCL all-reduce / all-gather at the serving message sizes under the CURRENT NCCL_* environment; prints one JSON
line. run_l4.py calls it once per candidate setting at notebook startup and launches the server with the fastest.

  NCCL_PROTO=Simple python dots3/nccl_pick.py [NGPU]
Sizes: decode all-reduce 160 tok x 5120 x bf16 (1.6 MB), prefill-chunk all-reduce 1024 tok (10.5 MB), sharded-indexer
all-gather (~0.4 MB per rank).
"""
import json, os, sys, time

import torch
import torch.multiprocessing as mp

NG = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()
SIZES = {"ar_decode": 160 * 5120 * 2, "ar_prefill": 1024 * 5120 * 2, "ag_index": 400 << 10}


def worker(rank, q):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("nccl", rank=rank, world_size=NG)
    r = {}
    for k, nb in SIZES.items():
        x = torch.ones(nb // 2, device="cuda", dtype=torch.bfloat16)
        if k.startswith("ag"):
            out = torch.empty(NG * x.numel(), device="cuda", dtype=x.dtype)
            f = lambda: dist.all_gather_into_tensor(out, x)
        else:
            f = lambda: dist.all_reduce(x)
        for _ in range(5):
            f()
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(40):
            f()
        torch.cuda.synchronize()
        r[k] = (time.time() - t0) / 40 * 1e6
    if rank == 0:
        q.put(r)
    dist.destroy_process_group()


if __name__ == "__main__":
    if not os.environ.get("MASTER_PORT"):   # fresh port per run: a leftover listener must not break the next candidate
        import socket
        sk = socket.socket(); sk.bind(("127.0.0.1", 0)); os.environ["MASTER_PORT"] = str(sk.getsockname()[1]); sk.close()
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, q)) for i in range(NG)]
    [p.start() for p in ps]
    r = q.get(timeout=600)
    [p.join() for p in ps]
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k != "NCCL_P2P_DISABLE"}
    print(json.dumps({"env": env, "us": {k: round(v, 1) for k, v in r.items()}}), flush=True)
