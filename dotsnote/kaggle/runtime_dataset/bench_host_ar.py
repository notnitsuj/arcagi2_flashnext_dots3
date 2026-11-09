"""Correctness + timing of arcstats.host_ar vs NCCL (NCCL_P2P_DISABLE=1 to mimic L4), eager and CUDA-graph replay.
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 python dots3/bench_host_ar.py [NGPU] [tokens,...]
Hang safety: fresh c10d port per run (BENCH_PORT overrides), workers die with the parent (PR_SET_PDEATHSIG) and dump
their stacks + exit at BENCH_DEADLINE_S (default 540), no NCCL teardown (os._exit after a final barrier).
"""
import ctypes, faulthandler, json, os, signal, socket, sys, time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
NG = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()
TOKS = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "8,64,160,256,1024").split(",")]
H = 5120


DEADLINE = float(os.environ.get("BENCH_DEADLINE_S", "540"))


def worker(rank, q, parent):
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG: never outlive the parent
    except Exception:
        pass
    if os.getppid() != parent:
        os._exit(1)
    faulthandler.dump_traceback_later(max(DEADLINE - 15, 10), exit=True)
    import torch.distributed as dist
    from arcstats.host_ar import HostAllReduce
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=NG)
    cpu = dist.new_group(backend="gloo")
    har = HostAllReduce(rank, NG, f"bench{os.getpid() if NG == 1 else 0}", cpu)
    res = {}
    for T in TOKS:
        g = torch.Generator(device="cuda").manual_seed(1000 + rank)
        xs = [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3)]
        # correctness, repeated calls of alternating sizes exercise the parity buffers
        err = 0.0
        for i in range(20):
            x = xs[i % 3]
            ref = x.float().clone(); dist.all_reduce(ref)
            y = har(x)
            if T > 8:
                har(x[:8].contiguous())  # interleave a different size
            err = max(err, ((y.float() - ref).abs().max() / ref.abs().max()).item())
        torch.cuda.synchronize()

        def tm(f, it=50):
            for _ in range(5):
                f()
            torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
            for _ in range(it):
                f()
            torch.cuda.synchronize()
            return (time.time() - t0) / it * 1e6
        x = xs[0]
        t_nccl = tm(lambda: dist.all_reduce(x))
        t_host = tm(lambda: har(x))
        # graph: 10 all-reduces per replay
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                har(x)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
        gr = torch.cuda.CUDAGraph(); outs = []
        with torch.cuda.graph(gr):
            y = x
            for _ in range(10):
                y = har(y)
                y = y * (1.0 / NG)
        gr.replay(); torch.cuda.synchronize()
        mean = x.float().clone(); dist.all_reduce(mean); mean /= NG
        gerr = ((y.float() - mean).abs().max() / mean.abs().max()).item()
        t_graph = tm(lambda: gr.replay(), 20) / 10
        res[T] = dict(err=err, graph_err=gerr, nccl_us=round(t_nccl), host_us=round(t_host), host_graph_us=round(t_graph))
        if os.environ.get("BENCH_Q8", "1") == "1" and (T * H) % (NG * 128) == 0:   # int8-payload variant
            har.q8 = True
            har.bits, har.red_bits = 8, 8   # the int8 kernel (the gate's q8 numbers), whatever ARC_HOST_AR_BITS says
            qerr, same = 0.0, True
            for i in range(6):
                xq = xs[i % 3]
                ref = xq.float().clone(); dist.all_reduce(ref)
                y = har(xq)
                qerr = max(qerr, ((y.float() - ref).abs().max() / ref.abs().max()).item())
                yr = y.clone(); dist.broadcast(yr, 0); same = same and torch.equal(yr, y)   # identical on all ranks
            torch.cuda.synchronize()
            t_q8 = tm(lambda: har(x))
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    har(x)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
            gq = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gq):
                for _ in range(10):
                    har(x)
            gq.replay(); torch.cuda.synchronize()
            res[T].update(q8_err=qerr, q8_same=same, q8_us=round(t_q8), q8_graph_us=round(tm(lambda: gq.replay(), 20) / 10))
            from arcstats import host_ar as _ha
            if _ha.BITS != 8:   # ARC_HOST_AR_BITS: packed low-bit kernel (error vs fp32 sum, rank agreement, graph time)
                har.bits, har.red_bits = _ha.BITS, _ha.RED_BITS
                berr, bsame = 0.0, True
                for i in range(6):
                    xq = xs[i % 3]
                    ref = xq.float().clone(); dist.all_reduce(ref)
                    y = har(xq)
                    berr = max(berr, ((y.float() - ref).abs().max() / ref.abs().max()).item())
                    yr = y.clone(); dist.broadcast(yr, 0); bsame = bsame and torch.equal(yr, y)
                torch.cuda.synchronize(); dist.barrier()
                gb = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gb):
                    for _ in range(10):
                        har(x)
                gb.replay(); torch.cuda.synchronize()
                res[T].update(qb_bits=har.bits, qb_err=berr, qb_same=bsame, qb_graph_us=round(tm(lambda: gb.replay(), 20) / 10))
                har.bits, har.red_bits = 8, 8
            har.q8 = False
    for T in (64, 160, 256):   # sharded-indexer all-gather: [T, 2*768] fp32 per rank
        x = torch.randn(T, 1536, device="cuda", generator=torch.Generator(device="cuda").manual_seed(rank))
        ref = torch.empty(NG * T, 1536, device="cuda"); dist.all_gather_into_tensor(ref, x)
        ok = all(torch.equal(har.all_gather0(x), ref) and torch.equal(har(x.bfloat16()), har(x.bfloat16()))
                 for _ in range(5))

        def tm(f, it=50):
            for _ in range(5):
                f()
            torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
            for _ in range(it):
                f()
            torch.cuda.synchronize()
            return (time.time() - t0) / it * 1e6
        res[f"ag{T}"] = dict(exact=ok, nccl_us=round(tm(lambda: dist.all_gather_into_tensor(ref, x))),
                             host_us=round(tm(lambda: har.all_gather0(x))))
    if rank == 0:
        q.put(res)
    dist.barrier()
    q.close(); q.join_thread()
    os._exit(0)   # no destroy_process_group: an NCCL teardown must never be able to hang the caller


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    if not os.environ.get("BENCH_PORT"):
        sk = socket.socket(); sk.bind(("127.0.0.1", 0)); os.environ["MASTER_PORT"] = str(sk.getsockname()[1]); sk.close()
    else:
        os.environ["MASTER_PORT"] = os.environ["BENCH_PORT"]
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, q, os.getpid()), daemon=True) for i in range(NG)]
    [p.start() for p in ps]
    import queue
    t0, r = time.time(), None
    try:
        while time.time() - t0 < DEADLINE:
            try:
                r = q.get(timeout=2)
                break
            except queue.Empty:
                if not all(p.is_alive() for p in ps):
                    print("bench_host_ar FAILED: a worker exited", [p.exitcode for p in ps], file=sys.stderr, flush=True)
                    break
        for p in ps:
            p.join(timeout=20 if r is not None else 1)
    finally:
        for p in ps:
            if p.is_alive():
                p.kill()
    if r is None:
        sys.exit(1)
    for T, v in r.items():
        print(json.dumps({"tokens": T, "MB": round(T * H * 2 / 2**20, 2), **v} if isinstance(T, int) else {"op": T, **v}),
              flush=True)
