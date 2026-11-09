"""ARC_AG_U16 (arcstats.ag_pack) exactness + cost: the packed indexer all-gather payload must rebuild the int32 payload
bit for bit, and the merged top-k must be identical.
  CUDA_VISIBLE_DEVICES=0 python dots3/test_ag_u16.py                 # exactness (P fake ranks in one process)
  BENCH_GPUS=0,1 python dots3/test_ag_u16.py --ag 2 [rows]          # host all-gather time int32 vs packed (+pack/unpack)
"""
import os, sys
if os.environ.get("BENCH_GPUS"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["BENCH_GPUS"]
os.environ["ARC_AG_U16"] = "1"
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
from arcstats import ag_pack, fast_topk   # noqa: E402

P, KL, LC, TOPK = 4, 768, 65536, 2048


def local_payload(lg, ctx, rank):
    if fast_topk.lib() is not None:
        return fast_topk.local(lg, ctx, rank, P, KL)
    v, j = torch.topk(lg, KL, dim=1)          # torch path of shard_indexer.sharded_topk
    return torch.cat([v.float(), (j * P + rank).to(torch.int32).view(torch.float32)], 1)


def exactness():
    torch.manual_seed(0)
    bad = 0
    for n in (1, 7, 64, 256):
        ctx = torch.randint(1, LC * P, (n,), dtype=torch.int32, device="cuda")
        ctx[: max(1, n // 4)] = torch.randint(1, 3 * KL, (max(1, n // 4),), dtype=torch.int32, device="cuda")  # pads
        ctx[-1] = LC * P                                                                          # j = 65535 used
        pks = []
        for r in range(P):
            lg = torch.randn(n, LC, device="cuda")
            lg[:, ::97] = 0.5        # exact ties across columns
            lv = ((ctx - r + P - 1) // P).clamp(min=0)
            lg[torch.arange(LC, device="cuda")[None, :] >= lv[:, None]] = float("-inf")
            pks.append(local_payload(lg, ctx, r))
        ref = torch.cat(pks, 0)                                           # what the all-gather returns today
        got = ag_pack.unpack(torch.cat([ag_pack.pack(pk, P, r) for r, pk in enumerate(pks)], 0), n, P)
        same = torch.equal(ref.view(torch.int32), got.view(torch.int32))
        mt = "n/a"
        if fast_topk.lib() is not None:
            o1 = torch.empty(n, TOPK, dtype=torch.int32, device="cuda")
            o2 = torch.empty_like(o1)
            fast_topk.merge(ref, n, P, KL, TOPK, o1)
            fast_topk.merge(got, n, P, KL, TOPK, o2)
            mt = torch.equal(o1, o2)
            bad += not mt
        bad += not same
        print(f"ag_u16 n {n}: payload bit-identical {same}, merged top-{TOPK} identical {mt}, bytes/row "
              f"{ref.shape[1] * 4} -> {ag_pack.pack(pks[0], P, 0).shape[1] * 4}", flush=True)
    print("ag_u16 exactness", "OK" if bad == 0 else f"FAILED ({bad})")
    return bad


def ag_worker(rank, NG, n, q):
    import torch.distributed as dist
    from arcstats.host_ar import HostAllReduce
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NG)
    cpu = dist.new_group(backend="gloo")
    har = HostAllReduce(rank, NG, f"agu16{os.getppid()}", cpu)
    pk = torch.randn(n, 2 * KL, device="cuda")
    pk[:, KL:] = (torch.randint(0, LC, (n, KL), device="cuda", dtype=torch.int32) * NG + rank).view(torch.float32)

    def gt(fn, it=50):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier(group=cpu)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()
        g.replay(); torch.cuda.synchronize(); dist.barrier(group=cpu)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            g.replay()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / it / 10 * 1e3
    t32 = gt(lambda: har.all_gather0(pk))
    t16 = gt(lambda: ag_pack.unpack(har.all_gather0(ag_pack.pack(pk, NG, rank)), n, NG))
    a = har.all_gather0(pk)
    b = ag_pack.unpack(har.all_gather0(ag_pack.pack(pk, NG, rank)), n, NG)
    ok = torch.equal(a.view(torch.int32), b.view(torch.int32))
    if rank == 0:
        q.put(f"ag_u16 host AG rows {n}: int32 payload {t32:.0f} us, uint16 ids (pack+AG+unpack) {t16:.0f} us "
              f"({t16 / t32:.2f}x), identical {ok}")
    dist.barrier(group=cpu)
    os._exit(0)


if __name__ == "__main__":
    if "--ag" in sys.argv:
        import socket
        import torch.multiprocessing as mp
        NG = int(sys.argv[sys.argv.index("--ag") + 1])
        i = sys.argv.index("--ag")
        rows = [int(x) for x in (sys.argv[i + 2] if len(sys.argv) > i + 2 else "64,256").split(",")]
        os.environ.setdefault("ARC_HOST_AR_Q8", "1")
        for n in rows:
            s = socket.socket(); s.bind(("127.0.0.1", 0)); os.environ["MASTER_PORT"] = str(s.getsockname()[1]); s.close()
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            ctx = mp.get_context("spawn"); q = ctx.Queue()
            ps = [ctx.Process(target=ag_worker, args=(r, NG, n, q)) for r in range(NG)]
            [p.start() for p in ps]
            print(q.get(timeout=300), flush=True)
            [p.join(60) for p in ps]
    else:
        sys.exit(1 if exactness() else 0)
