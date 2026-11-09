"""ARC_FAKE_TP=<W> (local debugging only): run ONE process as rank 0 of a W-way tensor-parallel group on a single GPU.

Weights, KV specs, workspaces and per-layer buffers get exactly the shapes of a real TP-W rank 0, so the GPU memory
footprint (allocation sizes and order, allocator fragmentation) of an L4 rank can be reproduced on one local GPU.
Collectives are local stand-ins with the right output shapes (all-reduce = identity, all-gather = W copies,
reduce-scatter = first chunk), so the numbers the model computes are garbage: use it for memory, never for outputs.
Start vLLM with --tensor-parallel-size 1; the worker switches its parallel config to W after distributed init."""
import contextlib
import os

import torch


class _FakeComm:
    def __init__(self, w):
        self.w = w
        self.world_size = w
        self.rank_in_group = 0

    def all_reduce(self, x, *a, **kw):
        return x.clone()

    def all_gather(self, x, dim=-1):
        return torch.cat([x] * self.w, dim=dim)

    def all_gatherv(self, x, dim=0, sizes=None):
        if isinstance(x, (list, tuple)):
            return [self.all_gather(t, dim) for t in x]
        return self.all_gather(x, dim)

    def reduce_scatter(self, x, dim=-1):
        return x.chunk(self.w, dim=dim)[0].contiguous()

    def reduce_scatterv(self, x, dim=-1, sizes=None):
        if sizes is not None:
            return x.narrow(dim, 0, sizes[0]).contiguous()
        return self.reduce_scatter(x, dim)

    def gather(self, x, dst=0, dim=-1):
        return self.all_gather(x, dim)

    def broadcast(self, x, src=0):
        return x

    def send(self, *a, **kw):
        pass

    def recv(self, *a, **kw):
        raise RuntimeError("fake TP: recv")

    def destroy(self):
        pass

    def __getattr__(self, k):   # capture contexts, flags ... of real communicators: inert
        if k.startswith("__"):
            raise AttributeError(k)
        return None


def _fake_group(g, w):
    g.world_size = w
    g.rank_in_group = 0
    g.ranks = list(range(w))
    g.device_communicator = _FakeComm(w)
    g.use_device_communicator = True
    g.broadcast = lambda x, src=0: x
    g.broadcast_object = lambda obj=None, src=0: obj
    g.broadcast_object_list = lambda lst, src=0, group=None: lst
    g.broadcast_tensor_dict = lambda d=None, src=0, *a, **kw: d
    g.barrier = lambda: None
    g.gather = lambda x, dst=0, dim=-1: torch.cat([x] * w, dim=dim)
    g.all_gatherv = g.device_communicator.all_gatherv
    g.reduce_scatterv = g.device_communicator.reduce_scatterv
    orig_gc = g.graph_capture

    @contextlib.contextmanager
    def graph_capture(ctx=None):   # vLLM asserts a real CUDA communicator here: hide the fake one while entering
        dc, g.device_communicator = g.device_communicator, None
        try:
            with orig_gc(ctx) as c:
                g.device_communicator = dc
                yield c
        finally:
            g.device_communicator = dc
    g.graph_capture = graph_capture


def install():
    w = int(os.environ.get("ARC_FAKE_TP", "0"))
    if w <= 1:
        return
    from vllm.v1.worker import gpu_worker as gw
    from vllm.distributed import parallel_state as ps
    orig = gw.init_worker_distributed_environment

    def init(vllm_config, *a, **kw):
        r = orig(vllm_config, *a, **kw)
        for name in ("_TP", "_EP"):
            g = getattr(ps, name, None)
            if g is not None and g.world_size == 1:
                _fake_group(g, w)
        vllm_config.parallel_config.tensor_parallel_size = w
        print(f"[fake_tp] this process is rank 0 of a fake TP{w} group (memory debugging only)", flush=True)
        return r
    gw.init_worker_distributed_environment = init
