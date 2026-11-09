import os
import torch
import triton
import triton.language as tl
from . import host_ar as _ha
MAX_BYTES = int(float(os.environ.get('FN_HOST_DP_MAX_MB', '32')) * (1 << 20))
DO_AG = os.environ.get('FN_HOST_DP_AG', '1') == '1'
DO_RS = os.environ.get('FN_HOST_DP_RS', '1') == '1'
_DBG = int(os.environ.get('FN_HOST_DP_DEBUG', '0'))
_dbg_n = [0]

def _fb(what, why):
    if _dbg_n[0] < _DBG:
        _dbg_n[0] += 1
        print(f'[host_dp] fallback to NCCL ({what}): {why}', flush=True)
    return None

@triton.jit
def _host_rs_kernel(x_ptr, out_ptr, hin_ptr, f1_ptr, ep_ptr, err_ptr, stk_ptr, q, per, maxn, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * maxn
    for j in tl.static_range(W):
        if j != RANK:
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                tl.store(my_in + base + o, tl.load(x_ptr + base + o, mask=m), mask=m)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _ha._wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 410 + k, TMO, LD)
    base = RANK * q
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        m = o < hi
        acc = tl.load(x_ptr + base + o, mask=m).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin_ptr + (par * W + k).to(tl.int64) * maxn
                acc += tl.load(pk + base + o, mask=m, cache_modifier='.cv').to(tl.float32)
        tl.store(out_ptr + o, acc.to(tl.bfloat16), mask=m)
    tl.store(ep_ptr + b, e)

class HostDP:

    def __init__(self, rank, world, name, cpu_group):
        self.h = _ha.HostAllReduce(rank, world, name, cpu_group, max_bytes=MAX_BYTES)
        self.rank, self.world = (rank, world)
        self.n_ag = self.n_rs = self.n_fb = 0

    def all_gatherv(self, input_, sizes):
        if not DO_AG:
            return None
        if sizes is not None and any((s != sizes[0] for s in sizes)):
            return _fb('ag', f'sizes {sizes}')
        single = isinstance(input_, torch.Tensor)
        ts = [input_] if single else list(input_)
        T = ts[0].shape[0]
        if T == 0 or any((t.shape[0] != T or not t.is_cuda or t[0].numel() * t.element_size() % 4 for t in ts)):
            return _fb('ag', f'T {T} shapes {[tuple(t.shape) for t in ts]} dtypes {[t.dtype for t in ts]}')
        words = [t[0].numel() * t.element_size() // 4 for t in ts]
        W = sum(words)
        if T * W > self.h.maxn // 2:
            return _fb('ag', f'T*W {T * W} > {self.h.maxn // 2}')
        pk = [t.contiguous().reshape(T, -1).view(torch.int32) for t in ts]
        packed = pk[0] if len(pk) == 1 else torch.cat(pk, dim=1)
        g = self.h.all_gather0(packed.contiguous())
        if g is None:
            return None
        self.n_ag += 1
        if self.n_ag == 1:
            print(f'[host_dp] rank {self.rank}: first host all-gather T={T} words={W}', flush=True)
        outs, c = ([], 0)
        for t, w in zip(ts, words):
            o = g[:, c:c + w].contiguous().view(t.dtype).reshape((self.world * T,) + tuple(t.shape[1:]))
            outs.append(o)
            c += w
        return outs[0] if single else outs

    def reduce_scatterv(self, input_, dim, sizes, output=None):
        if not DO_RS:
            return None
        if dim < 0:
            dim += input_.dim()
        if dim != 0 or input_.dtype != torch.bfloat16 or (not input_.is_contiguous()) or (not input_.is_cuda):
            return _fb('rs', f'dim {dim} dtype {input_.dtype} contiguous {input_.is_contiguous()}')
        if sizes is not None and any((s != sizes[0] for s in sizes)):
            return _fb('rs', f'sizes {sizes}')
        N = input_.shape[0]
        if N == 0 or N % self.world:
            return _fb('rs', f'N {N}')
        n = input_.numel()
        if n > self.h.maxn:
            return _fb('rs', f'numel {n} > {self.h.maxn}')
        T = N // self.world
        q = n // self.world
        if output is None:
            output = torch.empty((T,) + tuple(input_.shape[1:]), dtype=input_.dtype, device=input_.device)
        elif not output.is_contiguous() or output.dtype != input_.dtype:
            return _fb('rs', f'output contiguous {output.is_contiguous()} dtype {output.dtype}')
        h = self.h
        per = triton.cdiv(triton.cdiv(q, _ha.NB), 8) * 8
        st = h._pre()
        h._go(_host_rs_kernel, (_ha.NB,))(input_, output, h.hin, h.f1, h.ep, h.err, h.stk, q, per, h.maxn, RANK=self.rank, W=self.world, NBLK=_ha.NB, FS=_ha.FLAG_STRIDE, BLOCK=_ha.TILE, TMO=_ha.TIMEOUT_NS, LD=_ha.POLL_LD, num_warps=_ha.WARPS)
        h._post(st)
        self.n_rs += 1
        if self.n_rs == 1:
            print(f'[host_dp] rank {self.rank}: first host reduce-scatter N={N}', flush=True)
        return output

def patch():
    from vllm.distributed.device_communicators import cuda_communicator as cc
    from vllm.logger import init_logger
    log = init_logger('vllm.fnext.host_dp')
    orig_init = cc.CudaCommunicator.__init__
    orig_agv = cc.CudaCommunicator.all_gatherv
    orig_rsv = cc.CudaCommunicator.reduce_scatterv
    orig_rsvo = cc.CudaCommunicator.reduce_scatterv_into_output

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._fn_hdp = None
        groups = ('dp', 'ep') if os.environ.get('FN_HOST_DP_EP') == '1' else ('dp',)
        if os.environ.get('FN_HOST_SP') == '1':
            groups += ('tp',)
        if self.world_size in (2, 4) and self.unique_name.split(':')[0] in groups:
            ranks = sorted(torch.distributed.get_process_group_ranks(self.cpu_group))
            tag = self.unique_name.split(':')[0]
            obj = [f'{os.getpid()}_h{tag}_{'_'.join(map(str, ranks))}' if self.rank_in_group == 0 else None]
            torch.distributed.broadcast_object_list(obj, src=ranks[0], group=self.cpu_group)
            name = obj[0]
            self._fn_hdp = HostDP(self.rank_in_group, self.world_size, name, self.cpu_group)
            log.info('FN_HOST_DP: host all-gather/reduce-scatter for %s (world %d, ranks %s, %.0f MB max)', self.unique_name, self.world_size, ranks, MAX_BYTES / 2 ** 20)
            if tag == 'tp':
                h = self._fn_hdp
                self.custom_all_gather = lambda x: h.all_gatherv(x, None) if x.dim() >= 1 else None
                self.custom_reduce_scatter = lambda x: h.reduce_scatterv(x, 0, None)

    def all_gatherv(self, input_, dim=0, sizes=None):
        h = getattr(self, '_fn_hdp', None)
        if h is not None and dim != 0:
            _fb('ag', f'dim {dim}')
        if h is not None and dim == 0:
            r = h.all_gatherv(input_, sizes)
            if r is not None:
                return r
        return orig_agv(self, input_, dim, sizes)

    def reduce_scatterv(self, input_, dim=-1, sizes=None):
        h = getattr(self, '_fn_hdp', None)
        if h is not None:
            r = h.reduce_scatterv(input_, dim, sizes)
            if r is not None:
                return r
        return orig_rsv(self, input_, dim, sizes)

    def reduce_scatterv_into_output(self, input_, output, dim=-1, sizes=None):
        h = getattr(self, '_fn_hdp', None)
        if h is not None:
            r = h.reduce_scatterv(input_, dim, sizes, output)
            if r is not None:
                return r
        return orig_rsvo(self, input_, output, dim, sizes)
    orig_ag = cc.CudaCommunicator.all_gather
    host_logits = os.environ.get('FN_HOST_LOGITS') == '1'

    def all_gather(self, input_, dim=-1):
        h = getattr(self, '_fn_hdp', None)
        if host_logits and h is not None and (input_.dim() == 2) and (dim in (-1, 1)) and self.unique_name.startswith('tp'):
            T, C = input_.shape
            r = h.all_gatherv(input_.contiguous(), None)
            if r is not None:
                return r.reshape(self.world_size, T, C).movedim(0, 1).reshape(T, self.world_size * C)
        return orig_ag(self, input_, dim)
    cc.CudaCommunicator.__init__ = init
    cc.CudaCommunicator.all_gather = all_gather
    cc.CudaCommunicator.all_gatherv = all_gatherv
    cc.CudaCommunicator.reduce_scatterv = reduce_scatterv
    cc.CudaCommunicator.reduce_scatterv_into_output = reduce_scatterv_into_output
