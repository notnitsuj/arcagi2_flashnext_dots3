import os
import torch
import triton
import triton.language as tl
from . import host_ar as _ha
from . import host_dp as _hd
ON = os.environ.get('FN_HOST_DP_FUSE') == '1'

@triton.jit
def _src_load(p0, p1, p2, row, col, msk, W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr):
    c1 = W0
    c2 = W0 + W1
    m0 = msk & (col < c1)
    m1 = msk & (col >= c1) & (col < c2)
    m2 = msk & (col >= c2)
    v = tl.load(p0 + row * W0 + col, mask=m0, other=0)
    if W1 > 0:
        v = tl.where(m1, tl.load(p1 + row * W1 + (col - c1), mask=m1, other=0), v)
    if W2 > 0:
        v = tl.where(m2, tl.load(p2 + row * W2 + (col - c2), mask=m2, other=0), v)
    return v

@triton.jit
def _dst_store(o0, o1, o2, row, col, v, msk, W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr):
    c1 = W0
    c2 = W0 + W1
    tl.store(o0 + row * W0 + col, v, mask=msk & (col < c1))
    if W1 > 0:
        tl.store(o1 + row * W1 + (col - c1), v, mask=msk & (col >= c1) & (col < c2))
    if W2 > 0:
        tl.store(o2 + row * W2 + (col - c2), v, mask=msk & (col >= c2))

@triton.jit
def _host_agm_kernel(x0, x1, x2, o0, o1, o2, hin_ptr, f1_ptr, ep_ptr, err_ptr, stk_ptr, T, m, per, maxn, W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    WT: tl.constexpr = W0 + W1 + W2
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, m)
    offs0 = tl.arange(0, BLOCK)
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * maxn
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        msk = o < hi
        row = o // WT
        col = o % WT
        v = _src_load(x0, x1, x2, row, col, msk, W0, W1, W2)
        tl.store(my_in + o, v, mask=msk)
        _dst_store(o0, o1, o2, RANK * T + row, col, v, msk, W0, W1, W2)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _ha._wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 320 + k, TMO, LD)
            pk = hin_ptr + (par * W + k).to(tl.int64) * maxn
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                msk = o < hi
                row = o // WT
                col = o % WT
                v = tl.load(pk + o, mask=msk, cache_modifier='.cv')
                _dst_store(o0, o1, o2, k * T + row, col, v, msk, W0, W1, W2)
    tl.store(ep_ptr + b, e)

def all_gatherv_fused(self, input_, sizes):
    if not _hd.DO_AG:
        return None
    if sizes is not None and any((s != sizes[0] for s in sizes)):
        return _hd._fb('ag', f'sizes {sizes}')
    single = isinstance(input_, torch.Tensor)
    ts = [input_] if single else list(input_)
    if not 1 <= len(ts) <= 3:
        return _ORIG(self, input_, sizes)
    T = ts[0].shape[0]
    if T == 0 or any((t.shape[0] != T or not t.is_cuda or t[0].numel() * t.element_size() % 4 for t in ts)):
        return _hd._fb('ag', f'T {T} shapes {[tuple(t.shape) for t in ts]} dtypes {[t.dtype for t in ts]}')
    words = [t[0].numel() * t.element_size() // 4 for t in ts]
    WT = sum(words)
    h = self.h
    if T * WT > h.maxn // 2:
        return _hd._fb('ag', f'T*W {T * WT} > {h.maxn // 2}')
    srcs = [t.contiguous().view(T, -1).view(torch.int32) for t in ts]
    outs = [torch.empty((self.world * T,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device) for t in ts]
    dsts = [o.view(self.world * T, -1).view(torch.int32) for o in outs]
    while len(srcs) < 3:
        srcs.append(srcs[0])
        dsts.append(dsts[0])
        words.append(0)
    m = T * WT
    _ha._trace(h, 'AGM', m)
    per = triton.cdiv(triton.cdiv(m, _ha.NB), 8) * 8
    st = h._pre()
    h._go(_host_agm_kernel, (_ha.NB,))(srcs[0], srcs[1], srcs[2], dsts[0], dsts[1], dsts[2], h.hin32, h.f1, h.ep, h.err, h.stk, T, m, per, h.maxn // 2, W0=words[0], W1=words[1], W2=words[2], RANK=self.rank, W=self.world, NBLK=_ha.NB, FS=_ha.FLAG_STRIDE, BLOCK=_ha.TILE, TMO=_ha.TIMEOUT_NS, LD=_ha.POLL_LD, num_warps=_ha.WARPS)
    h._post(st)
    self.n_ag += 1
    if self.n_ag == 1:
        print(f'[host_dp] rank {self.rank}: first fused host all-gather T={T} words={WT} (FN_HOST_DP_FUSE)', flush=True)
    return outs[0] if single else outs
_ORIG = _hd.HostDP.all_gatherv

def install():
    if ON and _hd.HostDP.all_gatherv is _ORIG:
        _hd.HostDP.all_gatherv = all_gatherv_fused
        print('[host_dp] FN_HOST_DP_FUSE: fused multi-tensor host all-gather installed', flush=True)
