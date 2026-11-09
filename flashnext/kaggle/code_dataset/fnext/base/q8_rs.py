import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from . import host_ar as _ha
from . import host_dp as _hd
ON_EP = os.environ.get('FN_EP_Q8_RS') == '1'
ON_TP = os.environ.get('FN_TP_Q8_RS') == '1'
ON = ON_EP or ON_TP
MIN_T = int(os.environ.get('FN_EP_Q8_RS_MIN_T', '24'))
SIM = os.environ.get('FN_EP_Q8_RS_SIM') == '1'

def _sim_inplace(self, input_, sizes):
    from . import q8_dispatch as Q
    if input_.dim() != 2 or input_.dtype != torch.bfloat16 or input_.shape[1] % G or (not input_.is_contiguous()) or (sizes is None) or (sum(sizes) != input_.shape[0]):
        return
    o = 0
    for j, n in enumerate(sizes):
        if j != self.rank and n:
            rows = input_[o:o + n]
            rows.copy_(Q.unpack(Q.pack(rows), input_.shape[1]))
        o += n
G = 128

@triton.jit
def _host_rs_q8_kernel(x_ptr, out_ptr, hin8_ptr, f1_ptr, ep_ptr, err_ptr, stk_ptr, q, per, slot_bytes, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    NGB: tl.constexpr = BLOCK // 128
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs = tl.arange(0, BLOCK)
    goffs = tl.arange(0, NGB)
    my = hin8_ptr + (par * W + RANK).to(tl.int64) * slot_bytes
    for j in tl.static_range(W):
        if j != RANK:
            for s in range(lo, hi, BLOCK):
                o = s + offs
                m = o < hi
                v = tl.load(x_ptr + j * q + o, mask=m, other=0.0).to(tl.float32)
                v2 = tl.reshape(v, (NGB, 128))
                amax = tl.max(tl.abs(v2), axis=1)
                sc = tl.where(amax > 0, tl.math.div_rn(amax, tl.full(amax.shape, 127.0, tl.float32)), 1e-30)
                c = libdevice.rint(tl.math.div_rn(v2, sc[:, None] + tl.zeros(v2.shape, tl.float32)))
                c = tl.minimum(tl.maximum(c, -127.0), 127.0).to(tl.int8)
                tl.store(my + j * q + o, tl.reshape(c, (BLOCK,)), mask=m)
                gi = (j * q + s) // 128 + goffs
                tl.store((my + W * q).to(tl.pointer_type(tl.float32)) + gi, sc, mask=s + goffs * 128 < hi)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _ha._wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 430 + k, TMO, LD)
    for s in range(lo, hi, BLOCK):
        o = s + offs
        m = o < hi
        acc = tl.load(x_ptr + RANK * q + o, mask=m, other=0.0).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin8_ptr + (par * W + k).to(tl.int64) * slot_bytes
                c = tl.load(pk + RANK * q + o, mask=m, other=0, cache_modifier='.cv').to(tl.float32)
                gi = (RANK * q + s) // 128 + goffs
                sc = tl.load((pk + W * q).to(tl.pointer_type(tl.float32)) + gi, mask=s + goffs * 128 < hi, other=0.0, cache_modifier='.cv')
                acc += tl.reshape(tl.reshape(c, (NGB, 128)) * sc[:, None], (BLOCK,))
        tl.store(out_ptr + o, acc.to(tl.bfloat16), mask=m)
    tl.store(ep_ptr + b, e)

def reduce_scatterv_q8(self, input_, dim, sizes, output=None):
    if not (self.world == 4 and ON_EP or (self.world == 2 and ON_TP)):
        return _ORIG(self, input_, dim, sizes, output)
    if dim < 0:
        dim += input_.dim()
    if dim != 0 or input_.dtype != torch.bfloat16 or (not input_.is_contiguous()) or (not input_.is_cuda):
        return _ORIG(self, input_, dim, sizes, output)
    if sizes is not None and any((s != sizes[0] for s in sizes)):
        if SIM and (self.world == 4 and ON_EP or (self.world == 2 and ON_TP)) and (dim == 0):
            _sim_inplace(self, input_, sizes)
        return _ORIG(self, input_, dim, sizes, output)
    N = input_.shape[0]
    if N == 0 or N % self.world or N // self.world < MIN_T:
        return _ORIG(self, input_, dim, sizes, output)
    n = input_.numel()
    q = n // self.world
    h = self.h
    slot_bytes = h.maxn * 2
    if q % G or self.world * q + 4 * (self.world * q // G) > slot_bytes:
        if SIM:
            _sim_inplace(self, input_, [N // self.world] * self.world)
        return _ORIG(self, input_, dim, sizes, output)
    T = N // self.world
    if output is None:
        output = torch.empty((T,) + tuple(input_.shape[1:]), dtype=input_.dtype, device=input_.device)
    elif not output.is_contiguous() or output.dtype != input_.dtype:
        return _ORIG(self, input_, dim, sizes, output)
    block = max(G, min(_ha.TILE, 2048))
    per = triton.cdiv(triton.cdiv(q, _ha.NB), block) * block
    st = h._pre()
    h._go(_host_rs_q8_kernel, (_ha.NB,))(input_, output, h.hin8, h.f1, h.ep, h.err, h.stk, q, per, slot_bytes, RANK=self.rank, W=self.world, NBLK=_ha.NB, FS=_ha.FLAG_STRIDE, BLOCK=block, TMO=_ha.TIMEOUT_NS, LD=_ha.POLL_LD, num_warps=_ha.WARPS)
    h._post(st)
    self.n_rs += 1
    if not getattr(self, '_q8rs_logged', False):
        self._q8rs_logged = True
        print(f'[host_dp] rank {self.rank}: first int8 reduce-scatter W={self.world} N={N} (FN_EP_Q8_RS/FN_TP_Q8_RS)', flush=True)
    return output
_ORIG = _hd.HostDP.reduce_scatterv

def install():
    if ON and _hd.HostDP.reduce_scatterv is _ORIG:
        _hd.HostDP.reduce_scatterv = reduce_scatterv_q8
        print(f'[host_dp] FN_EP_Q8_RS/FN_TP_Q8_RS: int8 reduce-scatter installed (EP {ON_EP}, TP {ON_TP})', flush=True)
        return True
    return False
