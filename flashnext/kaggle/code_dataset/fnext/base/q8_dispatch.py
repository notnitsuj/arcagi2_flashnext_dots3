import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
ON = os.environ.get('FN_EP_Q8') == '1'
MIN_T = int(os.environ.get('FN_EP_Q8_MIN_T', '24'))
G = 128

@triton.jit
def _q8_pack_kernel(x_ptr, out_ptr, H, stride_x, ROWB: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, NGP)
    c = tl.arange(0, 128)
    gm = g < NG
    m = gm[:, None]
    x = tl.load(x_ptr + r * stride_x + g[:, None] * 128 + c[None, :], mask=m, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    s = tl.where(amax > 0, tl.math.div_rn(amax, tl.full(amax.shape, 127.0, tl.float32)), 1e-30)
    q = libdevice.rint(tl.math.div_rn(x, s[:, None] + tl.zeros(x.shape, tl.float32)))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    row = out_ptr + r * ROWB
    tl.store(row + g[:, None] * 128 + c[None, :], q, mask=m)
    tl.store((row + NG * 128).to(tl.pointer_type(tl.float32)) + g, s, mask=gm)

@triton.jit
def _q8_unpack_kernel(in_ptr, y_ptr, ROWB: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, NGP)
    c = tl.arange(0, 128)
    gm = g < NG
    row = in_ptr + r * ROWB
    q = tl.load(row + g[:, None] * 128 + c[None, :], mask=gm[:, None], other=0).to(tl.float32)
    s = tl.load((row + NG * 128).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0)
    tl.store(y_ptr + r * (NG * 128) + g[:, None] * 128 + c[None, :], (q * s[:, None]).to(tl.bfloat16), mask=gm[:, None])

def pack(x):
    T, H = x.shape
    NG = H // G
    rowb = H + 4 * NG
    out = torch.empty(T, rowb // 4, dtype=torch.int32, device=x.device)
    if T:
        _q8_pack_kernel[T,](x, out.view(torch.int8), H, x.stride(0), ROWB=rowb, NG=NG, NGP=triton.next_power_of_2(NG), num_warps=4)
    return out

def unpack(p, H):
    N = p.shape[0]
    NG = H // G
    y = torch.empty(N, H, dtype=torch.bfloat16, device=p.device)
    if N:
        _q8_unpack_kernel[N,](p.view(torch.int8), y, ROWB=H + 4 * NG, NG=NG, NGP=triton.next_power_of_2(NG), num_warps=4)
    return y

def eligible(x):
    return x.dim() == 2 and x.dtype == torch.bfloat16 and (x.shape[1] % G == 0) and x.is_cuda and (x.stride(1) == 1)
_ST = {'n': 0}

def install():
    if not ON:
        return False
    from vllm.distributed.device_communicators import all2all as A
    cls = A.AgRsAll2AllManager
    if getattr(cls.dispatch, '_fn_q8', False):
        return True
    orig = cls.dispatch

    def dispatch(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None):
        if not eligible(hidden_states):
            return orig(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors)
        if MIN_T > 1:
            try:
                sizes = self._get_sizes(hidden_states.shape[0], self._get_comm_group(is_sequence_parallel))
                small = max(sizes) < MIN_T
            except Exception:
                small = False
            if small:
                return orig(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors)
        H = hidden_states.shape[1]
        res = orig(self, pack(hidden_states), topk_weights, topk_ids, is_sequence_parallel, extra_tensors)
        _ST['n'] += 1
        if _ST['n'] == 1:
            print(f'[q8_dispatch] first int8 EP dispatch: T={hidden_states.shape[0]} H={H} ({(H + 4 * (H // G)) / (2 * H):.3f}x bytes)', flush=True)
        return (unpack(res[0], H),) + tuple(res[1:])
    dispatch._fn_q8 = True
    cls.dispatch = dispatch
    print('[q8_dispatch] FN_EP_Q8: int8 (per-128 absmax) EP dispatch payload installed', flush=True)
    return True
