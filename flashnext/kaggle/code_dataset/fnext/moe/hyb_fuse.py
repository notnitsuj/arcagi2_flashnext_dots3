import os
import torch
import triton
import triton.language as tl
FUSE = os.environ.get('FN_MOE_HYB_FUSE', '0') in ('1', '2')
CHUNK = int(os.environ.get('FN_MOE_CHUNK', '0') or 0)
CONC_MAX = int(os.environ.get('FN_MOE_HYB_CONC_MAX', '512'))
ON = FUSE or CHUNK > 0
CONC = os.environ.get('FN_MOE_HYB_FUSE') == '2'
_SIDE = {}

def _log(msg):
    print(f'[hyb_fuse] {msg}', flush=True)

@triton.jit
def _combine_add_kernel(c3_ptr, ids_ptr, map_ptr, y4_ptr, out_ptr, N, E, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = cols < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k).to(tl.int64)
        inr = (e >= 0) & (e < E)
        ok = inr & (tl.load(map_ptr + e, mask=inr, other=-1) >= 0)
        v = tl.load(c3_ptr + (t * TOPK + k) * N + cols, mask=cm & ok, other=0.0)
        acc += v.to(tl.float32)
    lb = acc.to(tl.bfloat16).to(tl.float32)
    m4 = tl.load(y4_ptr + t * N + cols, mask=cm, other=0.0).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t * N + cols, (m4 + lb).to(tl.bfloat16), mask=cm)

def combine_add(c3, topk_ids, expert_map, y4, M, topk, block=512):
    N = c3.shape[1]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=c3.device)
    if M == 0:
        return out
    assert y4.is_contiguous() and y4.shape == (M, N)
    _combine_add_kernel[M, triton.cdiv(N, block)](c3, topk_ids.contiguous(), expert_map, y4, out, N, expert_map.numel(), TOPK=topk, BLOCK=block, num_warps=4)
    return out

def _side_stream(dev):
    s = _SIDE.get(dev)
    if s is None:
        s = _SIDE[dev] = torch.cuda.Stream(device=dev)
    return s

def _lowbit_c3(lb, xh, xs1, topk_w, topk_ids, expert_map):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk
    M = xh.shape[0]
    topk = topk_ids.shape[1]
    cg, cd = (lk._tile('gu', M, lb.block_m), lk._tile('dn', M, lb.block_m))
    if not lk._TUNE and getattr(lb.gu, 'pairs', False):
        cg, cd = ((cg[0], 128, 4, 2), (cd[0], 128, 4, 2))
    al = {}
    for bm in {cg[0], cd[0]}:
        al[bm] = moe_align_block_size(topk_ids, bm, lb.E, expert_map, ignore_invalid_experts=expert_map is not None)
    c1 = torch.empty(M * topk, lb.gu.N, dtype=torch.bfloat16, device=xh.device)
    tw = topk_w.reshape(-1).float()
    mr = lambda bm: topk_ids.numel() + lb.gu.bits.numel() * (bm - 1)
    lb.gu(xh, c1, tw, *al[cg[0]], topk, True, False, *cg, xs=xs1, max_rows=mr(cg[0]), evict=lk._EVICT)
    c2, xs2 = lk.silu_prep_local(c1, topk_ids, expert_map)
    c3 = torch.empty(M * topk, lb.dn.N, dtype=torch.bfloat16, device=xh.device)
    lb.dn(c2, c3, tw, *al[cd[0]], 1, False, True, *cd, xs=xs2, max_rows=mr(cd[0]), evict=lk._EVICT)
    return c3

def _lowbit_plan(lb, xh, xs1, topk_w, topk_ids, expert_map):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk
    M = xh.shape[0]
    topk = topk_ids.shape[1]
    cg, cd = (lk._tile('gu', M, lb.block_m), lk._tile('dn', M, lb.block_m))
    if not lk._TUNE and getattr(lb.gu, 'pairs', False):
        cg, cd = ((cg[0], 128, 4, 2), (cd[0], 128, 4, 2))
    al = {bm: moe_align_block_size(topk_ids, bm, lb.E, expert_map, ignore_invalid_experts=True) for bm in {cg[0], cd[0]}}
    R, I = (M * topk, lb.gu.N // 2)
    ids = topk_ids.contiguous()
    return dict(M=M, topk=topk, cg=cg, cd=cd, al=al, xh=xh, xs1=xs1, ids=ids, emap=expert_map, tw=topk_w.reshape(-1).float(), c1=torch.empty(R, lb.gu.N, dtype=torch.bfloat16, device=xh.device), c2=torch.empty(R, I, dtype=torch.float16, device=xh.device), xs2=torch.empty(R, I // 128, dtype=torch.float32, device=xh.device), c3=torch.empty(R, lb.dn.N, dtype=torch.bfloat16, device=xh.device))

def _lowbit_launch(lb, p):
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk
    cg, cd, al, topk = (p['cg'], p['cd'], p['al'], p['topk'])
    mr = lambda bm: p['ids'].numel() + lb.gu.bits.numel() * (bm - 1)
    lb.gu(p['xh'], p['c1'], p['tw'], *al[cg[0]], topk, True, False, *cg, xs=p['xs1'], max_rows=mr(cg[0]), evict=lk._EVICT)
    R, I = p['c2'].shape
    if R:
        rows, gb = (2, triton.next_power_of_2(I // 128))
        lk._silu_prep_local_kernel[triton.cdiv(R, rows), triton.cdiv(I // 128, gb)](p['c1'], p['c2'], p['xs2'], p['ids'], p['emap'], R, I, p['emap'].numel(), HAS_MAP=True, ROWS=rows, GB=gb, num_warps=4)
    lb.dn(p['c2'], p['c3'], p['tw'], *al[cd[0]], 1, False, True, *cd, xs=p['xs2'], max_rows=mr(cd[0]), evict=lk._EVICT)

def _lowbit_body(lb, xh, xs1, topk_w, topk_ids, expert_map, y4):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk
    M = xh.shape[0]
    topk = topk_ids.shape[1]
    cg, cd = (lk._tile('gu', M, lb.block_m), lk._tile('dn', M, lb.block_m))
    if not lk._TUNE and getattr(lb.gu, 'pairs', False):
        cg, cd = ((cg[0], 128, 4, 2), (cd[0], 128, 4, 2))
    al = {}
    for bm in {cg[0], cd[0]}:
        al[bm] = moe_align_block_size(topk_ids, bm, lb.E, expert_map, ignore_invalid_experts=expert_map is not None)
    c1 = torch.empty(M * topk, lb.gu.N, dtype=torch.bfloat16, device=xh.device)
    tw = topk_w.reshape(-1).float()
    mr = lambda bm: topk_ids.numel() + lb.gu.bits.numel() * (bm - 1)
    lb.gu(xh, c1, tw, *al[cg[0]], topk, True, False, *cg, xs=xs1, max_rows=mr(cg[0]), evict=lk._EVICT)
    c2, xs2 = lk.silu_prep_local(c1, topk_ids, expert_map)
    c3 = torch.empty(M * topk, lb.dn.N, dtype=torch.bfloat16, device=xh.device)
    lb.dn(c2, c3, tw, *al[cd[0]], 1, False, True, *cd, xs=xs2, max_rows=mr(cd[0]), evict=lk._EVICT)
    return combine_add(c3, topk_ids, expert_map, y4, M, topk)

def install():
    if not ON:
        return False
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk
    if getattr(lk.HybridMoE.__call__, '_hyb_fuse', False):
        return True
    orig = lk.HybridMoE.__call__

    def call(self, x, topk_w, topk_ids, expert_map=None, tiles=None, lb2=None):
        M = x.shape[0] if x.dim() == 2 else 0
        if CHUNK and M > CHUNK and (tiles is None) and (not lb2):
            out = None
            for s in range(0, M, CHUNK):
                e = min(M, s + CHUNK)
                y = call(self, x[s:e], topk_w[s:e], topk_ids[s:e], expert_map)
                if out is None:
                    out = torch.empty((M,) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
                out[s:e] = y
            return out
        if not FUSE:
            return orig(self, x, topk_w, topk_ids, expert_map, tiles, lb2)
        lb = self.lb
        if lb is None or tiles is not None or lb2 or lk._LB2 or (not lk._FUSE_SILU) or (x.dim() != 2) or (x.dtype != torch.bfloat16) or (x.shape[0] == 0):
            return orig(self, x, topk_w, topk_ids, expert_map, tiles, lb2)
        xh, xs1 = lk.prep_x(x)
        if CONC and x.shape[0] <= CONC_MAX:
            plan = _lowbit_plan(lb, xh, xs1, topk_w, topk_ids, self.mapl)
            main = torch.cuda.current_stream()
            side = _side_stream(x.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                _lowbit_launch(lb, plan)
            y4 = self.m4(xh, topk_w, topk_ids, self.map4, self.E)
            main.wait_stream(side)
            return combine_add(plan['c3'], topk_ids, self.mapl, y4.contiguous(), x.shape[0], topk_ids.shape[1])
        y4 = self.m4(xh, topk_w, topk_ids, self.map4, self.E)
        return _lowbit_body(lb, xh, xs1, topk_w, topk_ids, self.mapl, y4.contiguous())
    call._hyb_fuse = True
    lk.HybridMoE.__call__ = call
    _log('HybridMoE: shared fp16 input, Marlin cast + part sum fused into the low-bit combine (FN_MOE_HYB_FUSE)' + (f'; low-bit part concurrent on a side stream (=2, M <= {CONC_MAX})' if CONC else '') + (f'; chunks of {CHUNK} tokens (FN_MOE_CHUNK)' if CHUNK else ''))
    return True
