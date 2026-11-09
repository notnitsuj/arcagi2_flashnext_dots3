import json
import os
_ST = {'done': False, 'marlin': {}, 'bf16': {}}

def _log(msg):
    print(f'[l4cfg] {msg}', flush=True)

def _pick(table, m):
    for row in table:
        if m <= row[0]:
            return row[1:]
    return None

def register():
    f = os.environ.get('FN_MOE_L4CFG')
    if not f or _ST['done']:
        return
    _ST['done'] = True
    d = json.load(open(f))
    _ST['marlin'] = {tuple((int(v) for v in k.split('x'))): sorted(v) for k, v in d.get('marlin', {}).items()}
    _ST['bf16'] = {tuple((int(v) for v in k.split('x'))): sorted(v) for k, v in d.get('bf16mm', {}).items()}
    if _ST['marlin']:
        _install_marlin()
    if _ST['bf16']:
        _install_bf16()
    if d.get('lowbit'):
        try:
            from . import lowbit_kernels as lk
        except ImportError:
            import lowbit_kernels as lk
        if os.environ.get('FN_MOE_TUNE'):
            _log('FN_MOE_TUNE set: lowbit section ignored')
        else:
            lk._TUNE = {k: sorted(((int(m), tuple(c)) for m, c in v.items())) for k, v in d['lowbit'].items()}
    _log(f'FN_MOE_L4CFG={f}: marlin {sorted(_ST['marlin'])} bf16mm {sorted(_ST['bf16'])} lowbit {sorted(d.get('lowbit', {}))}')

def marlin_cfg(size_m, size_n, size_k):
    t = _ST['marlin'].get((size_n, size_k))
    return _pick(t, size_m) if t else None

def _install_marlin():
    from vllm import _custom_ops as ops
    orig = ops.moe_wna16_marlin_gemm
    if getattr(orig, '_k2', False):
        return

    def moe_wna16_marlin_gemm(*args, **kw):
        c = marlin_cfg(kw.get('size_m'), kw.get('size_n'), kw.get('size_k'))
        if c is not None and kw.get('thread_k', -1) == -1 and (kw.get('blocks_per_sm', -1) == -1):
            kw['thread_k'], kw['thread_n'], kw['blocks_per_sm'] = (int(c[0]), int(c[1]), int(c[2]))
        return orig(*args, **kw)
    moe_wna16_marlin_gemm._k2 = True
    ops.moe_wna16_marlin_gemm = moe_wna16_marlin_gemm

def bf16_cfg(M, K, N):
    t = _ST['bf16'].get((K, N))
    return _pick(t, M) if t else None

def _install_bf16():
    import torch
    from vllm.model_executor.layers import linear as lin
    try:
        from . import bf16mm
    except ImportError:
        import bf16mm
    orig = lin.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if bias is None and x.dim() == 2 and (x.dtype == torch.bfloat16) and (w.dtype == torch.bfloat16) and (x.stride(1) == 1) and (w.stride(1) == 1) and (x.shape[0] >= 1):
            c = bf16_cfg(x.shape[0], w.shape[1], w.shape[0])
            if c is not None:
                return bf16mm.mm(x, w, tuple((int(v) for v in c)))
        return orig(self, layer, x, bias)
    lin.UnquantizedLinearMethod.apply = apply
