import os
import torch
ON = os.environ.get('FN_GDN_BA_OVERLAP') == '1'
MAX_T = int(os.environ.get('FN_GDN_BA_OVERLAP_MAX_T', '512'))
_SIDE = {}

def _log(msg):
    print(f'[gdn_ba_overlap] {msg}', flush=True)

def _side(dev):
    s = _SIDE.get(dev)
    if s is None:
        s = _SIDE[dev] = torch.cuda.Stream(device=dev)
    return s

def wrap(layer):
    qkvz, ba = (getattr(layer, 'in_proj_qkvz', None), getattr(layer, 'in_proj_ba', None))
    if qkvz is None or ba is None or getattr(layer, '_fn_ba_overlap', False):
        return False
    q_fwd, b_fwd = (qkvz.forward, ba.forward)
    st = {'pending': None}

    def qkvz_forward(x, *a, **kw):
        st['pending'] = None
        if x.is_cuda and (not a) and (not kw) and (x.shape[0] <= MAX_T):
            main = torch.cuda.current_stream()
            side = _side(x.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                res = b_fwd(x)
            st['pending'] = (x, res, side)
        return q_fwd(x, *a, **kw)

    def ba_forward(x, *a, **kw):
        p, st['pending'] = (st['pending'], None)
        if p is not None and p[0] is x and (not a) and (not kw):
            _, res, side = p
            main = torch.cuda.current_stream()
            main.wait_stream(side)
            out = res[0] if isinstance(res, tuple) else res
            if torch.is_tensor(out):
                out.record_stream(main)
            return res
        return b_fwd(x, *a, **kw)
    qkvz.forward = qkvz_forward
    ba.forward = ba_forward
    layer._fn_ba_overlap = True
    return True

def install():
    if not ON:
        return False
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as G
    cls = G.QwenGatedDeltaNetAttention
    if getattr(cls.__init__, '_fn_ba', False):
        return True
    orig = cls.__init__
    n = {'n': 0}

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        if wrap(self):
            n['n'] += 1
            if n['n'] == 1:
                _log('in_proj_ba forked onto a side stream concurrently with in_proj_qkvz (FN_GDN_BA_OVERLAP)')
    init._fn_ba = True
    cls.__init__ = init
    return True
