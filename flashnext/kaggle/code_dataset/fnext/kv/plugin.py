import json
import os
FP8_MAX = 448.0
_HIST_LO, _HIST_HI, _HIST_BINS = (-24.0, 16.0, 80)

def _log(msg):
    try:
        from vllm.logger import init_logger
        init_logger('fnext.kv').info(msg)
    except Exception:
        print('[fnext.kv]', msg, flush=True)

def _rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get('RANK', '0'))

def _layer_idx(name):
    import re
    m = re.search('layers\\.(\\d+)\\.', name)
    return int(m.group(1)) if m else -1

def _idx_key(name):
    return ('mtp' in name.split('layers.')[0], _layer_idx(name))

def _tp_head(attn):
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        tp, r = (get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank())
    except Exception:
        return None
    total = attn.total_num_kv_heads
    if attn.num_kv_heads != 1:
        return None
    if total >= tp:
        return r
    return r // (tp // total)

def _set_scale(attn, ks, vs):
    import torch
    for nm, s in (('k', ks), ('v', vs)):
        s = float(s)
        if not (s > 0 and s < float('inf')):
            raise ValueError(f'bad {nm}_scale {s} for {attn.layer_name}')
        getattr(attn, f'_{nm}_scale').fill_(s)
        setattr(attn, f'_{nm}_scale_float', s)
        cpu = getattr(attn, f'_{nm}_scale_cpu', None)
        if isinstance(cpu, torch.Tensor):
            cpu.fill_(s)

def _qsa_mod():
    import importlib
    importlib.import_module('vllm.models.qwen4_exp.nvidia.model')
    return importlib.import_module('vllm.models.qwen4_exp.nvidia.qsa')

def _qsa_modules(model):
    Qwen4ExpQSAAttention = _qsa_mod().Qwen4ExpQSAAttention
    return [m for m in model.modules() if isinstance(m, Qwen4ExpQSAAttention)]

def apply_scales(model, path):
    side = json.load(open(path))
    layers = side.get('layers', side)
    per_head = os.environ.get('FN_KV_SCALES_PER_HEAD') == '1'
    by_idx = {_idx_key(k): v for k, v in layers.items() if isinstance(v, dict)}
    n, miss = (0, [])
    for attn in _qsa_modules(model):
        if attn.kv_cache_dtype not in ('fp8', 'fp8_e4m3'):
            continue
        e = layers.get(attn.layer_name) or by_idx.get(_idx_key(attn.layer_name))
        if e is None:
            miss.append(attn.layer_name)
            continue
        ks, vs = (e['k_scale'], e['v_scale'])
        h = _tp_head(attn) if per_head else None
        if h is not None and 'k_scale_heads' in e:
            ks, vs = (e['k_scale_heads'][h], e['v_scale_heads'][h])
        _set_scale(attn, ks, vs)
        n += 1
    if miss:
        raise RuntimeError(f'FN_KV_SCALES {path}: no scale for QSA layers {miss}')
    _log(f'FN_KV_SCALES: set static fp8 k/v scales on {n} QSA layers from {path} (per_head={per_head})')
_CAL = {'layers': {}, 'first': None, 'calls': 0}

def _cal_record(layer, key, value, slot_mapping):
    import torch
    n = slot_mapping.shape[0]
    ok = slot_mapping[:n] >= 0
    st = _CAL['layers'].get(layer.layer_name)
    H = key.shape[1]
    if st is None:
        dev = key.device
        st = {'kmax': torch.zeros(H, device=dev), 'vmax': torch.zeros(H, device=dev), 'khist': torch.zeros(_HIST_BINS, device=dev, dtype=torch.float64), 'vhist': torch.zeros(_HIST_BINS, device=dev, dtype=torch.float64), 'ntok': torch.zeros((), device=dev, dtype=torch.float64), 'head': _tp_head(layer)}
        _CAL['layers'][layer.layer_name] = st
        if _CAL['first'] is None:
            _CAL['first'] = layer.layer_name
    for nm, x in (('k', key), ('v', value)):
        x = x[:n].float().abs() * ok.view(-1, 1, 1)
        st[nm + 'max'] = torch.maximum(st[nm + 'max'], x.amax(dim=(0, 2)))
        lx = torch.log2(x.clamp_min(2.0 ** (_HIST_LO - 1))).flatten()
        st[nm + 'hist'] += torch.histc(lx, bins=_HIST_BINS, min=_HIST_LO, max=_HIST_HI).double()
    st['ntok'] += ok.sum()
    if layer.layer_name == _CAL['first']:
        _CAL['calls'] += 1
        if _CAL['calls'] % int(os.environ.get('FN_KV_CALIB_EVERY', '64')) == 0:
            _cal_dump()

def _cal_dump():
    if not _CAL['layers']:
        return
    out = {'format': 'fn-kv-calib-v1', 'rank': _rank(), 'calls': _CAL['calls'], 'hist': {'lo': _HIST_LO, 'hi': _HIST_HI, 'bins': _HIST_BINS}, 'layers': {}}
    for name, st in _CAL['layers'].items():
        out['layers'][name] = {'head': st['head'], 'ntok': float(st['ntok'].item()), 'kmax': st['kmax'].tolist(), 'vmax': st['vmax'].tolist(), 'khist': st['khist'].tolist(), 'vhist': st['vhist'].tolist()}
    p = f'{os.environ['FN_KV_CALIB']}.r{_rank()}.json'
    tmp = p + '.tmp'
    json.dump(out, open(tmp, 'w'))
    os.replace(tmp, p)

def _install_calib():
    import atexit
    Q = _qsa_mod()
    orig_init = Q.Qwen4ExpQSAAttention.__init__
    orig_upd = Q.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.use_fused_qsa_prepare = False

    def upd(self, layer, key, value, kv_cache, slot_mapping):
        _cal_record(layer, key, value, slot_mapping)
        return orig_upd(self, layer, key, value, kv_cache, slot_mapping)
    Q.Qwen4ExpQSAAttention.__init__ = init
    Q.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update = upd
    atexit.register(_cal_dump)
    _log(f'FN_KV_CALIB: recording QSA K/V absmax to {os.environ['FN_KV_CALIB']}.r*.json (unfused prepare)')

def _install_scales(path):
    from vllm.model_executor.model_loader import base_loader, utils
    orig = utils.process_weights_after_loading

    def pwal(model, model_config, target_device):
        orig(model, model_config, target_device)
        if _qsa_modules(model):
            apply_scales(model, path)
    utils.process_weights_after_loading = pwal
    base_loader.process_weights_after_loading = pwal
    try:
        from vllm.model_executor.model_loader.weight_cache import ipc_loader
        ipc_loader.process_weights_after_loading = pwal
    except Exception:
        pass
_TIE_NEWEST = os.environ.get('FN_KV_TOPK_TIE', 'oldest') == 'newest'

def det_topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace=None):
    import torch
    k = token_topk // compress_ratio
    rows, width = logits.shape
    if rows == 0:
        return
    vis = visible_blocks[:rows].to(torch.int64)
    col = torch.arange(width, device=logits.device)
    masked = logits.float().masked_fill(col[None, :] >= vis[:, None], float('-inf'))
    if _TIE_NEWEST:
        order = width - 1 - torch.sort(masked.flip(1), dim=1, descending=True, stable=True).indices[:, :k]
        order = order.to(torch.int32)
    else:
        order = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :k].to(torch.int32)
    if order.shape[1] < k:
        order = torch.cat([order, order.new_full((rows, k - order.shape[1]), -1)], dim=1)
    valid = torch.arange(k, device=logits.device)[None, :] < vis[:, None]
    block_indices.copy_(torch.where(valid, order, torch.full_like(order, -1)))

def _install_eps_topk(mode):
    import importlib
    import torch
    _qsa_mod()
    ix = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
    orig = ix._topk
    ramps = {}
    maxw = int(os.environ.get('FN_KV_TOPK_TIE_MAXW', str(1 << 18)))

    def ramp_for(dev):
        r = ramps.get(dev)
        if r is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('FN_KV_TOPK_TIE_EPS ramp must be built before CUDA-graph capture')
            idx = torch.arange(maxw, device=dev, dtype=torch.float32)
            r = idx - maxw if mode == 'newest' else -1.0 - idx
            ramps[dev] = r
        return r

    def eps_topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
        r = ramp_for(logits.device)[:logits.shape[1]]
        logits.copy_(torch.where(logits == 0, r, logits))
        return orig(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace)
    ix._topk = eps_topk
    _log(f'FN_KV_TOPK_TIE_EPS={mode}: deterministic zero-tie break before the stock top-k')

def key_topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace=None, newest=True):
    import torch
    k = token_topk // compress_ratio
    rows, width = logits.shape
    if rows == 0:
        return
    dev = logits.device
    vis = visible_blocks[:rows].to(torch.int64)
    col = torch.arange(width, device=dev, dtype=torch.int64)
    bits = logits.float().contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits < 0, torch.zeros_like(bits), bits)
    tie = col if newest else width - 1 - col
    key = bits << 32 | tie[None, :]
    key = key.masked_fill(col[None, :] >= vis[:, None], -1)
    kk = min(k, width)
    sel = torch.topk(key, kk, dim=1, sorted=False).indices
    sel = torch.sort(sel, dim=1).values.to(torch.int32)
    if kk < k:
        sel = torch.cat([sel, sel.new_full((rows, k - kk), -1)], dim=1)
    short = vis < k
    ar = torch.arange(k, device=dev, dtype=torch.int32)[None, :].expand(rows, k)
    short_out = torch.where(ar < vis[:, None].to(torch.int32), ar, torch.full_like(ar, -1))
    block_indices.copy_(torch.where(short[:, None], short_out, sel))

def _install_key_topk(mode):
    import importlib
    _qsa_mod()
    ix = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
    newest = mode == 'newest'

    def f(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
        return key_topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, newest=newest)
    ix._topk = f
    _log(f'FN_KV_TOPK_KEY={mode}: deterministic int64-key QSA top-k (ties -> {mode})')
_TAG_KERNEL = None

def _tag_kernel():
    global _TAG_KERNEL
    if _TAG_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _tag(x_ptr, stride_r, width, NB: tl.constexpr, NEWEST: tl.constexpr, BLOCK: tl.constexpr):
            r = tl.program_id(0)
            c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            m = c < width
            p = x_ptr + r.to(tl.int64) * stride_r + c
            b = tl.load(p, mask=m, other=0.0).to(tl.int32, bitcast=True)
            if NEWEST:
                t = c
                zv = (c - width).to(tl.float32)
            else:
                t = width - 1 - c
                zv = (-1 - c).to(tl.float32)
            tagged = (b & ~((1 << NB) - 1) | t).to(tl.float32, bitcast=True)
            out = tl.where(b == 0, zv, tagged)
            tl.store(p, out, mask=m)
        _TAG_KERNEL = _tag
    return _TAG_KERNEL

def tag_scores_(logits, newest=True):
    import triton
    rows, width = logits.shape
    if rows == 0:
        return logits
    assert logits.dtype.itemsize == 4 and logits.stride(1) == 1
    nb = max(1, (width - 1).bit_length())
    BLOCK = 1024
    _tag_kernel()[rows, triton.cdiv(width, BLOCK)](logits, logits.stride(0), width, NB=nb, NEWEST=newest, BLOCK=BLOCK)
    return logits

def sort_block_indices_(bi):
    import torch
    if bi.numel() == 0:
        return bi
    big = torch.iinfo(bi.dtype).max
    v = torch.sort(torch.where(bi < 0, torch.full_like(bi, big), bi), dim=1).values
    bi.copy_(torch.where(v == big, torch.full_like(v, -1), v))
    return bi

def _install_tag_topk(mode):
    import importlib
    _qsa_mod()
    ix = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
    orig = ix._topk
    newest = mode == 'newest'
    sort_idx = os.environ.get('FN_KV_TOPK_SORT', '1') == '1'

    def f(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
        tag_scores_(logits, newest)
        r = orig(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace)
        if sort_idx:
            sort_block_indices_(block_indices)
        return r
    ix._topk = f
    _log(f'FN_KV_TOPK_TAG={mode}: deterministic recency tie-break (score tagging) before the stock top-k')

def _install_det_topk():
    import importlib
    _qsa_mod()
    ix = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
    ix._topk = det_topk
    _log('FN_KV_DET_TOPK: deterministic QSA indexer top-k (test only)')

def _install_budget_dump(path):
    import dataclasses
    from vllm.v1.core import kv_cache_utils as KU
    from vllm.v1.engine import core as EC
    orig = KU.get_kv_cache_configs

    def wrapped(vllm_config, *a, **k):
        cfgs = orig(vllm_config, *a, **k)
        try:
            out = {'max_model_len': vllm_config.model_config.max_model_len, 'num_spec': vllm_config.num_speculative_tokens, 'tp': vllm_config.parallel_config.tensor_parallel_size, 'use_recoverssm': bool(getattr(vllm_config.cache_config, 'use_recoverssm', False)), 'cache_dtype': str(vllm_config.cache_config.cache_dtype), 'mamba_ssm_cache_dtype': str(getattr(vllm_config.cache_config, 'mamba_ssm_cache_dtype', '')), 'skip_layers': [str(x) for x in getattr(vllm_config.cache_config, 'kv_cache_dtype_skip_layers', [])], 'workers': []}
            for c in cfgs:
                w = {'num_blocks': c.num_blocks, 'tensor_bytes': sum((t.size for t in c.kv_cache_tensors)), 'num_tensors': len(c.kv_cache_tensors), 'groups': []}
                for g in c.kv_cache_groups:
                    sp = g.kv_cache_spec
                    f = {}
                    if dataclasses.is_dataclass(sp):
                        for fld in dataclasses.fields(sp):
                            v = getattr(sp, fld.name)
                            f[fld.name] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                    w['groups'].append({'spec': type(sp).__name__, 'n_layers': len(g.layer_names), 'layers': g.layer_names[:4], 'block_size': sp.block_size, 'page_size_bytes': sp.page_size_bytes, 'max_bytes_per_req': sp.max_memory_usage_bytes(vllm_config), 'eagle': bool(getattr(g, 'is_eagle_group', False)), 'fields': f})
                out['workers'].append(w)
            json.dump(out, open(path, 'w'), indent=1)
            _log(f'FN_KV_BUDGET_DUMP: wrote {path}')
        except Exception as e:
            _log(f'FN_KV_BUDGET_DUMP failed: {e!r}')
        return cfgs
    KU.get_kv_cache_configs = wrapped
    EC.get_kv_cache_configs = wrapped
_DONE = False

def register():
    global _DONE
    if _DONE:
        return
    _DONE = True
    sc, cal = (os.environ.get('FN_KV_SCALES'), os.environ.get('FN_KV_CALIB'))
    if os.environ.get('FN_KV_BUDGET_DUMP'):
        _install_budget_dump(os.environ['FN_KV_BUDGET_DUMP'])
    if os.environ.get('FN_KV_PACK_STATE') == '1':
        from . import state_pack
        state_pack.install()
    if os.environ.get('FN_KV_DET_TOPK') == '1':
        _install_det_topk()
    elif os.environ.get('FN_KV_TOPK_TAG') in ('newest', 'oldest'):
        _install_tag_topk(os.environ['FN_KV_TOPK_TAG'])
    elif os.environ.get('FN_KV_TOPK_KEY') in ('newest', 'oldest'):
        _install_key_topk(os.environ['FN_KV_TOPK_KEY'])
    elif os.environ.get('FN_KV_TOPK_TIE_EPS') in ('newest', 'oldest'):
        _install_eps_topk(os.environ['FN_KV_TOPK_TIE_EPS'])
    if not sc and (not cal):
        return
    if sc and cal:
        raise RuntimeError('FN_KV_SCALES and FN_KV_CALIB are exclusive')
    if sc:
        json.load(open(sc))
        _install_scales(sc)
    else:
        _install_calib()
