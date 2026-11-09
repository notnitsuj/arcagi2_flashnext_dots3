import os
import torch
from vllm.triton_utils import tl, triton
ON = os.environ.get('FN_GDN_LAZY') == '1'
DEBUG = int(os.environ.get('FN_GDN_LAZY_DEBUG', '0'))
LBV = int(os.environ.get('FN_GDN_LAZY_BV', '0'))
LWARPS = int(os.environ.get('FN_GDN_LAZY_WARPS', '0'))
_DBG = {'v': 0, 'c': 0}
_REG = {}
_GROUPS = []

@triton.jit
def _verify_lazy_kernel(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, state_ptr, replay_ptr, out_ptr, query_start_loc_ptr, state_indices_ptr, scale, softplus_beta, softplus_threshold, null_block_id, stride_q_token, stride_k_token, stride_v_token, stride_a_token, stride_b_token, stride_state_block, stride_state_head, stride_state_v, stride_state_k, stride_replay_block, stride_replay_head, stride_replay_pos, stride_replay_dim, stride_out_token, stride_qsl, stride_si, pending_ptr, cnt_ptr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SPEC_QUERY_LEN: tl.constexpr, USE_QK_L2NORM: tl.constexpr, NVT: tl.constexpr):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hv = tl.program_id(2)
    pid_h = pid_hv // (HV // H)
    bos = tl.load(query_start_loc_ptr + pid_b * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_qsl).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    if state_idx <= null_block_id:
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(out_ptr + (bos + t) * stride_out_token + pid_hv * V + offs_v, tl.zeros([BV], dtype=tl.float32), mask=(t < query_len) & mask_v)
        return
    sp_off = state_idx * stride_state_block + pid_hv * stride_state_head + offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k
    state = tl.load(state_ptr + sp_off, mask=mask_state, other=0.0).to(tl.float32)
    rec_base = replay_ptr + state_idx * stride_replay_block + pid_hv * stride_replay_head
    n = tl.load(pending_ptr + state_idx)
    if n > 0:
        h = state
        for t in range(n):
            rec = rec_base + t * stride_replay_pos
            c = tl.load(rec + offs_v * stride_replay_dim, mask=mask_v, other=0.0)
            kk = tl.load(rec + (V + offs_k) * stride_replay_dim, mask=mask_k, other=0.0)
            decay = tl.load(rec + (V + K) * stride_replay_dim)
            h = tl.fma(c[:, None], kk[None, :], h * decay)
        hb = h.to(state_ptr.dtype.element_ty)
        tl.store(state_ptr + sp_off, hb, mask=mask_state)
        state = hb.to(tl.float32)
    cidx = state_idx * HV + pid_hv
    arrived = tl.atomic_add(cnt_ptr + cidx, 1, sem='acq_rel', scope='gpu')
    last = arrived == NVT - 1
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_hv).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        tok = bos + t
        q = tl.load(q_ptr + tok * stride_q_token + pid_h * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + tok * stride_k_token + pid_h * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + tok * stride_v_token + pid_hv * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
        x = tl.load(a_ptr + tok * stride_a_token + pid_hv, mask=valid, other=0.0).to(tl.float32) + dt_bias
        sp = tl.where(softplus_beta * x <= softplus_threshold, 1 / softplus_beta * tl.log(1 + tl.exp(softplus_beta * x)), x)
        g = neg_A * sp
        beta = tl.sigmoid(tl.load(b_ptr + tok * stride_b_token + pid_hv, mask=valid, other=0.0).to(tl.float32))
        if USE_QK_L2NORM:
            q = q * tl.rsqrt(tl.sum(q * q) + 1e-06)
            k = k * tl.rsqrt(tl.sum(k * k) + 1e-06)
        q = q * scale
        decay = tl.exp(g)
        new_state = state * decay
        corr = (v - tl.sum(new_state * k[None, :], 1)) * beta
        new_state = new_state + corr[:, None] * k[None, :]
        state = tl.where(valid, new_state, state)
        o = tl.sum(state * q[None, :], 1)
        tl.store(out_ptr + tok * stride_out_token + pid_hv * V + offs_v, o.to(out_ptr.dtype.element_ty), mask=valid & mask_v)
        rec = rec_base + t * stride_replay_pos
        tl.store(rec + offs_v * stride_replay_dim, corr, mask=valid & mask_v)
        tl.store(rec + (V + offs_k) * stride_replay_dim, k, mask=valid & mask_k & last)
        tl.store(rec + (V + K) * stride_replay_dim, decay, mask=valid & last)
    if last:
        tl.store(cnt_ptr + cidx, 0)

@triton.jit
def _set_pending_kernel(state_indices_ptr, commit_lens_ptr, pending_ptr, null_block_id, stride_si):
    i = tl.program_id(0)
    src = tl.load(state_indices_ptr + i * stride_si).to(tl.int64)
    if src > null_block_id:
        tl.store(pending_ptr + src, tl.load(commit_lens_ptr + i))

def _group_of(t):
    return _REG.get(id(t))

class _VerifyProxy:

    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, grid):

        def launch(*args, **kw):
            g = _group_of(args[7])
            if DEBUG and _DBG['v'] < DEBUG and (not torch.cuda.is_current_stream_capturing()):
                _DBG['v'] += 1
                print(f'[gdn_lazy] verify grid {grid} group {('yes' if g else 'NO')} ptr {args[7].data_ptr():x} pending>0 {(int((g['pending'] > 0).sum()) if g else -1)} si {args[11][:4].tolist()}', flush=True)
            if g is None or g.get('align'):
                return self.orig[grid](*args, **kw)
            gr = grid
            if LBV:
                kw['BV'] = LBV
                gr = (triton.cdiv(kw['V'], LBV),) + tuple(grid[1:])
            if LWARPS:
                kw['num_warps'] = LWARPS
            return _verify_lazy_kernel[gr](*args, g['pending'], g['cnt'], NVT=gr[0], **kw)
        return launch

class _CommitProxy:

    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, grid):

        def launch(*args, **kw):
            g = _group_of(args[0])
            if DEBUG and _DBG['c'] < DEBUG:
                _DBG['c'] += 1
                print(f'[gdn_lazy] commit grid {grid} group {('yes' if g else 'NO')} lens {args[7][:4].tolist()} si {args[6][:4].tolist()} boundary {kw.get('BOUNDARY')}', flush=True)
            if g is None or kw.get('BOUNDARY') or g.get('align'):
                return self.orig[grid](*args, **kw)
            state_indices, commit_lens, null_id, stride_si = (args[6], args[7], args[11], args[18])
            _set_pending_kernel[grid[1],](state_indices, commit_lens, g['pending'], null_id, stride_si, num_warps=1)
        return launch

def flush_rows(g, idx):
    from vllm.model_executor.layers.mamba.gdn import recoverssm_gdn as R
    if g is None or g.get('align') or idx is None or (idx.numel() == 0):
        return
    idx = idx.reshape(-1).to(torch.int32)
    safe = idx.clamp(min=0).long()
    pending = g['pending']
    lens = torch.where(idx > 0, pending[safe], torch.zeros_like(idx))
    ctx = g['ctx']
    ref, rr = (ctx.checkpoints[0], ctx.replays[0])
    _, HV, V, K = ref.shape
    BV = min(triton.next_power_of_2(V), 16)
    zero = torch.zeros_like(idx)
    R._commit_gdn_state_kernel.orig[triton.cdiv(V, BV), idx.shape[0], len(ctx.checkpoints) * HV](ref, ctx.state_base_addrs, ctx.state_block_strides, rr, ctx.replay_base_addrs, ctx.replay_block_strides, idx, lens, idx, zero, zero, R.NULL_BLOCK_ID, ref.stride(1), ref.stride(2), ref.stride(3), rr.stride(1), rr.stride(2), rr.stride(3), idx.stride(0), K=K, V=V, BK=triton.next_power_of_2(K), BV=BV, NUM_HEADS=HV, BOUNDARY=False, num_warps=4, num_stages=2)
    pending[safe] = 0
    g['flushes'] = g.get('flushes', 0) + 1

def register_context(ctx, align=False):
    ref = ctx.checkpoints[0]
    nb, HV = (ref.shape[0], ref.shape[1])
    dev = ref.device
    g = dict(ctx=ctx, pending=torch.zeros(nb, dtype=torch.int32, device=dev), cnt=torch.zeros(nb * HV, dtype=torch.int32, device=dev), blocks=torch.arange(nb, dtype=torch.int32, device=dev), align=align)
    g['keep'] = list(ctx.checkpoints)
    for c in ctx.checkpoints:
        _REG[id(c)] = g
    _GROUPS.append(g)
    return g

def patch():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.mamba.gdn import recoverssm_gdn as R
    from vllm.v1.attention.backends import gdn_recoverssm as B
    log = init_logger('vllm.fnext.gdn_lazy')
    R._gdn_recoverssm_verify_kernel = _VerifyProxy(R._gdn_recoverssm_verify_kernel)
    R._commit_gdn_state_kernel = _CommitProxy(R._commit_gdn_state_kernel)
    orig_create = R.GDNRecoverSSMCommitContext.create.__func__

    def create(cls, layers, *, spec_query_len, max_num_reqs):
        ctx = orig_create(cls, layers, spec_query_len=spec_query_len, max_num_reqs=max_num_reqs)
        register_context(ctx)
        if DEBUG:
            for c, r in zip(ctx.checkpoints, ctx.replays):
                print(f'[gdn_lazy] ctx ckpt ptr {c.data_ptr():x} shape {tuple(c.shape)} stride {c.stride()} replay ptr {r.data_ptr():x} stride {r.stride()}', flush=True)
        log.info('FN_GDN_LAZY: lazy GDN commit for %d layers (%d blocks)', len(ctx.checkpoints), ctx.checkpoints[0].shape[0])
        return ctx
    R.GDNRecoverSSMCommitContext.create = classmethod(create)
    orig_commit = R.GDNRecoverSSMCommitContext.commit

    def commit(self, *a, **kw):
        if kw.get('block_table') is not None:
            g = _group_of(self.checkpoints[0])
            if g is not None and (not g['align']):
                raise RuntimeError('FN_GDN_LAZY does not support mamba align mode (prefix caching)')
        return orig_commit(self, *a, **kw)
    R.GDNRecoverSSMCommitContext.commit = commit
    orig_build = B.GDNRecoverSSMMetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, *a, **kw):
        meta = orig_build(self, common_prefix_len, common_attn_metadata, *a, **kw)
        if not getattr(self, '_capturing', False) and self._context is not None and (meta.num_prefills > 0):
            flush_rows(_group_of(self._context.checkpoints[0]), meta.non_spec_state_indices_tensor)
        return meta
    B.GDNRecoverSSMMetadataBuilder.build = build
    log.info('FN_GDN_LAZY: installed')
