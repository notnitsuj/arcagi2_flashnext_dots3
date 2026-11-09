import os
MAXS = 16

def _mode():
    return os.environ.get('FN_SESSION_RESUMEFIX', '1')

def commit_counts(sampled, n, tot, limit, stop, frozen):
    import torch
    ar = torch.arange(sampled.shape[1], device=n.device)
    live = ar[None, :] < n[:, None]
    hit = ((sampled.long()[:, :, None] == stop[:, None, :]) & (stop[:, None, :] >= 0)).any(-1) & live
    any_hit = hit.any(1)
    first = torch.where(any_hit, hit.int().argmax(1).long() + 1, n)
    cnt_len = (limit - tot).clamp(min=0)
    c = torch.minimum(n, torch.minimum(first, cnt_len))
    c = torch.where(frozen, torch.zeros_like(c), c)
    return (c, any_hit | (cnt_len <= n))
_K = {}

def _kernels():
    if _K:
        return _K
    from vllm.triton_utils import tl, triton

    @triton.jit
    def counts_kernel(idx_ptr, sampled_ptr, s_stride, n_ptr, tot_ptr, limit_ptr, stop_ptr, frozen_ptr, c_ptr, cnt_ptr, K1P: tl.constexpr, S: tl.constexpr, COUNT: tl.constexpr):
        b = tl.program_id(0)
        r = tl.load(idx_ptr + b).to(tl.int64)
        n = tl.load(n_ptr + b).to(tl.int64)
        if r < 0:
            tl.store(c_ptr + b, n.to(c_ptr.dtype.element_ty))
            return
        tot = tl.load(tot_ptr + r).to(tl.int64)
        lim = tl.load(limit_ptr + r).to(tl.int64)
        frz = tl.load(frozen_ptr + r).to(tl.int32)
        offs = tl.arange(0, K1P)
        toks = tl.load(sampled_ptr + b * s_stride + offs, mask=offs < n, other=-2).to(tl.int64)
        stops = tl.load(stop_ptr + r * S + tl.arange(0, S))
        hit2 = (toks[:, None] == stops[None, :]) & (stops[None, :] >= 0)
        hit = tl.max(hit2.to(tl.int32), axis=1)
        first = tl.min(tl.where(hit > 0, offs.to(tl.int64) + 1, n), axis=0)
        any_hit = tl.max(hit, axis=0)
        cnt_len = tl.maximum(lim - tot, 0)
        c = tl.minimum(n, tl.minimum(first, cnt_len))
        c = tl.where(frz > 0, 0, c)
        reach = (any_hit > 0) | (cnt_len <= n)
        tl.store(c_ptr + b, c.to(c_ptr.dtype.element_ty))
        tl.store(frozen_ptr + r, tl.where(reach | (frz > 0), 1, 0).to(tl.uint8))
        if COUNT:
            d = n - c
            tl.atomic_add(cnt_ptr + 0, tl.where((d > 0) & (frz == 0), 1, 0))
            tl.atomic_add(cnt_ptr + 1, tl.where((d > 0) & (frz > 0), 1, 0))
            tl.atomic_add(cnt_ptr + 2, tl.maximum(d, 0))
            tl.atomic_add(cnt_ptr + 3, 1)
            tl.atomic_add(cnt_ptr + 4, tl.where(reach & (frz == 0), 1, 0))

    @triton.jit
    def null_kernel(rows_ptr, idx_ptr, frozen_ptr, si_ptr, si_stride, null_id, HAS_ROWS: tl.constexpr):
        i = tl.program_id(0)
        row = i
        if HAS_ROWS:
            row = tl.load(rows_ptr + i)
        r = tl.load(idx_ptr + row).to(tl.int64)
        if r >= 0:
            if tl.load(frozen_ptr + r) > 0:
                tl.store(si_ptr + i * si_stride, null_id)
    _K.update(counts=counts_kernel, null=null_kernel)
    return _K

def install():
    m = _mode()
    if m == '0':
        return
    from vllm.v1.worker.gpu import model_runner as mr
    R = mr.GPUModelRunner
    if getattr(R, '_fn_resumefix', False):
        return
    R._fn_resumefix = True
    add0, post0 = (R.add_requests, R.postprocess_sampled)
    log_every = int(os.environ.get('FN_SESSION_RESUMEFIX_LOG', '0') or 0)

    def _state(self):
        st = self.__dict__.get('_fn_rf')
        if st is None:
            import torch
            n = self.req_states.max_num_reqs
            dev = self.device
            st = self._fn_rf = {'stop': torch.full((n + 1, MAXS), -1, dtype=torch.int64, device=dev), 'limit': torch.full((n + 1,), 1 << 30, dtype=torch.int64, device=dev), 'frozen': torch.zeros(n + 1, dtype=torch.uint8, device=dev), 'cnt': torch.zeros(5, dtype=torch.int32, device=dev), 'steps': 0, 'dummy': n, 'on': getattr(self.model_state, 'recoverssm', None) is not None}
            self.model_state._fn_frozen = st['frozen']
        return st

    def add_requests(self, scheduler_output):
        add0(self, scheduler_output)
        if not scheduler_output.scheduled_new_reqs:
            return
        import torch
        st = _state(self)
        for nr in scheduler_output.scheduled_new_reqs:
            i = self.req_states.req_id_to_index.get(nr.req_id)
            if i is None:
                continue
            sp = nr.sampling_params
            ids = []
            if sp is not None:
                if sp.eos_token_id is not None:
                    ids.append(int(sp.eos_token_id))
                ids += [int(x) for x in sp.stop_token_ids or ()]
            row = torch.full((MAXS,), -1, dtype=torch.int64)
            if len(ids) <= MAXS:
                row[:len(ids)] = torch.tensor(ids, dtype=torch.int64)
            mt = sp.max_tokens if sp is not None and sp.max_tokens is not None else 1 << 28
            lim = min(int(nr.prompt_len) + int(mt), int(self.max_model_len))
            st['stop'][i].copy_(row, non_blocking=True)
            st['limit'][i] = lim
            st['frozen'][i] = 0
            if os.environ.get('FN_SESSION_RESUMEFIX_DEBUG') == '1':
                print(f'[rfdbg] add {nr.req_id} idx {i} prompt_len {nr.prompt_len} computed {nr.num_computed_tokens} max_tokens {mt} lim {lim} stops {ids}', flush=True)

    def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
        st = _state(self)
        if not st['on'] or isinstance(num_sampled, int) or sampled_tokens is None or (sampled_tokens.ndim != 2):
            return post0(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
        import torch
        st['steps'] += 1
        dbg = os.environ.get('FN_SESSION_RESUMEFIX_DEBUG') == '1'
        if dbg:
            idx = idx_mapping.long()
            ok = idx >= 0
            idxc = torch.where(ok, idx, torch.full_like(idx, st['dummy']))
            tot = self.req_states.total_len.gpu[idxc.clamp(max=self.req_states.max_num_reqs - 1)].long()
            n = num_sampled.long()
            frz = st['frozen'][idxc].bool()
            c, reach = commit_counts(sampled_tokens, n, tot, st['limit'][idxc], st['stop'][idxc], frz)
            c = torch.where(ok, c, n)
            st['frozen'][idxc] = (frz | reach & ok).to(torch.uint8)
            st['frozen'][st['dummy']] = 0
            c = c.to(num_sampled.dtype)
            if self.device.index in (0, None):
                print(f'[rfdbg] step {st['steps']} idx {idx.tolist()} tot {tot.tolist()} n {n.tolist()} c {c.tolist()} frz {frz.tolist()}', flush=True)
        else:
            B = idx_mapping.shape[0]
            c = torch.empty_like(num_sampled)
            if B:
                k1 = sampled_tokens.shape[1]
                _kernels()['counts'][B,](idx_mapping, sampled_tokens, sampled_tokens.stride(0), num_sampled, self.req_states.total_len.gpu, st['limit'], st['stop'], st['frozen'], c, st['cnt'], K1P=max(2, 1 << (k1 - 1).bit_length()), S=MAXS, COUNT=bool(log_every), num_warps=1)
        if log_every and st['steps'] % log_every == 0:
            v = st['cnt'].tolist()
            print(f'[fnsession] resumefix({_mode()}): steps {st['steps']} rows {v[3]} stops {v[4]} capped {v[0]} frozen-skips {v[1]} extra tokens not folded {v[2]}', flush=True)
        if _mode() == 'measure':
            return post0(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
        ms = self.model_state
        orig = ms.postprocess_state
        ms.postprocess_state = lambda im, ns, nc=None: orig(im, c, nc)
        try:
            return post0(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
        finally:
            del ms.postprocess_state
    R.add_requests = add_requests
    R.postprocess_sampled = postprocess_sampled
    if m == 'measure':
        return
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    MS = mh.MambaHybridModelState
    pa0 = MS.prepare_attn

    def prepare_attn(self, input_batch, *a, **kw):
        out = pa0(self, input_batch, *a, **kw)
        frozen = getattr(self, '_fn_frozen', None)
        if frozen is None or kw.get('for_capture') or (len(a) >= 6 and a[5] is True):
            return out
        import torch
        seen = set()
        for md in out if isinstance(out, list) else [out]:
            if not isinstance(md, dict):
                continue
            for meta in md.values():
                c = getattr(meta, 'recoverssm_commit', None)
                if c is None or id(meta) in seen:
                    continue
                seen.add(id(meta))
                n = meta.num_spec_decodes
                if not n:
                    continue
                si = getattr(c, 'state_indices', None)
                if si is None:
                    continue
                v = si[:n, 0] if si.ndim == 2 else si[:n]
                ri = c.request_indices
                _kernels()['null'][n,](ri if ri is not None else v, input_batch.idx_mapping, frozen, v, v.stride(0), NULL_BLOCK_ID, HAS_ROWS=ri is not None, num_warps=1)
        return out
    MS.prepare_attn = prepare_attn
