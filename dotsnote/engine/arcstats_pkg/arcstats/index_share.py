"""IndexCache-style top-k sharing across DSA layers (arXiv 2603.12201) + top-k overlap statistics. Debug/eval only.

ARC_INDEX_SHARE=<pattern> | @<file>
    One char per DSA layer in model order (Dots3: layers 0,1,5,9,...,45 = 13 chars): 'F' = run the layer's own indexer,
    'S' = skip it and reuse the top-k of the most recent F layer (the model-level topk_indices_buffer is shared by all
    DSA layers, so a skipped indexer simply leaves the previous layer's indices in place). The first DSA layer is always
    F. '@file' re-reads the file when its mtime changes (eager servers: score several patterns on one server).
    A skipped layer also skips its indexer-key cache insert (its indexer K cache is never read).

ARC_TOPK_OVERLAP=<dir>
    Prefill-only: per query token, |topk_a & topk_b| / topk for every DSA layer pair, binned by context length
    (number of visible keys). Accumulated over the server lifetime; written to <dir>/overlap_rank<r>.npz after each step.
"""
import os
import re

import numpy as np
import torch

BINS = [0, 2048, 4096, 8192, 16384, 32768, 1 << 30]


def _layer_id(prefix):
    m = re.search(r"layers\.(\d+)\.", prefix)
    return int(m.group(1)) if m else -1


_DSA = []   # backbone layer ids that own an indexer, filled at indexer construction (config is only set then)


def _dsa_layers():
    return sorted(set(_DSA)) or None


def _register(prefix):
    from vllm.config import get_current_vllm_config
    lid = _layer_id(prefix)
    nl = getattr(get_current_vllm_config().model_config.hf_config, "num_hidden_layers", 1 << 30)
    if 0 <= lid < nl:   # MTP/nextn layers keep their own indexer
        _DSA.append(lid)


class IndexShare:
    def __init__(self, spec):
        self.spec = spec
        self._file = spec[1:] if spec.startswith("@") else None
        self._mtime = None
        self._skip = set()
        self._dsa = None
        if self._file is None:
            self._pattern = spec.strip()

    def _parse(self, pattern):
        if self._dsa is None:
            self._dsa = _dsa_layers()
        p = pattern.strip().upper()
        if self._dsa is None or len(p) != len(self._dsa) or set(p) - {"F", "S"}:
            raise ValueError(f"ARC_INDEX_SHARE pattern {pattern!r}: need {len(self._dsa or [])} chars of F/S")
        self._skip = {lid for lid, c in zip(self._dsa[1:], p[1:]) if c == "S"}
        print(f"[index_share] pattern={p} skip_layers={sorted(self._skip)}", flush=True)

    def skip(self, lid):
        if self._file is not None:
            try:
                mt = os.stat(self._file).st_mtime_ns
            except OSError:
                mt = None
            if mt != self._mtime:
                self._mtime = mt
                if mt is None:
                    self._skip = set()
                else:
                    self._parse(open(self._file).read())
        elif self._dsa is None:
            self._parse(self._pattern)
        return lid in self._skip


class TopkOverlap:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self._dsa = None
        self._cur = {}
        self.sum_rec = None

    def record(self, lid, topk_buf, md, topk):
        if self._dsa is None:
            self._dsa = _dsa_layers()
            L = len(self._dsa)
            nb = len(BINS) - 1
            self.sum_rec = np.zeros((nb, L, L))
            self.sum_jac = np.zeros((nb, L, L))
            self.sum_rec_prev = np.zeros((nb, L))      # vs nearest previous DSA layer (diagonal-1 duplicate, convenience)
            self.cnt = np.zeros(nb)
        if md is None or md.num_prefills == 0 or md.prefill is None:
            self._cur = {}
            return
        nd = md.num_decode_tokens
        if lid == self._dsa[0]:
            self._cur = {}
        rows, ctx = [], []
        for ch in md.prefill.chunks:
            rows.append(topk_buf[ch.token_start:ch.token_end, :topk])
            n = ch.token_end - ch.token_start
            ctx.append((ch.cu_seqlen_ke[:n] - ch.cu_seqlen_ks[:n]).long())
        t = torch.cat(rows)
        self._cur[lid] = (torch.sort(t, dim=1).values, torch.cat(ctx))
        if lid != self._dsa[-1] or len(self._cur) != len(self._dsa):
            return
        self._accumulate(topk)
        self._cur = {}

    @torch.no_grad()
    def _accumulate(self, topk):
        L = len(self._dsa)
        ctx = self._cur[self._dsa[0]][1]
        bins = torch.bucketize(ctx, torch.tensor(BINS[1:-1], device=ctx.device), right=False)
        nb = len(BINS) - 1
        onehot = torch.nn.functional.one_hot(bins, nb).double()            # [n, nb]
        self.cnt += onehot.sum(0).cpu().numpy()
        S = [self._cur[l][0] for l in self._dsa]
        for a in range(L):
            for b in range(a + 1, L):
                sa, sb = S[a], S[b]
                pos = torch.searchsorted(sa, sb).clamp_(max=sa.shape[1] - 1)
                hit = ((sa.gather(1, pos) == sb) & (sb >= 0)).sum(1).double()
                na = (sa >= 0).sum(1).double().clamp_min(1)
                nb_ = (sb >= 0).sum(1).double().clamp_min(1)
                rec = hit / nb_                                              # recall of b's set by a's set
                jac = hit / (na + nb_ - hit)
                r = (onehot * rec[:, None]).sum(0).cpu().numpy()
                j = (onehot * jac[:, None]).sum(0).cpu().numpy()
                self.sum_rec[:, a, b] += r
                self.sum_rec[:, b, a] += r
                self.sum_jac[:, a, b] += j
                self.sum_jac[:, b, a] += j
        r = 0
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            r = get_tensor_model_parallel_rank()
        except Exception:
            pass
        np.savez(os.path.join(self.outdir, f"overlap_rank{r}.npz"), bins=np.array(BINS), dsa=np.array(self._dsa),
                 cnt=self.cnt, sum_rec=self.sum_rec, sum_jac=self.sum_jac)


def wrap(forward_cuda):
    """Wrap SparseAttnIndexer.forward_cuda with sharing / overlap stats (only installed when an env flag is set)."""
    from vllm.forward_context import get_forward_context
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    _init = sai.SparseAttnIndexer.__init__

    def init(self, *a, **kw):
        _init(self, *a, **kw)
        _register(self.k_cache.prefix)
        _CACHES[self.k_cache.prefix] = self.k_cache

    sai.SparseAttnIndexer.__init__ = init
    share = IndexShare(os.environ["ARC_INDEX_SHARE"]) if os.environ.get("ARC_INDEX_SHARE") else None
    _SHARE[0] = share
    if share is not None and share._file is None and os.environ.get("ARC_INDEX_SHARE_FREE") == "1":
        _install_free()
    ovl = TopkOverlap(os.environ["ARC_TOPK_OVERLAP"]) if os.environ.get("ARC_TOPK_OVERLAP") else None

    def fwd(self, hidden_states, q_quant, k, weights):
        lid = getattr(self, "_arc_lid", None)
        if lid is None:
            lid = self._arc_lid = _layer_id(self.k_cache.prefix)
        md = get_forward_context().attn_metadata
        real = isinstance(md, dict)
        if share is not None and real and share.skip(lid):
            return self.topk_indices_buffer
        out = forward_cuda(self, hidden_states, q_quant, k, weights)
        if ovl is not None and real:
            ovl.record(lid, self.topk_indices_buffer, md.get(self.k_cache.prefix), self.topk_tokens)
        return out

    return fwd


# ---- ARC_INDEX_SHARE_FREE=1 (static patterns only): skipped layers get NO indexer-key KV; their offload path uses the
# source F layer's indexer metadata / slot mapping / slot space (all indexer caches are one KV group -> same slots).
_SHARE = [None]
_CACHES = {}   # indexer k_cache prefix -> DeepseekV32IndexerCache module


def source_prefix(prefix):
    """k_cache prefix of the F layer whose top-k (and slot space) a skipped layer reuses; None if not skipped."""
    share = _SHARE[0]
    lid = _layer_id(prefix)
    if share is None or share._file is not None or not share.skip(lid):
        return None
    src = max(l for l in share._dsa if l < lid and l not in share._skip)
    return re.sub(r"layers\.\d+\.", f"layers.{src}.", prefix, count=1)


def source_cache(prefix):
    sp = source_prefix(prefix)
    return _CACHES[sp] if sp else None


def _install_free():
    import vllm.model_executor.models.deepseek_v2 as dv2
    orig = dv2.DeepseekV32IndexerCache.get_kv_cache_spec

    def spec(self, vllm_config):
        if source_prefix(self.prefix) is not None:
            return None            # never read: no KV allocated
        return orig(self, vllm_config)
    dv2.DeepseekV32IndexerCache.get_kv_cache_spec = spec
