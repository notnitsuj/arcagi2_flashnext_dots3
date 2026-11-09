"""TP-sharded DSA indexer keys (ARC_SHARD_INDEXER=1, with ARC_DOTS3_PORTABLE_DSA=1 and ARC_DOTS3_OFFLOAD=1).

The indexer's key cache (128 fp8 + 4 B scale per token per DSA layer) is replicated on every TP rank in vLLM. Here
each rank stores only the tokens it owns: owner(position) = position % P (P = TP size; with block_size % P == 0 this
is slot % P), local row = slot // P. vLLM's indexer cache spec is shrunk to 132/P bytes per token, so its tensor is
exactly this rank's [slots/P, 132] store and vLLM's capacity accounting stays truthful (4x more tokens at TP4).

Top-k: each rank scores its own positions (one Triton kernel for decode and prefill rows: per-row request block
table + visible length), takes a local top-K_LOC, all-gathers (value, position) over the TP group, and merges to
the global top-2048. Positions are interleaved, so the global top-2048 splits ~evenly over ranks; K_LOC = 640
(ARC_SHARD_KLOC) makes the merge exact unless one rank holds >640 of the global top-2048.
Output: topk_indices_buffer rows hold positions within the request (-1 = none), the format the offload attention
consumes for both prefill and decode.
"""
import os

import torch
import triton
import triton.language as tl

K_LOC = int(os.environ.get("ARC_SHARD_KLOC", "640"))
ROW_CHUNK = int(os.environ.get("ARC_SHARD_ROWS", "256"))
_PK = os.environ.get("ARC_SHARD_PK") == "1"   # request-packed FP8 decode logits (shard_logits_pk.py)
# ARC_SHARD_V2="grp,f8=0,BN=128,KB=2,w=4" (or "row8,..."): decode logits via shard_logits_v2 (picked by a startup gate)
_V2 = os.environ.get("ARC_SHARD_V2", "")
_V2_KIND = _V2.split(",")[0] if _V2 else ""
_V2_LOGGED = []
# ARC_SHARD_F16OUT=1 (with ARC_SHARD_V2): fp16 decode logits -> half the bytes and ~1.8x faster local top-k
_F16OUT = os.environ.get("ARC_SHARD_F16OUT") == "1"
_V2_ARGS = {k: int(v) for k, v in (x.split("=") for x in _V2.split(",")[1:])} if _V2 else {}
D = 128
ROWB = D + 4


@triton.jit
def _row_addr(base_ptr, row, RPB, STRIDE_BLK):
    """Byte address of local row `row` in vLLM's (possibly strided) per-layer indexer cache."""
    return base_ptr + (row // RPB) * STRIDE_BLK + (row % RPB) * 132


@triton.jit
def _shard_write_kernel(k_ptr, slot_ptr, base_ptr, rank, P, R_local, RPB, STRIDE_BLK, stride_k,
                        D: tl.constexpr, UE8M0: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    if slot % P != rank:
        return
    row = slot // P
    if row >= R_local:
        return
    d = tl.arange(0, D)
    v = tl.load(k_ptr + t * stride_k + d).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-4) / 448.0
    if UE8M0:
        sc = tl.exp2(tl.ceil(tl.log2(sc)))
    a = _row_addr(base_ptr, row, RPB, STRIDE_BLK)
    tl.store(a.to(tl.pointer_type(tl.float8e4nv)) + d, (v / sc).to(tl.float8e4nv))
    tl.store((a + 128).to(tl.pointer_type(tl.float32)), sc)


@triton.jit
def _shard_logits_kernel(q_ptr, w_ptr, base_ptr, bt_ptr, req_ptr, ctx_ptr, out_ptr,
                         rank, P, BS, R_local, RPB, STRIDE_BLK, stride_q_row, stride_q_h, stride_w_row, stride_bt,
                         stride_out, Lc,
                         H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    j0 = tl.program_id(1) * BN
    ctx = tl.load(ctx_ptr + row)
    j = j0 + tl.arange(0, BN)
    pos = j * P + rank
    valid = pos < ctx
    if j0 * P + rank >= ctx:
        tl.store(out_ptr + row * stride_out + j, float("-inf"), mask=j < Lc)
        return
    req = tl.load(req_ptr + row).to(tl.int64)
    blk = tl.load(bt_ptr + req * stride_bt + pos // BS, mask=valid, other=0).to(tl.int64)
    lrow = (blk * BS + pos % BS) // P
    valid = valid & (blk >= 0) & (lrow < R_local)          # dummy/capture batches carry arbitrary block ids
    lrow = tl.where(valid, lrow, 0)
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    a = _row_addr(base_ptr, lrow, RPB, STRIDE_BLK)
    k = tl.load(a.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float16)
    sc = tl.load((a + 128).to(tl.pointer_type(tl.float32)), mask=valid, other=0.0)
    q = tl.load(q_ptr + row * stride_q_row + h[:, None] * stride_q_h + d[None, :]).to(tl.float16)
    s = tl.maximum(tl.dot(k, tl.trans(q)), 0.0)                    # [BN, H]
    w = tl.load(w_ptr + row * stride_w_row + h)
    logit = tl.sum(s * w[None, :], axis=1) * sc
    tl.store(out_ptr + row * stride_out + j, tl.where(valid, logit, float("-inf")), mask=j < Lc)


class ShardStore:
    """This rank's key store inside vLLM's (shrunk, possibly strided) indexer cache tensor [num_blocks, BS, 132/P]:
    local row r = slot // P lives in block r // (BS/P) at byte (r % (BS/P)) * 132 (4 consecutive owned slots share
    a block since BS % P == 0). No reshape/copy: kernels address the strided view directly."""

    def __init__(self, kv_cache, P, rank):
        kc = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
        if kc.dim() == 4:
            kc = kc.squeeze(2) if kc.shape[2] == 1 else kc.squeeze(1)
        self.kc, self.P, self.rank = kc, P, rank
        self.BS = kc.shape[1]
        assert kc.dtype == torch.uint8 and kc.shape[2] * P == ROWB and kc.stride(2) == 1 and kc.stride(1) == kc.shape[2], \
            (kc.shape, kc.stride(), kc.dtype)
        self.RPB = self.BS // P
        self.STRIDE_BLK = kc.stride(0)          # bytes (uint8)
        self.R_local = kc.shape[0] * self.RPB
        self.base = kc

    def write(self, k, slots, ue8m0):
        n = min(k.shape[0], slots.shape[0])
        if n:
            _shard_write_kernel[(n,)](k, slots, self.base, self.rank, self.P, self.R_local, self.RPB, self.STRIDE_BLK,
                                      k.stride(0), D=D, UE8M0=ue8m0)

    def logits(self, q, w, block_table, req, ctx, max_ctx, group=1, noinf=False):
        """q fp8 [rows, H, D], w [rows, H], req/ctx int32 [rows] -> compact logits [rows, Lc] (col j = pos j*P+rank).
        noinf (shard_logits_v2 kernels only): tiles wholly beyond the rows' contexts are left unwritten."""
        rows, H, _ = q.shape
        Lc = (max_ctx + self.P - 1) // self.P
        v2 = bool(rows and _V2_KIND)   # any rows: MTP verify rows arrive as 4-token "prefill" chunks here
        out = torch.empty(rows, max(Lc, 1), device=q.device, dtype=torch.float16 if v2 and _F16OUT else torch.float32)
        BN = 64
        if v2:
            from . import shard_logits_v2
            a = _V2_ARGS
            if not _V2_LOGGED:
                _V2_LOGGED.append(1); print(f"[shard_v2] decode logits {_V2}", flush=True)
            return shard_logits_v2.logits(_V2_KIND, self, q, w, block_table, req, ctx, out, max(Lc, 1), G=a.get("G", 4),
                                          BN=a.get("BN", 128), KB=a.get("KB", 1), warps=a.get("w", 4),
                                          stages=a.get("st", 2), f8=bool(a.get("f8", 0)), noinf=noinf)
        if rows and _PK and group > 1 and rows % group == 0:   # request-packed FP8 kernel (shard_logits_pk)
            from . import shard_logits_pk
            return shard_logits_pk.logits(self, q, w, block_table, req, ctx, out, max(Lc, 1), group)
        if rows:
            _shard_logits_kernel[(rows, triton.cdiv(max(Lc, 1), BN))](
                q, w, self.base, block_table, req, ctx, out, self.rank, self.P, self.BS, self.R_local, self.RPB,
                self.STRIDE_BLK,
                q.stride(0), q.stride(1), w.stride(0), block_table.stride(0), out.stride(0), max(Lc, 1),
                H=H, D=D, BN=BN, num_warps=4)
        return out


def _host_ag_fn():
    if os.environ.get("ARC_HOST_AG", "1") != "1":
        return None
    from . import host_ar
    h = host_ar.get("tp")
    return h.all_gather0 if h is not None else None


_FAST = []


def _ag_u16(kl, Lc):
    """arcstats.ag_pack if ARC_AG_U16=1 and this payload qualifies (kl even, local columns fit uint16), else None"""
    if os.environ.get("ARC_AG_U16", "0") != "1":
        return None
    from . import ag_pack
    return ag_pack if ag_pack.usable(kl, Lc) else None


def _fast_level():
    """ARC_FAST_TOPK (fast_topk.py): 0 torch.topk, 1 fast local select, 2 + fast merge; needs ctx int32 and the
    loadable library (else 0, logged once by fast_topk)."""
    if not _FAST:
        lvl = 0
        if os.environ.get("ARC_FAST_TOPK", "0") not in ("", "0"):
            from . import fast_topk
            lvl = int(fast_topk.MODE) if fast_topk.lib() is not None else 0
            print(f"[fast_topk] sharded indexer top-k level {lvl}", flush=True)
        _FAST.append(lvl)
    return _FAST[0]


def sharded_topk(store, q, w, block_table, req, ctx, max_ctx, topk_out, topk_tokens, group=1):
    """Fill topk_out [rows, topk_tokens] (int32 positions, -1 = none) from this rank's keys + TP all-gather merge."""
    from vllm.distributed import tensor_model_parallel_all_gather
    _host_ag = _host_ag_fn()
    P = store.P
    rows = q.shape[0]
    fast = _fast_level()
    for r0 in range(0, rows, ROW_CHUNK):
        r1 = min(rows, r0 + ROW_CHUNK)
        lg = store.logits(q[r0:r1], w[r0:r1], block_table, req[r0:r1], ctx[r0:r1], max_ctx,
                          group if r0 % group == 0 else 1, noinf=fast >= 1)
        kl = min(K_LOC, lg.shape[1])
        if fast >= 1:   # exact radix select over the valid columns, straight into the all-gather payload
            from . import fast_topk
            pk = fast_topk.local(lg, ctx[r0:r1], store.rank, P, kl)
        else:
            v, j = torch.topk(lg, kl, dim=1)
            v = v.float()
            pos = (j * P + store.rank).to(torch.int32)
            pk = torch.cat([v, pos.view(torch.float32)], 1)            # [n, 2*kl]
        n = r1 - r0
        u16 = _ag_u16(kl, lg.shape[1])
        if u16 is not None:   # ARC_AG_U16: [values | uint16 local ids] on the wire, unpacked bit-exactly below
            pk = u16.pack(pk, P, store.rank)
        allp = _host_ag(pk.contiguous()) if _host_ag else None
        if allp is None:
            allp = tensor_model_parallel_all_gather(pk.contiguous(), 0)  # [P*n, 2*kl]
        if u16 is not None:
            allp = u16.unpack(allp, n, P)
        if fast >= 2 and topk_out.dtype == torch.int32 and topk_out.stride(1) == 1:
            # sorted merge written straight into topk_out (-1 for +-inf values and for columns >= kk)
            from . import fast_topk
            fast_topk.merge(allp.contiguous(), n, P, kl, min(topk_tokens, P * kl), topk_out[r0:r1, :topk_tokens])
            continue
        allp = allp.view(P, n, 2 * kl).transpose(0, 1)               # [n, P, 2kl]
        vals = allp[:, :, :kl].reshape(n, P * kl)
        poss = allp[:, :, kl:].contiguous().view(torch.int32).reshape(n, P * kl)
        kk = min(topk_tokens, P * kl)
        tv, ti = torch.topk(vals, kk, dim=1)
        sel = torch.gather(poss, 1, ti)
        sel = torch.where(torch.isinf(tv), torch.full_like(sel, -1), sel)
        topk_out[r0:r1, :kk] = sel
        if kk < topk_tokens:
            topk_out[r0:r1, kk:topk_tokens] = -1


CHECK = os.environ.get("ARC_SHARD_CHECK") == "1"
_full = {}
_stats = {"n": 0, "recall": 0.0, "minrec": 1.0}


class _FullStore(ShardStore):
    """Debug: replicated full-context key store over the same slot address space (P=1, contiguous)."""

    def __init__(self, kc, slots):
        self.P, self.rank, self.BS = 1, 0, kc.shape[1]
        self.RPB, self.R_local = self.BS, slots
        self.base = torch.zeros(slots * ROWB, dtype=torch.uint8, device=kc.device)
        self.STRIDE_BLK = self.BS * ROWB
        self.kc = kc


def _check_store(name, kc):
    if name not in _full:
        _full[name] = _FullStore(kc, kc.shape[0] * kc.shape[1])
    return _full[name]


def _check_decode(name, kc, im, k, q, w, bt, req, ctx, max_ctx, got, topk_tokens, ue8m0):
    fs = _check_store(name, kc)
    if k is not None:
        fs.write(k[: im.slot_mapping.shape[0]], im.slot_mapping, ue8m0)
    lg = fs.logits(q, w, bt, req, ctx, max_ctx)
    from vllm.distributed import get_tensor_model_parallel_rank as _gr
    sh = _check_decode.shard_store.logits(q, w, bt, req, ctx, max_ctx)
    own = lg[:, _gr()::_check_decode.shard_store.P][:, : sh.shape[1]]
    fin = torch.isfinite(own) & torch.isfinite(sh)
    if fin.any():
        d = float((own[fin] - sh[fin]).abs().max())
        _stats["maxdiff"] = max(_stats.get("maxdiff", 0.0), d)
    _stats["finmis"] = _stats.get("finmis", 0) + int((torch.isfinite(own) != torch.isfinite(sh)).sum())
    kk = min(topk_tokens, lg.shape[1])
    v, ref = torch.topk(lg, kk, dim=1)
    ref = torch.where(torch.isinf(v), torch.full_like(ref, -1), ref)
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    P_, rk = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    g = got[got >= 0]
    _stats["own"] = _stats.get("own", 0) + int((g % P_ == rk).sum()); _stats["tot"] = _stats.get("tot", 0) + int(g.numel())
    rr = ref[ref >= 0]
    _stats["rown"] = _stats.get("rown", 0) + int((rr % P_ == rk).sum()); _stats["rtot"] = _stats.get("rtot", 0) + int(rr.numel())
    for r in range(q.shape[0]):
        a = set(ref[r][ref[r] >= 0].tolist()); b = set(got[r][got[r] >= 0].tolist())
        rec = len(a & b) / max(1, len(a))
        _stats["n"] += 1; _stats["recall"] += rec; _stats["minrec"] = min(_stats["minrec"], rec)
    if _stats["n"] % 2000 < q.shape[0]:
        from vllm.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() == 0:
            print(f"[shardcheck] rows={_stats['n']} mean recall={_stats['recall']/_stats['n']:.5f} "
                  f"min={_stats['minrec']:.4f} ctx_max={int(ctx.max())} sharded-own-frac={_stats['own']/max(1,_stats['tot']):.3f} "
                  f"ref-own-frac={_stats['rown']/max(1,_stats['rtot']):.3f} own-logit maxdiff={_stats.get('maxdiff',0):.4g} "
                  f"finite-mismatch={_stats.get('finmis',0)}", flush=True)


def install(sai, ram, idx_mod):
    """Patch vLLM: shrink the indexer cache spec to 132/P bytes per token, and replace the portable indexer flow."""
    from dataclasses import replace
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models import deepseek_v2 as dv2

    orig_spec = dv2.DeepseekV32IndexerCache.get_kv_cache_spec

    def spec(self, vllm_config):
        s = orig_spec(self, vllm_config)
        P = vllm_config.parallel_config.tensor_parallel_size
        assert s.head_size % P == 0, (s.head_size, P)
        # vLLM puts all 13 indexer layers in one group whose page is padded up to the largest page (139264 B per
        # 64-token block here), so shrinking bytes/token alone saves nothing. A larger indexer block (default
        # 64*20/P tokens: 320 at TP4) fills that padded page with sharded rows: 13*320*33 B ~= 139264 B.
        bs = int(os.environ.get("ARC_SHARD_BS", str(s.block_size * 20 // P)))
        assert bs % P == 0, (bs, P)
        return replace(s, head_size=s.head_size // P, block_size=bs)
    dv2.DeepseekV32IndexerCache.get_kv_cache_spec = spec
    # our kernels address keys by (block, offset) for any block size: let manager blocks be kernel blocks
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
    from vllm.v1.attention.backend import MultipleOf
    DeepseekV32IndexerBackend.get_supported_kernel_block_sizes = staticmethod(lambda: [MultipleOf(4)])

    stores = {}
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    @eager_break_during_capture   # like vLLM's indexer op: runs eagerly between captured graph segments
    def indexer(hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights, quant_block_size, scale_fmt,
                topk_tokens, head_dim, max_model_len, total_seq_lens, topk_indices_buffer, *a, **kw):
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return topk_indices_buffer
        from vllm.utils.torch_utils import _resolve_layer_name
        name = _resolve_layer_name(k_cache_prefix)
        im = md[name]
        P, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        st = stores.get(name)
        kc = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
        if st is None or st.kc.data_ptr() != kc.data_ptr():
            st = stores[name] = ShardStore(kv_cache, P, rank)
        n_tok = im.slot_mapping.shape[0]
        topk_indices_buffer[: hidden_states.shape[0]] = -1
        if k is not None and not kw.get("skip_k_cache_insert", False):
            st.write(k[:n_tok], im.slot_mapping, scale_fmt == "ue8m0")
            if CHECK and im.num_decode_tokens == 0:
                _check_store(name, kc).write(k[:n_tok], im.slot_mapping, scale_fmt == "ue8m0")
        nd = im.num_decode_tokens
        if im.num_prefills > 0:
            for ch in im.prefill.chunks:
                ks, ke = ch.cu_seqlen_ks, ch.cu_seqlen_ke
                req = (torch.searchsorted(ch.cu_seq_lens, ks.to(ch.cu_seq_lens.dtype), right=True) - 1).to(torch.int32)
                ctx = (ke - ks).to(torch.int32)
                max_ctx = int(ch.max_local_total_seq_lens or ch.total_seq_lens)
                sharded_topk(st, q_fp8[ch.token_start:ch.token_end], weights[ch.token_start:ch.token_end],
                             ch.block_table, req, ctx, max_ctx,
                             topk_indices_buffer[ch.token_start:ch.token_end], topk_tokens)
        if nd > 0:
            dec = im.decode
            if dec.requires_padding:
                raise NotImplementedError("sharded indexer: padded (non-uniform) spec-decode batches")
            B = dec.block_table.shape[0]
            next_n = nd // max(1, B)
            sl = dec.seq_lens
            if sl.dim() == 2:
                ctx = sl.reshape(-1)[:nd].to(torch.int32)
            else:
                jj = torch.arange(nd, device=sl.device) % next_n
                ctx = (sl.repeat_interleave(next_n)[:nd] - (next_n - 1 - jj)).to(torch.int32)
            req = (torch.arange(nd, device=sl.device) // next_n).to(torch.int32)
            sharded_topk(st, q_fp8[:nd], weights[:nd], dec.block_table, req, ctx, max_model_len,
                         topk_indices_buffer[:nd], topk_tokens, group=next_n)
            if CHECK:
                _check_decode.shard_store = st
                _check_decode(name, kc, im, k, q_fp8[:nd], weights[:nd], dec.block_table, req, ctx, max_model_len,
                              topk_indices_buffer[:nd], topk_tokens, scale_fmt == "ue8m0")
        return topk_indices_buffer

    return indexer
