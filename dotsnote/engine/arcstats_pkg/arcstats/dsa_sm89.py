"""Portable DeepSeek-style sparse attention (DSA) for Dots3 on GPUs without SM90 kernels (e.g. L4/SM89).

vLLM's CUDA DSA path needs DeepGEMM (indexer logits) and FlashAttention/FlashMLA sparse kernels, all
SM90+. This module supplies:
  * Triton indexer-logit kernels (paged decode + contiguous prefill) that reduce over index heads in
    registers, replacing the per-row PyTorch references (host syncs, [H, M, N] temporaries);
  * the reference (ROCm/portable) indexer flow wired to those kernels;
  * a Triton sparse-MLA attention backend (vLLM's XPU kernel) over Dots3's padded cache rows.
Enable with ARC_DOTS3_PORTABLE_DSA=1 (and without ARC_DOTS3_DENSE).
"""
import os
_VERIFY = os.environ.get("ARC_OFFLOAD_VERIFY") == "1"   # debug: keep full GPU KV and compare offload vs paged attention
_CUDA_PF_SPLIT = int(os.environ.get("ARC_OFFLOAD_CUDA_PF_SPLIT", "1"))   # prefill: many query tokens, few splits needed
# ARC_OFFLOAD_PF_COMPACT=N (default 0 = off): an offload prefill chunk whose context exceeds N rows stages only the distinct
# rows its top-k actually selects (one host sync), not all `total_seq_lens` context rows. Bit-exact (same rows, same order
# per query). The full staging is an unbudgeted [tot, 576] fp8 transient: a forced conclusion pushed onto a ~240k-token
# session stages ~136 MiB (+ ~140 MB of host reads per layer) and OOMed the L4 engine (long-session
# test). With N set, such a chunk stages <= min(tot, query tokens x top-k) rows.
_PF_COMPACT = int(os.environ.get("ARC_OFFLOAD_PF_COMPACT", "0") or 0)
# ARC_OFFLOAD_PF_OUT=1 (default off, bit-exact): a step with prefill tokens writes the decode
# and every prefill chunk's attention straight into one [n, heads, 512] output instead of per-part outputs + torch.cat
# (-32 MiB of transient GPU memory at 1024 tokens; same kernels, same values)
_PF_OUT = os.environ.get("ARC_OFFLOAD_PF_OUT", "0") == "1"
import torch


def compact_stage(rows, sidx):
    """rows [tot] int64 host row ids of a prefill chunk's context, sidx [M, K] int32 positions into them (-1 = none)
    -> (sel [U] int64 host row ids of the distinct selected positions (sorted), sidx2 [M, K] int32 into sel).
    Attention over (gather(sel), sidx2) reads exactly the rows (gather(rows), sidx) reads, in the same order."""
    v = sidx >= 0
    u, inv = torch.unique(sidx[v], sorted=True, return_inverse=True)
    sidx2 = torch.full_like(sidx, -1)
    sidx2[v] = inv.to(sidx.dtype)
    return rows[u.long()], sidx2
import triton

_DEBUG = os.environ.get("ARC_DSA_DEBUG") == "1"
import triton.language as tl


# ---------------------------------------------------------------- indexer logits kernels
@triton.jit
def _paged_logits_kernel(q_ptr, w_ptr, kv8_ptr, kv32_ptr, bt_ptr, sl_ptr, out_ptr,
                         stride_q_row, stride_q_h, stride_w_row, stride_blk8, stride_blk32,
                         stride_bt, stride_out, next_n,
                         SL_2D: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    row = tl.program_id(0)
    pg = tl.program_id(1)
    b = row // next_n
    j = row % next_n
    if SL_2D:
        ctx = tl.load(sl_ptr + b * next_n + j)
    else:
        ctx = tl.load(sl_ptr + b) - (next_n - 1 - j)
    start = pg * BS
    if start >= ctx:
        return
    page = tl.load(bt_ptr + b * stride_bt + pg).to(tl.int64)
    t = tl.arange(0, BS)
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    # vLLM's indexer cache write uses the "SHUFFLE" layout: 16-token x 16-byte tiles
    koff = (t[:, None] // 16) * (16 * D) + (t[:, None] % 16) * 16 + (d[None, :] // 16) * 256 + d[None, :] % 16
    k = tl.load(kv8_ptr + page * stride_blk8 + koff).to(tl.float16)
    sc = tl.load(kv32_ptr + page * stride_blk32 + (BS * D) // 4 + t)
    q = tl.load(q_ptr + row * stride_q_row + h[:, None] * stride_q_h + d[None, :]).to(tl.float16)
    s = tl.dot(k, tl.trans(q))                       # [BS, H]
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + row * stride_w_row + h)
    logit = tl.sum(s * w[None, :], axis=1) * sc      # [BS]
    tl.store(out_ptr + row * stride_out + start + t, logit, mask=(start + t) < ctx)


def paged_mqa_logits_triton(q, kv_cache, weights, seq_lens, block_table, schedule_metadata=None,
                            max_model_len=None):
    """q [B, next_n, H, D] fp8; kv_cache [blocks, bs, 1, D+4] uint8 (values then fp32 scales per block);
    weights [B*next_n, H] fp32 -> logits [B*next_n, max_model_len] fp32 (-inf outside context)."""
    B, next_n, H, D = q.shape
    kv = kv_cache.reshape(kv_cache.shape[0], -1)     # [blocks, bs*(D+4)] bytes
    bs = kv_cache.shape[1]
    kv8 = kv.view(torch.float8_e4m3fn)
    kv32 = kv.view(torch.float32)
    rows = B * next_n
    out = torch.full((rows, max_model_len), float("-inf"), device=q.device, dtype=torch.float32)
    qf = q.reshape(rows, H, D)
    sl = seq_lens.contiguous()
    grid = (rows, triton.cdiv(max_model_len, bs))
    _paged_logits_kernel[grid](qf, weights, kv8, kv32, block_table, sl, out,
                               qf.stride(0), qf.stride(1), weights.stride(0), kv8.stride(0), kv32.stride(0),
                               block_table.stride(0), out.stride(0), next_n,
                               SL_2D=sl.dim() == 2, H=H, D=D, BS=bs, num_warps=4)
    return out


@triton.jit
def _mqa_logits_kernel(q_ptr, w_ptr, k_ptr, s_ptr, ks_ptr, ke_ptr, out_ptr, N,
                       stride_q_m, stride_q_h, stride_w_m, stride_out_m,
                       H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    n = n0 + tl.arange(0, BN)
    in_n = n < N
    if (n0 >= ke) | (n0 + BN <= ks):
        tl.store(out_ptr + m * stride_out_m + n, float("-inf"), mask=in_n)
        return
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    k = tl.load(k_ptr + n[:, None] * D + d[None, :], mask=in_n[:, None], other=0.0).to(tl.float16)
    q = tl.load(q_ptr + m * stride_q_m + h[:, None] * stride_q_h + d[None, :]).to(tl.float16)
    s = tl.maximum(tl.dot(k, tl.trans(q)), 0.0)       # [BN, H]
    w = tl.load(w_ptr + m * stride_w_m + h)
    sc = tl.load(s_ptr + n, mask=in_n, other=0.0)
    logit = tl.sum(s * w[None, :], axis=1) * sc
    valid = (n >= ks) & (n < ke)
    tl.store(out_ptr + m * stride_out_m + n, tl.where(valid, logit, float("-inf")), mask=in_n)


def mqa_logits_triton(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """q [M, H, D] fp8; kv = (k_fp8 [N, D], k_scale [N] or [N,1] fp32); weights [M, H] -> [M, N]."""
    k_fp8, k_scale = kv
    M, H, D = q.shape
    N = k_fp8.shape[0]
    out = torch.empty((M, N), device=q.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return out
    BN = 64
    grid = (M, triton.cdiv(N, BN))
    _mqa_logits_kernel[grid](q, weights, k_fp8.contiguous(), k_scale.reshape(-1).contiguous(),
                             cu_seqlen_ks, cu_seqlen_ke, out, N,
                             q.stride(0), q.stride(1), weights.stride(0), out.stride(0),
                             H=H, D=D, BN=BN, num_warps=4)
    return out


# ---------------------------------------------------------------- vLLM wiring
def install():
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    import vllm.v1.attention.backends.mla.indexer as idx
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ram
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.workspace import current_workspace_manager
    from vllm.platforms import current_platform

    # Indexer prefill K-gather buffer = max_model_len * 40 tokens (x132 B): 1.3 GiB/GPU at 262k context, meant for
    # many full-length prefills per chunk. A smaller multiple only splits prefill into more chunks.
    mult = int(os.environ.get("ARC_INDEXER_PREFILL_MULT", "0"))
    if mult:
        from vllm.model_executor.models import deepseek_v2 as _dv2
        _f = lambda vllm_config: vllm_config.model_config.max_model_len * mult
        idx.get_max_prefill_buffer_size = _f
        if hasattr(_dv2, "get_max_prefill_buffer_size"):
            _dv2.get_max_prefill_buffer_size = _f

    # Dots3's DSA layer first builds a stock MLAAttention (FlashMLA-sparse on Hopper, fp8_ds_mla) before it is
    # replaced by the padded portable layer; that impl reserves a 5 x max_model_len BF16 prefill workspace
    # (1.5 GiB at 262k) in the never-shrinking shared workspace. The portable path never uses it.
    from vllm.v1.attention.backends.mla import flashmla_sparse as _fms
    _fms.get_prefill_workspace_size = lambda max_model_len: 8192

    # indexer construction passes; metadata builder must not use DeepGEMM scheduling metadata
    sai.has_deep_gemm = lambda: True
    idx.has_deep_gemm = lambda: False
    ram.rocm_fp8_mqa_logits = mqa_logits_triton
    ram.rocm_fp8_paged_mqa_logits = paged_mqa_logits_triton
    ram._get_aiter_top_k_kernel = lambda **kw: None
    orig = ram.rocm_aiter_sparse_attn_indexer

    def indexer(hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights, quant_block_size, scale_fmt,
                topk_tokens, head_dim, max_model_len, total_seq_lens, topk_indices_buffer, *a, **kw):
        if not isinstance(get_forward_context().attn_metadata, dict):
            # profiling/dummy pass: reserve only the prefill K-gather workspace (no [H, rows, L] buffer)
            current_workspace_manager().get_simultaneous(
                ((total_seq_lens, head_dim), current_platform.fp8_dtype()),
                ((total_seq_lens, 4), torch.uint8))
            return topk_indices_buffer
        return orig(hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights, quant_block_size, scale_fmt,
                    topk_tokens, head_dim, max_model_len, total_seq_lens, topk_indices_buffer, *a, **kw)

    if os.environ.get("ARC_SWA_FP8") == "1":
        from .sparse_offload import install_fp8_gather
        install_fp8_gather()

    if os.environ.get("ARC_SHARD_INDEXER") == "1":
        from .shard_indexer import install as _shard_install
        indexer = _shard_install(sai, ram, idx)

    def forward_cuda(self, hidden_states, q_quant, k, weights):
        q = q_quant[0] if isinstance(q_quant, tuple) else q_quant
        return indexer(hidden_states, sai._encode_layer_name(self.k_cache.prefix), self.k_cache.kv_cache, q, k,
                       weights, self.quant_block_size, self.scale_fmt, self.topk_tokens, self.head_dim,
                       self.max_model_len, self.max_total_seq_len, self.topk_indices_buffer,
                       skip_k_cache_insert=self.skip_k_cache_insert)

    if os.environ.get("ARC_INDEX_SHARE") or os.environ.get("ARC_TOPK_OVERLAP"):
        from .index_share import wrap as _share_wrap   # IndexCache top-k sharing / overlap stats (eval only)
        forward_cuda = _share_wrap(forward_cuda)

    sai.SparseAttnIndexer.forward_cuda = forward_cuda
    _install_sparse_backend()


def _install_sparse_backend():
    from vllm.models.dots3_note.nvidia import attention as da
    from vllm.models.dots3_note.nvidia import model as dm
    from vllm.v1.attention.backends.mla import xpu_mla_sparse as xs
    from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
    from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index

    orig_backend = da.Dots3NotePaddedSparseBackend
    if getattr(orig_backend, "_arc_portable", False):  # already installed in this process
        return

    class PortableSparseImpl(xs.XPUMLASparseImpl):
        masked_mha_available = False  # FA sparse builder: no masked-MHA prefill path here
        def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype,
                     logits_soft_cap, attn_type, kv_sharing_target_layer_name, topk_indices_buffer=None,
                     indexer=None, **mla_args):
            igb = mla_args.pop("index_group_builder", None)
            self._indexer_mod = indexer
            self._offload = os.environ.get("ARC_DOTS3_OFFLOAD") == "1"
            self._off = None
            super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype,
                             logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                             topk_indices_buffer=topk_indices_buffer, indexer=indexer, **mla_args)
            # register with the sparse index group like SparseMLACommonImpl (enables HiSparse binding)
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLAIndexGroupBuilder
            self.index_group, self.index_group_index = None, 0
            if not os.environ.get("ARC_HISPARSE_ROW"):
                # Index groups are only needed by HiSparse; registering one otherwise adds a per-step
                # runtime whose pinned H2D copies get recorded inside CUDA-graph capture and break torch's
                # pinned host allocator afterwards ("CUDA error: invalid argument" at the next pinned alloc).
                igb = None
            elif igb is None and self.topk_indices_buffer is not None:
                igb = SparseMLAIndexGroupBuilder(self.topk_indices_buffer)
            if igb is not None:
                self.index_group, self.index_group_index = igb.register_layer(
                    indexer is not None, get_current_vllm_config(),
                    head_size=int(os.environ.get("ARC_HISPARSE_ROW", head_size)), kv_cache_dtype=kv_cache_dtype)

        def _logical(self, c):
            return c[..., : self.head_size]

        # ---- sparse-aware host offload (ARC_DOTS3_OFFLOAD=1) ----
        def _off_state(self):
            # (re)create when the indexer cache changes: vLLM first runs a profiling pass on a tiny
            # temporary KV cache, then allocates the real one.
            kc = self._kc_mod().kv_cache
            kc = kc[0] if isinstance(kc, (list, tuple)) else kc
            key = (kc.data_ptr(), kc.shape[0], kc.shape[1])
            if self._off is None or self._off_key != key:
                from .sparse_offload import LayerOffload
                self._off = None
                self._idx_bs = kc.shape[1]
                self._off = LayerOffload(kc.shape[0] * kc.shape[1], kc.device, key=self._indexer_mod.k_cache.prefix)
                self._off_key = key
            return self._off

        def _kc_mod(self):
            """indexer k_cache whose metadata/slots this layer's offload uses (the source layer's when its own indexer
            KV is freed by ARC_INDEX_SHARE_FREE)"""
            m = getattr(self, "_kcm", None)
            if m is None:
                m = self._indexer_mod.k_cache
                if os.environ.get("ARC_INDEX_SHARE_FREE") == "1":
                    from .index_share import source_cache
                    m = source_cache(m.prefix) or m
                self._kcm = m
            return m

        def _idx_md(self):
            from vllm.forward_context import get_forward_context
            fc = get_forward_context()
            md = fc.attn_metadata
            pre = self._kc_mod().prefix
            return (md.get(pre) if isinstance(md, dict) else None), fc.slot_mapping.get(pre)

        def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
            if self._offload:
                im, slots = self._idx_md()
                if im is None or slots is None:   # dummy/profile run
                    return
                self._off_state().store(kv_c_normed, k_pe, slots, 1.0)
                if not _VERIFY:
                    return
            return super().do_kv_cache_update(kv_c_normed, k_pe, self._logical(kv_cache), slot_mapping,
                                              kv_cache_dtype, k_scale)

        def _forward_offload(self, q, attn_metadata, layer):
            n = q.shape[0]
            im, slots = self._idx_md()
            if im is None:
                return torch.zeros((n, self.num_heads, self.kv_lora_rank), dtype=q.dtype, device=q.device)
            st = self._off_state()
            topk = self.topk_indices_buffer[:n]
            outs = []
            nd = im.num_decode_tokens
            res = (torch.empty((n, self.num_heads, self.kv_lora_rank), dtype=q.dtype, device=q.device)
                   if _PF_OUT and im.num_prefill_tokens else None)
            if nd:
                dec = im.decode
                if dec.requires_padding:
                    raise NotImplementedError("offload: padded (non-uniform) spec-decode batches")
                if dec.block_table.shape[0] >= nd and dec.decode_lens.shape[0] >= nd:
                    # flatten path (no native MTP kernel, e.g. SM89): one expanded block-table row per decode token
                    req = torch.arange(nd, device=q.device, dtype=torch.int32)
                    ql = max(1, nd // max(1, im.num_decodes))
                else:
                    ql = max(1, nd // max(1, im.num_decodes))
                    req = torch.arange(nd, device=q.device, dtype=torch.int32) // ql
                rows = triton_convert_req_index_to_global_index(
                    req, dec.block_table, topk[:nd], BLOCK_SIZE=self._idx_bs, NUM_TOPK_TOKENS=topk.shape[1])
                o = st.attend(q[:nd], rows, self.softmax_scale, 1.0, self.num_heads, ql=ql,
                              mkey=(id(im), topk.data_ptr(), nd), slots=slots,
                              out=res[:nd] if res is not None else None)
                if res is None:
                    outs.append(o)
                elif o.data_ptr() != res.data_ptr():   # (a path that did not take the slice)
                    res[:nd].copy_(o)
            if im.num_prefill_tokens:
                for ch in im.prefill.chunks:
                    tot = ch.total_seq_lens
                    j = ch.token_to_seq[:tot].long()
                    p = torch.arange(tot, device=q.device) - ch.cu_seq_lens[j].long()
                    bt = ch.block_table.long()
                    j = j.clamp(0, bt.shape[0] - 1)
                    p = p.clamp(min=0)
                    rows = bt[j, (p // self._idx_bs).clamp(max=bt.shape[1] - 1)] * self._idx_bs + p % self._idx_bs
                    lt = topk[ch.token_start:ch.token_end]
                    ks = ch.cu_seqlen_ks[: ch.token_end - ch.token_start].to(lt.dtype)
                    sidx = None
                    if _PF_COMPACT and tot > _PF_COMPACT:
                        rows, sidx = compact_stage(rows, torch.where(lt >= 0, lt + ks[:, None], torch.full_like(lt, -1)))
                        if rows.shape[0] == 0:                     # nothing selected: one dummy row, never read
                            rows = torch.zeros(1, dtype=torch.int64, device=q.device)
                    staged = torch.empty((rows.shape[0], 576), dtype=torch.float8_e4m3fn, device=q.device)   # fp8 rows (host and, under ARC_OFFLOAD_CCACHE, cache may be compact)
                    st.gather(rows, staged)
                    if sidx is None and os.environ.get("ARC_OFFLOAD_DEBUG") == "1" and self._indexer_mod.k_cache.prefix.endswith("layers.1.self_attn.indexer.k_cache"):
                        ke = ch.cu_seqlen_ke[: ch.token_end - ch.token_start].to(lt.dtype)
                        v = lt >= 0
                        cs = rows % st.C
                        hit = (st.tag[cs] == rows) & (rows >= 0)
                        same = (staged[hit].view(torch.uint8) == st.cache[cs[hit]].view(torch.uint8)).all().item() if hit.any() else None
                        print(f"[offdbg] tot={tot} q={lt.shape[0]} ks[min,max]=({int(ks.min())},{int(ks.max())}) "
                              f"ke[min,max]=({int(ke.min())},{int(ke.max())}) topk[min,max]=({int(lt[v].min())},{int(lt[v].max())}) "
                              f"max(topk-(ke-ks))={int((lt - (ke - ks)[:, None])[v].max())} rows[min,max]=({int(rows.min())},{int(rows.max())}) "
                              f"R={st.R} bs={self._idx_bs} bt={tuple(bt.shape)} staged_vs_cache_hits={int(hit.sum())} equal={same}", flush=True)
                    if sidx is None:
                        sidx = torch.where(lt >= 0, lt + ks[:, None], torch.full_like(lt, -1))
                    qc = q[ch.token_start:ch.token_end]
                    from . import sparse_offload as _so
                    if (_so._CUDA and qc.shape[1] == 32 and qc.stride(2) == 1 and lt.shape[1] % (_CUDA_PF_SPLIT * 32) == 0
                            and _so._cuda_ok()):
                        from . import offload_cuda
                        inplace = res is not None and self.num_heads == 32
                        dst = (res[ch.token_start:ch.token_end] if inplace else
                               torch.empty((qc.shape[0], 32, 512), dtype=qc.dtype, device=q.device))
                        o = offload_cuda.attend_rows(qc, staged, sidx.int().contiguous(), self.softmax_scale, 1.0, dst,
                                                     split=_CUDA_PF_SPLIT, stages=_so._CUDA_STAGES)
                        if res is None:
                            outs.append(o[:, : self.num_heads])
                        elif not inplace:
                            res[ch.token_start:ch.token_end].copy_(o[:, : self.num_heads])
                        continue
                    o, _, _ = triton_bf16_mla_sparse_interface(
                        qc, staged.view(staged.shape[0], 1, 576), sidx.view(lt.shape[0], 1, -1).int(),
                        sm_scale=self.softmax_scale, kv_scale=1.0)
                    if res is not None:
                        res[ch.token_start:ch.token_end].copy_(o[:, : self.num_heads])
                    else:
                        outs.append(o[:, : self.num_heads])
            if res is not None:
                return res
            return torch.cat(outs) if len(outs) > 1 else outs[0]

        def _run(self, q, kvc, gidx, layer):
            """kvc: paged [blocks, bs, phys] or flat [rows, phys] cache; gidx: physical row ids [n, topk]."""
            phys = kvc.shape[-1]
            if kvc.dim() == 3:
                stride_rows = kvc.stride(0) // phys
                num_rows = (kvc.shape[0] - 1) * stride_rows + kvc.shape[1]
            else:
                num_rows = kvc.shape[0]
            rows = kvc.as_strided((num_rows, 1, self.head_size), (phys, phys, 1))
            out, _, _ = triton_bf16_mla_sparse_interface(
                q, rows, gidx.reshape(q.shape[0], 1, -1).to(torch.int32), sm_scale=self.softmax_scale,
                kv_scale=float(getattr(layer, "_k_scale_float", 1.0)))
            return out[:, : self.num_heads, :]

        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            if isinstance(q, tuple):
                q = torch.cat(q, dim=-1)
            if self._offload and not _VERIFY:
                return self._forward_offload(q, attn_metadata, layer), None
            if self._offload:
                o_off = self._forward_offload(q, attn_metadata, layer)
                o_ref, _ = self._forward_paged(q, kv_c_and_k_pe_cache, attn_metadata, layer)
                self._verify(o_off, o_ref, attn_metadata)
                return o_ref, None
            return self._forward_paged(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        def _verify(self, o_off, o_ref, md):
            im, _ = self._idx_md()
            if im is None:
                return
            err = (o_off.float() - o_ref.float()).abs().amax((1, 2))
            ref = o_ref.float().abs().amax((1, 2)).clamp_min(1e-3)
            rel = err / ref
            bad = (rel > 0.1).nonzero().flatten()
            n = getattr(PortableSparseImpl, "_vcount", 0)
            PortableSparseImpl._vcount = n + 1
            if bad.numel() and getattr(PortableSparseImpl, "_vbad", 0) < 400:
                PortableSparseImpl._vbad = getattr(PortableSparseImpl, "_vbad", 0) + 1
                nd = im.num_decode_tokens
                print(f"[verify] {self._indexer_mod.k_cache.prefix.split('.')[-4]} step={n} tokens={rel.shape[0]} nd={nd} "
                      f"ndec={im.num_decodes} npre={im.num_prefill_tokens} bad={bad.numel()} first={bad[:8].tolist()} "
                      f"rel.max={float(rel.max()):.3f} pad={getattr(im.decode, 'requires_padding', None) if nd else None} "
                      f"dlens={im.decode.decode_lens.tolist()[:16] if nd and hasattr(im.decode, 'decode_lens') else None}",
                      flush=True)
            elif n % 20000 == 0:
                print(f"[verify] step={n} ok rel.max={float(rel.max()):.4f} tokens={rel.shape[0]}", flush=True)

        def _forward_paged(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            n = q.shape[0]
            topk = self.topk_indices_buffer[:n]
            ig = getattr(self, "index_group", None)
            if ig is not None and type(ig).__name__ == "HiSparseMLAIndexGroup":
                nd = attn_metadata.num_decode_tokens
                outs = []
                if nd:
                    cap = ig.cache(self.index_group_index).runtime.index_group.shared_topk.physical_topk_indices.shape[0]
                    if nd > cap:
                        # warmup/capture batch larger than the swap workspace: resident mapping, output unused
                        leader = ig.cache(0)
                        ptopk = ig._convert_once(
                            self.index_group_index, topk[:nd], attn_metadata.req_id_per_token[:nd],
                            leader.block_table, leader.view.block_size,
                            block_stride_rows=leader.view.attention_block_stride, return_valid_counts=True)[0]
                    else:
                        ptopk = ig.convert_decode_logical_to_physical_topk(
                            self.index_group_index, topk[:nd], attn_metadata, return_valid_counts=True)[0]
                    outs.append(self._run(q[:nd], ig.physical_kv_cache(self.index_group_index).view(
                        kv_c_and_k_pe_cache.dtype), ptopk, layer))
                if nd < n:
                    cache = ig.cache(self.index_group_index)
                    if nd == 0 and cache.all_context_pages_resident:
                        # all context resident: map logical -> physical rows directly (no hot-buffer swap)
                        leader = ig.cache(0)
                        ptopk = ig._convert_once(
                            self.index_group_index, topk, attn_metadata.req_id_per_token[:n],
                            leader.block_table, leader.view.block_size,
                            block_stride_rows=leader.view.attention_block_stride, return_valid_counts=True)[0]
                        pcache = ig.physical_kv_cache(self.index_group_index).view(kv_c_and_k_pe_cache.dtype)
                    else:
                        pcache, bt, req_ids = ig.stage_prefill_rows(self.index_group_index, kv_c_and_k_pe_cache,
                                                                    attn_metadata)
                        ptopk = triton_convert_req_index_to_global_index(
                            req_ids, bt, topk[nd:], BLOCK_SIZE=attn_metadata.block_size,
                            NUM_TOPK_TOKENS=topk.shape[1], return_valid_counts=True)[0]
                    outs.append(self._run(q[nd:], pcache, ptopk, layer))
                return (torch.cat(outs) if len(outs) > 1 else outs[0]), None
            c = kv_c_and_k_pe_cache
            stride_rows = c.stride(0) // c.shape[-1]
            gidx = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:n], attn_metadata.block_table, topk,
                BLOCK_SIZE=attn_metadata.block_size, BLOCK_STRIDE_ROWS=stride_rows, NUM_TOPK_TOKENS=topk.shape[1])
            return self._run(q, c, gidx, layer), None

    _OrigBuilder = orig_backend.get_builder_cls()

    class PortableBuilder(_OrigBuilder):
        """Decode = genuine decode/spec-verify steps only (HiSparse sizes its swap workspace for
        max_num_seqs x (1 + num_speculative_tokens)); the FA builder's 128-256 threshold would
        classify short prefills as decode."""
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            sc = vllm_config.speculative_config
            k = (sc.num_speculative_tokens or 0) if sc is not None else 0
            self._init_reorder_batch_threshold(1 + k, supports_spec_as_decode=True)

    class PortableXPUBuilder(xs.XPUMLASparseMetadataBuilder):
        """XPU builder with req_id_per_token computed on the GPU (searchsorted over query_start_loc)
        instead of staging through a freshly pinned host tensor every step."""
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            cam = common_attn_metadata
            n = cam.num_actual_tokens
            qsl = cam.query_start_loc
            tok = torch.arange(self.req_id_per_token_buffer.shape[0], device=qsl.device, dtype=qsl.dtype)
            rid = torch.searchsorted(qsl[1:cam.num_reqs + 1], tok, right=True).clamp_(max=max(cam.num_reqs - 1, 0))
            self.req_id_per_token_buffer.copy_(rid.to(torch.int32))
            return xs.XPUMLASparseMetadata(
                num_reqs=cam.num_reqs, max_query_len=cam.max_query_len, max_seq_len=cam.max_seq_len,
                num_actual_tokens=n, query_start_loc=qsl, slot_mapping=cam.slot_mapping,
                block_table=cam.block_table_tensor, req_id_per_token=self.req_id_per_token_buffer[:n],
                block_size=self.kv_cache_spec.block_size, topk_tokens=self.topk_tokens,
                num_decodes=cam.num_reqs, num_prefills=0, num_decode_tokens=n)

    class PortableSparseBackend(xs.XPUMLASparseBackend):
        _arc_portable = True

        @classmethod
        def supported_kv_cache_layouts(cls):
            return orig_backend.supported_kv_cache_layouts()

        @staticmethod
        def get_name():
            return "DOTS3_PORTABLE_MLA_SPARSE"

        @staticmethod
        def get_impl_cls():
            return PortableSparseImpl

        # HiSparse needs the decode/prefill-split metadata of the FA-sparse builder; otherwise the XPU
        # builder (every token through the sparse MQA kernel) is used.
        @staticmethod
        def get_builder_cls():
            if os.environ.get("ARC_HISPARSE_ROW"):
                return PortableBuilder
            return PortableXPUBuilder

        @staticmethod
        def get_metadata_cls():
            return orig_backend.get_metadata_cls() if os.environ.get("ARC_HISPARSE_ROW") else xs.XPUMLASparseMetadata

        @classmethod
        def get_supported_head_sizes(cls):
            return []

    dm.Dots3NotePaddedSparseBackend = PortableSparseBackend
    da.Dots3NotePaddedSparseBackend = PortableSparseBackend

    # Dots3's DSA layer first builds a stock sparse MLAAttention before swapping in the padded layer. On SM89 no
    # stock sparse-MLA backend exists (Kaggle L4: "No valid attention backend"), so fall back to ours.
    from vllm.v1.attention import selector as _sel
    if not getattr(_sel, "_arc_sparse_fallback", False):
        _orig_cached = _sel._cached_get_attn_backend

        def _cached(backend, attn_selector_config, num_heads=None):
            try:
                return _orig_cached(backend, attn_selector_config, num_heads=num_heads)
            except ValueError:
                if getattr(attn_selector_config, "use_sparse", False) and getattr(attn_selector_config, "use_mla", False):
                    return PortableSparseBackend
                raise
        _sel._cached_get_attn_backend = _cached
        _sel._arc_sparse_fallback = True

    # DeepSeek-V3.2-style models resolve "fp8" to the packed fp8_ds_mla layout (656 B/token, FlashMLA only);
    # the portable kernel reads plain per-tensor fp8 rows, so undo that for the sparse layers.
    import copy as _copy
    orig_init = dm.Dots3NotePaddedMLAAttention.__init__

    def padded_init(self, *a, **kw):
        cc = kw.get("cache_config")
        if cc is not None and cc.cache_dtype == "fp8_ds_mla":
            cc = _copy.copy(cc); cc.cache_dtype = "fp8"; kw["cache_config"] = cc
        orig_init(self, *a, **kw)

    dm.Dots3NotePaddedMLAAttention.__init__ = padded_init

    if os.environ.get("ARC_DOTS3_OFFLOAD") == "1" and not _VERIFY:
        from .sparse_offload import WINDOW
        orig_spec = dm.Dots3NotePaddedMLAAttention.get_kv_cache_spec

        from dataclasses import replace as _replace
        from vllm.model_executor.layers.attention import MLAAttention as _BaseMLA

        def offload_spec(self, vllm_config):
            # vLLM keeps only a tiny window for these layers; the full latent KV lives in host memory
            saved = self.sliding_window
            self.sliding_window = WINDOW
            try:
                spec = _BaseMLA.get_kv_cache_spec(self, vllm_config)
            finally:
                self.sliding_window = saved
            return _replace(spec, head_size=self.physical_head_size)

        dm.Dots3NotePaddedMLAAttention.get_kv_cache_spec = offload_spec
