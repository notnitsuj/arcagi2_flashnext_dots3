"""Sparse-aware KV offload for Dots3 DSA layers (ARC_DOTS3_OFFLOAD=1, on top of ARC_DOTS3_PORTABLE_DSA=1).

Memory layout (per DSA layer, per worker process):
  host  : pinned [R, 576] fp8 latent rows (512 NoPE + 64 RoPE), unpadded. Row id = the *indexer* cache slot
          (block_id * block_size + offset) of that token, so the indexer's block tables address it for the
          request's whole lifetime. GPU kernels access it directly through UVA (pinned memory).
  cache : GPU [C, 576] fp8 direct-mapped row cache + tag [C] (host row id or -1). Written through on every
          KV update (newest tokens are hot), filled with missed rows after each decode attention.
vLLM only keeps a tiny sliding window for these layers on GPU (spec = SlidingWindowMLASpec(window=W)); the
index keys (indexer cache, ~1.7 KB/token for 13 layers) and the SWA layers stay on GPU as before.

Decode: attention kernel reads each selected row from the cache on a tag hit, else from host (only misses
cross PCIe). Prefill: each chunk's context rows are gathered once from host into a GPU staging buffer.
"""
import os

import torch
import triton
import triton.language as tl

from . import offload_rowfmt as _rf   # ARC_OFFLOAD_ROW_FMT: compact host rows (default fp8: unchanged)

D_NOPE, D_ROPE = 512, 64
D_ROW = D_NOPE + D_ROPE
FP8 = torch.float8_e4m3fn
CACHE_ROWS = int(os.environ.get("ARC_OFFLOAD_CACHE_ROWS", str(1 << 18)))
# ARC_OFFLOAD_CACHE_ROWS_L="layer:rows,...": per-layer row-cache size (backbone layer ids; others ARC_OFFLOAD_CACHE_ROWS).
# An index-shared layer and its source should get the same size (ARC_OFFLOAD_MIRROR needs equal caches).
_WIN_LAZY = os.environ.get("ARC_OFFLOAD_WIN_LAZY", "0") == "1"
_CROWS_L = {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get("ARC_OFFLOAD_CACHE_ROWS_L", "").split(",") if x)}
WINDOW = int(os.environ.get("ARC_OFFLOAD_GPU_WINDOW", "64"))
_STATS = int(os.environ.get("ARC_OFFLOAD_STATS", "0"))           # log row-cache hit rate every N decode calls
# ARC_OFFLOAD_DUMP=path,after_s,calls,min_T: rank-0 first offload layer records its stored slots and decode top-k rows
# for `calls` decode calls (once after_s seconds and T >= min_T) to path.npz, for offline row-cache simulation
_DUMP = os.environ.get("ARC_OFFLOAD_DUMP", "")
_PREFILL = os.environ.get("ARC_OFFLOAD_PREFILL", "1") == "1"      # fill misses before (not after) decode attention
_SPLIT = int(os.environ.get("ARC_OFFLOAD_SPLIT", "8"))            # flash-decoding splits of the top-k rows (1 = single pass)
_SPLIT_BN = int(os.environ.get("ARC_OFFLOAD_SPLIT_BN", "32"))
_WAYS = int(os.environ.get("ARC_OFFLOAD_WAYS", "1"))              # >1: W-way set-associative row cache
_SPLIT_WARPS = int(os.environ.get("ARC_OFFLOAD_SPLIT_WARPS", "4"))
_SPLIT_STAGES = int(os.environ.get("ARC_OFFLOAD_SPLIT_STAGES", "2"))
_SPLIT_BH = int(os.environ.get("ARC_OFFLOAD_SPLIT_BH", "16"))     # heads per program
_SPLIT_DVS = int(os.environ.get("ARC_OFFLOAD_SPLIT_DVS", "1"))    # latent output column slices per (token, heads, split)
_SPLIT_F16 = os.environ.get("ARC_OFFLOAD_SPLIT_F16", "0") == "1"  # fp16 MMA inputs in the split kernel (bf16 default)
_CUDA = os.environ.get("ARC_OFFLOAD_CUDA", "0") == "1"     # CUDA split kernel (offload_cuda.py; 32 heads, resolved rows)
_CUDA_SPLIT = int(os.environ.get("ARC_OFFLOAD_CUDA_SPLIT", "4"))
_CUDA_STAGES = int(os.environ.get("ARC_OFFLOAD_CUDA_STAGES", "3"))   # 2 | 3 pipeline stages, +10 = 8-warp CTAs
_OA2 = int(os.environ.get("ARC_OFFLOAD_ATTN2", "0") or 0)   # decode attention v2 (offload_cuda2.py), default off


def _oa2_time_begin(dev):
    if os.environ.get("ARC_OFFLOAD_ATTN2_TIME", "0") in ("", "0"):
        return None
    from . import offload_cuda2
    return offload_cuda2.time_begin(dev)


def _oa2_ok(T, idx, res, q):
    """ARC_OFFLOAD_ATTN2 on (and, under ARC_OFFLOAD_AB, the window's set has 'attn2'; 'attn2f8' picks FP8 P.V)"""
    if not _OA2 or (_AB > 0 and not ab_on("attn2")):
        return False
    from . import offload_cuda2
    return offload_cuda2.ok(T, q.shape[1], idx.shape[1], q, idx, res)
_ROWMAJOR = int(os.environ.get("ARC_OFFLOAD_ROWMAJOR", "2"))      # 2: one row-major load per row via cache/host pointer select
# W>1: resolve every (token, k) entry to a GPU location before attention; rows that find no free way in their set this
# step go to a shared per-step spill buffer instead of being read from host memory inside the attention kernel
_RESOLVE = os.environ.get("ARC_OFFLOAD_RESOLVE", "0") == "1"
# warm the NEXT offload layer's row cache on a side stream with this layer's decode rows (index-shared layers use the
# same rows; others overlap heavily). Speculative: that layer's own scan/fill still runs, and it joins the prefetch
# before it writes new rows (store invalidates them), so a wrong guess costs only time.
_PREFETCH = os.environ.get("ARC_OFFLOAD_PREFETCH", "0") == "1"
_PF_STREAM = {}
_LAST = {}
_PF_LOGGED = []
_SPILL = int(os.environ.get("ARC_OFFLOAD_SPILL", "32768"))          # spill rows (shared by all layers)
_FILLX = any(os.environ.get(k, "0") == "1" for k in ("ARC_OFFLOAD_HT_EPOCH", "ARC_OFFLOAD_FILL_SPLIT",
                                                        "ARC_OFFLOAD_TWO_CHOICE"))   # offload_fill.py variants
# request-packed decode (needs W>1 + resolve): the QL MTP tokens of a request attend over the UNION of their top-k rows
# (they share 60-75% of them) with all QL x H query rows in one MMA tile and a per-row token-membership mask, so each
# row is loaded once per request instead of once per token
_PACKED = os.environ.get("ARC_OFFLOAD_PACKED", "0") == "1"
_PK_KS = int(os.environ.get("ARC_OFFLOAD_PK_KS", "8"))          # split-K over the union
_PK_BN = int(os.environ.get("ARC_OFFLOAD_PK_BN", "64"))
_PK_DV = int(os.environ.get("ARC_OFFLOAD_PK_DV", "512"))        # latent columns per program (QK recomputed per slice)
_PK_WARPS = int(os.environ.get("ARC_OFFLOAD_PK_WARPS", "8"))
_PK_STAGES = int(os.environ.get("ARC_OFFLOAD_PK_STAGES", "2"))
_PK_QCHUNK = int(os.environ.get("ARC_OFFLOAD_PK_QCHUNK", "1"))  # stream q in BD chunks (q tile not register-resident)
_PK_BD = int(os.environ.get("ARC_OFFLOAD_PK_BD", "128"))
# request-packed CUDA decode (offload_cuda.pk_*, needs W>1 + resolve): per request (QL MTP tokens) the distinct rows of
# the tokens' top-k are built once (pk_union, with per-row token bitmasks), cache-filled / resolved once, and attended by
# CTAs of 64 query rows (64 / HG tokens x HG heads) that load each union row once and skip MMA blocks no token selected
_CUDA_PK = os.environ.get("ARC_OFFLOAD_CUDA_PK", "0") == "1"
_CUDA_PK_KS = int(os.environ.get("ARC_OFFLOAD_CUDA_PK_KS", "1"))     # split-K over the union (merged in-kernel)
_CUDA_PK_CFG = int(os.environ.get("ARC_OFFLOAD_CUDA_PK_CFG", "2"))   # 2 | 3 stages of 16-row tiles (+10: 32-row, not sm_89)
_CUDA_PK_QL = int(os.environ.get("ARC_OFFLOAD_CUDA_PK_QL", "0"))     # >0: pack at most this many tokens per CTA group
# ARC_OFFLOAD_MIRROR=1 (needs W>1 + resolve + the split fill of offload_fill, ARC_INDEX_SHARE_FREE): an index-shared
# (S) layer attends over exactly the rows of its source F layer, so its row cache is kept as a MIRROR of the F layer's:
# same line for every row (the F layer's tags), its own data. The S layer then skips scan / alloc / sort / resolve and
# only copies the F layer's miss list (F's cpd destinations) from its own host rows; attention reuses F's resolution.
# Exact: every resolved location holds the S layer's own copy of the same row id.
# ARC_OFFLOAD_MIRROR=2: additionally the S layer's copy runs on a side stream right after the F layer's fill (overlapping
# F's attention and the layers in between) into a private spill buffer; the S layer's store joins it and writes the
# step's new rows through to the lines / spill rows the F layer gave them.
_MIRROR = os.environ.get("ARC_OFFLOAD_MIRROR", "0") in ("1", "2")
_MASYNC = os.environ.get("ARC_OFFLOAD_MIRROR", "0") == "2"
_MCTAS = int(os.environ.get("ARC_OFFLOAD_MIRROR_CTAS", "1"))      # side-stream copy CTAs per SM
_MSIDE, _MPEND, _MSPILL = {}, {}, {}
# ARC_OFFLOAD_LOWGRID=G (default 0 = off): the side-stream copies (overlapped fill, mirror) use G CTAs with 4 rows per
# warp instead of one CTA per SM, so fewer SMs host a copy CTA beside the co-running GEMM / attention CTAs
# (A/B feature 'lowgrid' under ARC_OFFLOAD_AB)
_LOWGRID = int(os.environ.get("ARC_OFFLOAD_LOWGRID", "0"))
# ARC_OFFLOAD_CUDA_HCHUNK=1: the CUDA split attention (32 heads per launch) also serves H = 64, 96, ... heads per rank
# (TP2 accuracy servers) as H / 32 launches over head slices; implied by compact host rows (ARC_OFFLOAD_ROW_FMT), whose
# decode needs the CUDA kernel. Default off: H != 32 keeps the Triton split kernel.
_HCHUNK = os.environ.get("ARC_OFFLOAD_CUDA_HCHUNK", "0") == "1" or _rf.ON


# In-run A/B of the side-stream copy's reads in flight (only under ARC_OFFLOAD_AB; default path unchanged):
# feature 'rpw' = ARC_OFFLOAD_AB_RPW rows per warp (default 4), feature 'grid2' = ARC_OFFLOAD_AB_GRID x the CTAs
# (default 2). Bit-exact (the copy moves the same bytes to the same rows; only its parallelism changes).
_AB_RPW = int(os.environ.get("ARC_OFFLOAD_AB_RPW", "4"))
_AB_GRID = int(os.environ.get("ARC_OFFLOAD_AB_GRID", "2"))


def side_grid(dev, default_ctas, rpw):
    """grid argument (ctas + 1000 * rows per warp) of a side-stream fill copy"""
    if _LOWGRID > 0 and ab_on("lowgrid"):
        return _LOWGRID + 4000
    if _AB > 0:
        if ab_on("rpw"):
            rpw = _AB_RPW
        if ab_on("grid2"):
            default_ctas = min(999, default_ctas * _AB_GRID)
    return default_ctas + 1000 * rpw


# ARC_OFFLOAD_MIRROR_GRID=G[,RPW] (default off): the MIRROR=2 S-layer side copies (background, ~1 layer span of slack)
# run with G CTAs x RPW rows per warp (RPW 1 | 2 | 4, default 4) instead of one CTA per SM. A resident copy CTA takes a
# register slot of the co-running MoE / W8A8 GEMM CTAs on its SM (MoE 127 regs x 128 thr: 4 -> 3 CTAs per SM; W8A8 243
# regs: 2 -> 1), which made every kernel overlapping a mirror copy ~60% slower on Kaggle (cfgprobe nsys, ~8 ms/step);
# with G << #SMs only G SMs lose a slot. Same rows, same destinations: bit-exact. A/B feature 'mgrid'.
_MGRID = os.environ.get("ARC_OFFLOAD_MIRROR_GRID", "").strip()
_MG_LOG = []
_GATE_LOG = []


def mirror_grid(dev, default_ctas, rpw):
    """grid argument of a MIRROR=2 side copy (ARC_OFFLOAD_MIRROR_GRID, else side_grid)"""
    if _MGRID and _MGRID != "0" and ab_on("mgrid"):
        g, _, r = _MGRID.partition(",")
        r = int(r or 4)
        assert r in (1, 2, 4), "ARC_OFFLOAD_MIRROR_GRID: rows per warp must be 1, 2 or 4"
        if not _MG_LOG:
            _MG_LOG.append(1)
            print(f"[offload] mirror side copies on {g} CTAs x {r} rows per warp (ARC_OFFLOAD_MIRROR_GRID)", flush=True)
        return max(1, min(999, int(g))) + 1000 * r
    return side_grid(dev, default_ctas, rpw)


def _ar_gate(dev):
    """ARC_OFFLOAD_AR_GATE (link_gate.py): the all-reduce gate word for a mirror side copy, else None (A/B 'argate')"""
    from . import link_gate
    if not link_gate.ON or not ab_on("argate"):
        return None
    from . import offload_cuda
    if not offload_cuda.gate_available():
        if not _GATE_LOG:
            _GATE_LOG.append(1)
            print("[offload] ARC_OFFLOAD_AR_GATE: offload_attn.so has no gated fill copy; gate off", flush=True)
        return None
    return link_gate.word(dev)


def mjoin(dev):
    """the current stream waits for every pending mirror side copy on this device"""
    ev = _MPEND.pop(dev, None)
    if ev:
        cs = torch.cuda.current_stream(dev)
        for e in ev:
            cs.wait_event(e)


def _side(dev):
    st = _MSIDE.get(dev)
    if st is None:   # high priority: its copy CTAs are dispatched ahead of the attention's queued CTAs
        st = _MSIDE[dev] = torch.cuda.Stream(dev, priority=-1)
    return st


def _gpu_rows(n, dev, zero=False):
    """n GPU rows (cache lines / spill rows): fp8 [n, 576], or compact uint8 [n, RB] under ARC_OFFLOAD_CCACHE"""
    f = torch.zeros if zero else torch.empty
    return f((n, _rf.LB), dtype=torch.uint8, device=dev) if _rf.CC else f((n, D_ROW), dtype=FP8, device=dev)


def _mspill(dev):
    b = _MSPILL.get(dev)
    if b is None:
        b = _MSPILL[dev] = _gpu_rows(_SPILL, dev)
    return b
# ARC_OFFLOAD_AB=P (diagnostics/A-B, default 0 = off): the features of this file follow a rotation of P-second windows
# of time since ARC_T0 (run_l4's start; else the epoch), so one serving run measures several configurations.
# ARC_OFFLOAD_AB_SETS: the windows' feature sets, '/'-separated, each '+'-joined from {mirror, spill} or 'none'
# (default "all/none": everything on, then off). 'spill' = spill rows above ARC_OFFLOAD_SPILL_BASE (cap otherwise),
# 'ovl' = ARC_OFFLOAD_OVERLAP (offload_fill).
_AB = float(os.environ.get("ARC_OFFLOAD_AB", "0"))
_AB_SETS = [set(x.split("+")) - {"none", ""} for x in os.environ.get("ARC_OFFLOAD_AB_SETS", "all/none").split("/")]
_AB_T0 = float(os.environ.get("ARC_T0", "0"))
_SPILL_BASE = int(os.environ.get("ARC_OFFLOAD_SPILL_BASE", "32768"))
# ARC_OFFLOAD_LSTATS=N: every N decode calls of each layer log distinct hits / misses / spills of that call (1 sync)
_LSTATS = int(os.environ.get("ARC_OFFLOAD_LSTATS", "0"))
_AB_LAST = [None]


def ab_on(feat="all"):
    """feature `feat` is on in the current ARC_OFFLOAD_AB window (always without ARC_OFFLOAD_AB)"""
    if _AB <= 0:
        return True
    import time
    t = time.time() - _AB_T0
    ph = int(t // _AB) % len(_AB_SETS)
    if ph != _AB_LAST[0]:
        _AB_LAST[0] = ph
        print(f"[offab] phase {ph} {'+'.join(sorted(_AB_SETS[ph])) or 'none'} t={t:.1f}", flush=True)
    fs = _AB_SETS[ph]
    return "all" in fs or feat in fs


def spill_cap():
    return _SPILL if ab_on("spill") else min(_SPILL, _SPILL_BASE)


_MOK = [None]


def _mirror_ok():
    if not _MIRROR:
        return False
    if _MOK[0] is None:
        from . import offload_fill as of
        _MOK[0] = bool(_RESOLVE and _WAYS > 1 and of.enabled() and (of.SPLIT or of.TWO) and not _PREFETCH
                       and not _CUDA_PK and not _PACKED)
        print(f"[offload] mirror of index-shared layers {'ON' if _MOK[0] else 'unavailable (needs W>1, resolve, the split fill, no prefetch/packed)'}",
              flush=True)
    return _MOK[0]


_REG = {}     # offload key (indexer k_cache prefix) -> LayerOffload (latest instance)


# ---------------------------------------------------------------- kernels
@triton.jit
def _store_rows_kernel(kvc_ptr, kpe_ptr, slot_ptr, host_ptr, win_ptr, inv_scale, C, R,
                       stride_kvc, stride_kpe, DN: tl.constexpr, DR: tl.constexpr):
    """Write-through phase 1: host row + claim of the cache line (several new slots of one step can map to the
    same direct-mapped line; only the claim winner writes it in phase 2, so tag and data always agree)."""
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if (slot < 0) | (slot >= R):
        return
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    a = tl.load(kvc_ptr + t * stride_kvc + dn).to(tl.float32) * inv_scale
    b = tl.load(kpe_ptr + t * stride_kpe + dr).to(tl.float32) * inv_scale
    a = tl.clamp(a, -448.0, 448.0).to(tl.float8e4nv)
    b = tl.clamp(b, -448.0, 448.0).to(tl.float8e4nv)
    D: tl.constexpr = DN + DR
    tl.store(host_ptr + slot * D + dn, a)
    tl.store(host_ptr + slot * D + DN + dr, b)
    tl.atomic_max(win_ptr + slot % C, slot)


@triton.jit
def _store_cache_kernel(kvc_ptr, kpe_ptr, slot_ptr, win_ptr, cache_ptr, tag_ptr, inv_scale, C, R,
                        stride_kvc, stride_kpe, DN: tl.constexpr, DR: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if (slot < 0) | (slot >= R):
        return
    cs = slot % C
    if tl.load(win_ptr + cs) != slot:
        return
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    a = tl.clamp(tl.load(kvc_ptr + t * stride_kvc + dn).to(tl.float32) * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
    b = tl.clamp(tl.load(kpe_ptr + t * stride_kpe + dr).to(tl.float32) * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
    D: tl.constexpr = DN + DR
    tl.store(cache_ptr + cs * D + dn, a)
    tl.store(cache_ptr + cs * D + DN + dr, b)
    tl.store(tag_ptr + cs, slot)


@triton.jit
def _gather_rows_kernel(src_ptr, rows_ptr, dst_ptr, n, R, DN: tl.constexpr, DR: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
    m = m & (r >= 0) & (r < R)
    D: tl.constexpr = DN + DR
    i64 = i.to(tl.int64)
    dn = tl.arange(0, DN)
    dr = DN + tl.arange(0, DR)
    tl.store(dst_ptr + i64[:, None] * D + dn[None, :],
             tl.load(src_ptr + r[:, None] * D + dn[None, :], mask=m[:, None], other=0.0), mask=(i < n)[:, None])
    tl.store(dst_ptr + i64[:, None] * D + dr[None, :],
             tl.load(src_ptr + r[:, None] * D + dr[None, :], mask=m[:, None], other=0.0), mask=(i < n)[:, None])


@triton.jit
def _propose_kernel(idx_ptr, tag_ptr, win_ptr, n, C, R, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    m = m & (h >= 0) & (h < R)
    s = h % C
    tg = tl.load(tag_ptr + s, mask=m, other=-1)
    miss = m & (tg != h)
    tl.atomic_max(win_ptr + s, h, mask=miss)


@triton.jit
def _fill_kernel(idx_ptr, win_ptr, tag_ptr, host_ptr, cache_ptr, n, C, R, DN: tl.constexpr, DR: tl.constexpr):
    i = tl.program_id(0)
    h = tl.load(idx_ptr + i).to(tl.int64)
    if (h < 0) | (h >= R):
        return
    s = h % C
    if tl.load(win_ptr + s) != h:
        return
    D: tl.constexpr = DN + DR
    dn = tl.arange(0, DN)
    dr = DN + tl.arange(0, DR)
    tl.store(cache_ptr + s * D + dn, tl.load(host_ptr + h * D + dn))
    tl.store(cache_ptr + s * D + dr, tl.load(host_ptr + h * D + dr))
    tl.store(tag_ptr + s, h)   # identical writers (duplicate h) write identical data


KEEP = tl.constexpr(1 << 62)   # line claim value of a cache line that holds a row needed this step (never evicted by this step)


@triton.jit
def _claim_kernel(idx_ptr, tag_ptr, win_ptr, n, C, R, BLOCK: tl.constexpr):
    """One pass over this step's rows: a line whose current row is needed is pinned (KEEP); otherwise the largest
    missing row id mapping to it wins the line."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    m = m & (h >= 0) & (h < R)
    s = h % C
    tg = tl.load(tag_ptr + s, mask=m, other=-1)
    tl.atomic_max(win_ptr + s, tl.where(tg == h, KEEP, h), mask=m)


@triton.jit
def _prefill_kernel(idx_ptr, win_ptr, tag_ptr, host_ptr, cache_ptr, n, C, R, DN: tl.constexpr, DR: tl.constexpr):
    """Copy each winning missing row host -> cache exactly once (duplicates of the row lose the CAS)."""
    i = tl.program_id(0)
    h = tl.load(idx_ptr + i).to(tl.int64)
    if (h < 0) | (h >= R):
        return
    s = h % C
    if tl.atomic_cas(win_ptr + s, h, h * 0 - 3) != h:   # int64 operands
        return
    D: tl.constexpr = DN + DR
    dn = tl.arange(0, DN)
    dr = DN + tl.arange(0, DR)
    tl.store(cache_ptr + s * D + dn, tl.load(host_ptr + h * D + dn))
    tl.store(cache_ptr + s * D + dr, tl.load(host_ptr + h * D + dr))
    tl.store(tag_ptr + s, h)


@triton.jit
def _sparse_mla_offload_kernel(q_ptr, host_ptr, cache_ptr, tag_ptr, idx_ptr, out_ptr, C, R,
                               stride_q_t, stride_q_h, stride_o_t, stride_o_h, stride_i_t,
                               h_q, sm_scale, kv_scale,
                               TOPK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                               DN: tl.constexpr, DR: tl.constexpr, W: tl.constexpr = 1):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = heads < h_q
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    D: tl.constexpr = DN + DR
    q = tl.load(q_ptr + t * stride_q_t + heads[:, None] * stride_q_h + dn[None, :], mask=mh[:, None], other=0.0)
    qr = tl.load(q_ptr + t * stride_q_t + heads[:, None] * stride_q_h + DN + dr[None, :], mask=mh[:, None], other=0.0)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, DN], dtype=tl.float32)
    for start in range(0, TOPK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        h = tl.load(idx_ptr + t * stride_i_t + offs, mask=offs < TOPK, other=-1).to(tl.int64)
        valid = (h >= 0) & (h < R)
        st = tl.where(valid, h % (C // W), 0)
        wv = tl.arange(0, W)
        mt = (tl.load(tag_ptr + st[:, None] * W + wv[None, :], mask=valid[:, None], other=-1) == h[:, None]).to(tl.int32)
        hit = valid & (tl.max(mt, 1) > 0)
        s = st * W + tl.argmax(mt, 1)
        miss = valid & (~hit)
        # K^T tiles [D, BLOCK_N] from cache (hits) or host (misses); masked loads -> zero elsewhere
        kn = tl.load(cache_ptr + s[None, :] * D + dn[:, None], mask=hit[None, :], other=0.0).to(tl.bfloat16) + \
            tl.load(host_ptr + h[None, :] * D + dn[:, None], mask=miss[None, :], other=0.0).to(tl.bfloat16)
        kr = tl.load(cache_ptr + s[None, :] * D + DN + dr[:, None], mask=hit[None, :], other=0.0).to(tl.bfloat16) + \
            tl.load(host_ptr + h[None, :] * D + DN + dr[:, None], mask=miss[None, :], other=0.0).to(tl.bfloat16)
        qk = tl.dot(q, kn) + tl.dot(qr, kr)
        qk = qk * sm_scale
        qk = tl.where(mh[:, None] & valid[None, :], qk, -1.0e30)
        v = tl.trans(kn)  # V = NoPE latent [BLOCK_N, DN]
        n_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp2(e_max - n_max)
        p = tl.exp2(qk - n_max[:, None])
        acc = acc * re[:, None] + tl.dot(p.to(tl.bfloat16), v)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_max
    acc = acc / e_sum[:, None] * kv_scale
    tl.store(out_ptr + t * stride_o_t + heads[:, None] * stride_o_h + dn[None, :], acc.to(tl.bfloat16), mask=mh[:, None])


@triton.jit
def _sparse_mla_offload_split_kernel(q_ptr, host_ptr, cache_ptr, tag_ptr, idx_ptr, m_ptr, l_ptr, acc_ptr, C, R,
                                     stride_q_t, stride_q_h, stride_i_t, h_q, sm_scale, res_ptr, spill_ptr,
                                     TOPK: tl.constexpr, SPLIT: tl.constexpr, BLOCK_H: tl.constexpr,
                                     BLOCK_N: tl.constexpr, HP: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
                                     W: tl.constexpr = 1, ROWMAJOR: tl.constexpr = 0, RES: tl.constexpr = 0,
                                     DVS: tl.constexpr = 1, CD: tl.constexpr = tl.bfloat16,
                                     ABL: tl.constexpr = 0, QK8: tl.constexpr = 0):
    """Flash-decoding variant: program (token, head block, split) covers TOPK/SPLIT selected rows and writes a partial
    (max, sum, acc) in fp32; _combine_kernel merges the SPLIT partials. Same hit/miss row sourcing as above.
    DVS > 1 (RES or ROWMAJOR=2 paths): DVS programs share a (token, head block, split) and each accumulates DN/DVS of
    the latent output columns (QK recomputed per slice): a [BLOCK_H, DN/DVS] accumulator instead of [BLOCK_H, DN]
    (DN=512 fp32 x 32 heads = 255 regs + spills, 1 CTA/SM on L4).
    CD: MMA input dtype for q, the FP8 rows and p (float16: sm89 converts e4m3 pairs to f16 in one instruction)."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2) // DVS
    vs = tl.program_id(2) % DVS
    DNV: tl.constexpr = DN // DVS
    dv = tl.arange(0, DNV)
    heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = heads < h_q
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    D: tl.constexpr = DN + DR
    q = tl.load(q_ptr + t * stride_q_t + heads[:, None] * stride_q_h + dn[None, :], mask=mh[:, None], other=0.0).to(CD)
    qr = tl.load(q_ptr + t * stride_q_t + heads[:, None] * stride_q_h + DN + dr[None, :], mask=mh[:, None], other=0.0).to(CD)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, DNV], dtype=tl.float32)
    CHUNK: tl.constexpr = TOPK // SPLIT
    for start in range(sp * CHUNK, sp * CHUNK + CHUNK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        h = tl.load(idx_ptr + t * stride_i_t + offs).to(tl.int64)
        valid = (h >= 0) & (h < R)
        if RES:   # resolved location: >= 0 cache line, <= -2 spill row -r-2, -1 host
            r = tl.load(res_ptr + t * TOPK + offs, mask=valid, other=-1).to(tl.int64)
            base = tl.where(r >= 0, cache_ptr + r * D, tl.where(r <= -2, spill_ptr + (-r - 2) * D, host_ptr + h * D))
            kt = tl.load(base[:, None] + dn[None, :], mask=valid[:, None], other=0.0).to(CD)
            krt = tl.load(base[:, None] + DN + dr[None, :], mask=valid[:, None], other=0.0).to(CD)
            kn = tl.trans(kt)
            kr = tl.trans(krt)
        else:
            st = tl.where(valid, h % (C // W), 0)
            wv = tl.arange(0, W)
            mt = (tl.load(tag_ptr + st[:, None] * W + wv[None, :], mask=valid[:, None], other=-1) == h[:, None]).to(tl.int32)
            hit = valid & (tl.max(mt, 1) > 0)
            s = st * W + tl.argmax(mt, 1)
            miss = valid & (~hit)
        if RES:
            pass
        elif ROWMAJOR == 2:   # one load per row through a per-row base pointer (cache line on hit, host row via UVA)
            base = tl.where(hit, cache_ptr + s * D, host_ptr + h * D)
            kt = tl.load(base[:, None] + dn[None, :], mask=valid[:, None], other=0.0).to(CD)
            krt = tl.load(base[:, None] + DN + dr[None, :], mask=valid[:, None], other=0.0).to(CD)
            kn = tl.trans(kt)
            kr = tl.trans(krt)
        elif ROWMAJOR:   # [BLOCK_N, D] tiles: contiguous 576 B rows -> wide vector loads; transposed in registers
            kt = tl.load(cache_ptr + s[:, None] * D + dn[None, :], mask=hit[:, None], other=0.0).to(CD) + \
                tl.load(host_ptr + h[:, None] * D + dn[None, :], mask=miss[:, None], other=0.0).to(CD)
            krt = tl.load(cache_ptr + s[:, None] * D + DN + dr[None, :], mask=hit[:, None], other=0.0).to(CD) + \
                tl.load(host_ptr + h[:, None] * D + DN + dr[None, :], mask=miss[:, None], other=0.0).to(CD)
            kn = tl.trans(kt)
            kr = tl.trans(krt)
        else:
            kn = tl.load(cache_ptr + s[None, :] * D + dn[:, None], mask=hit[None, :], other=0.0).to(CD) + \
                tl.load(host_ptr + h[None, :] * D + dn[:, None], mask=miss[None, :], other=0.0).to(CD)
            kr = tl.load(cache_ptr + s[None, :] * D + DN + dr[:, None], mask=hit[None, :], other=0.0).to(CD) + \
                tl.load(host_ptr + h[None, :] * D + DN + dr[:, None], mask=miss[None, :], other=0.0).to(CD)
        # ABL (diagnostics only, 0 = normal): 1 no P.V MMA, 2 no Q.K MMA, 3 loads only; QK8: Q.K on FP8 tensor cores
        if ABL == 2 or ABL == 3:
            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32) + tl.sum(krt.to(tl.float32), 1)[None, :] * 1e-6
        elif QK8:
            q8 = q.to(tl.float8e4nv)
            qr8 = qr.to(tl.float8e4nv)
            k8 = tl.load(base[:, None] + dn[None, :], mask=valid[:, None], other=0.0)
            kr8 = tl.load(base[:, None] + DN + dr[None, :], mask=valid[:, None], other=0.0)
            qk = (tl.dot(q8, tl.trans(k8.to(tl.float8e4nv))) + tl.dot(qr8, tl.trans(kr8.to(tl.float8e4nv)))) * sm_scale
        else:
            qk = (tl.dot(q, kn) + tl.dot(qr, kr)) * sm_scale
        qk = tl.where(mh[:, None] & valid[None, :], qk, -1.0e30)
        n_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp2(e_max - n_max)
        p = tl.where(valid[None, :], tl.exp2(qk - n_max[:, None]), 0.0)
        if ABL == 1 or ABL == 3:
            acc = acc * re[:, None] + kt.to(tl.float32) * 1e-6   # BLOCK_H == BLOCK_N: keeps the row loads live
        elif DVS == 1:
            acc = acc * re[:, None] + tl.dot(p.to(CD), tl.trans(kn))
        else:   # this program's latent column slice of the same rows (just loaded: L1/L2 hits)
            kv = tl.load(base[:, None] + vs * DNV + dv[None, :], mask=valid[:, None], other=0.0).to(CD)
            acc = acc * re[:, None] + tl.dot(p.to(CD), kv)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_max
    ob = (t * SPLIT + sp) * HP + heads
    if vs == 0:
        tl.store(m_ptr + ob, e_max, mask=mh)
        tl.store(l_ptr + ob, e_sum, mask=mh)
    tl.store(acc_ptr + ob[:, None] * DN + vs * DNV + dv[None, :], acc, mask=mh[:, None])


@triton.jit
def _combine_kernel(m_ptr, l_ptr, acc_ptr, out_ptr, stride_o_t, stride_o_h, h_q, kv_scale,
                    SPLIT: tl.constexpr, BLOCK_H: tl.constexpr, HP: tl.constexpr, DN: tl.constexpr):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = heads < h_q
    dn = tl.arange(0, DN)
    M = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    for sp in range(SPLIT):
        M = tl.maximum(M, tl.load(m_ptr + (t * SPLIT + sp) * HP + heads, mask=mh, other=-1.0e30))
    L = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, DN], dtype=tl.float32)
    for sp in range(SPLIT):
        b = (t * SPLIT + sp) * HP + heads
        w = tl.exp2(tl.load(m_ptr + b, mask=mh, other=-1.0e30) - M)
        L += w * tl.load(l_ptr + b, mask=mh, other=0.0)
        acc += w[:, None] * tl.load(acc_ptr + b[:, None] * DN + dn[None, :], mask=mh[:, None], other=0.0)
    acc = acc / tl.maximum(L, 1e-30)[:, None] * kv_scale
    tl.store(out_ptr + t * stride_o_t + heads[:, None] * stride_o_h + dn[None, :], acc.to(tl.bfloat16), mask=mh[:, None])


# ---- request-packed decode attention (ARC_OFFLOAD_PACKED)
@triton.jit
def _union_kernel(s_ptr, perm_ptr, u_ptr, mk_ptr, cnt_ptr, N: tl.constexpr, TOPK: tl.constexpr):
    """one request: s = its QL x TOPK row ids sorted (invalid = INT32_MAX, last), perm = their source position ->
    u = distinct rows compacted (-1 padded), mk = bitmask of the tokens that selected each, cnt = #distinct"""
    r = tl.program_id(0)
    i = tl.arange(0, N)
    s = tl.load(s_ptr + r * N + i)
    prev = tl.load(s_ptr + r * N + i - 1, mask=i > 0, other=-1)
    valid = s != 2147483647
    head = valid & (s != prev)
    pos = tl.cumsum(head.to(tl.int32), 0) - 1
    tot = tl.sum(head.to(tl.int32), 0)
    j = (tl.load(perm_ptr + r * N + i) // TOPK).to(tl.int32)
    tl.store(u_ptr + r * N + pos, s, mask=head)
    tl.store(u_ptr + r * N + i, tl.full([N], -1, tl.int32), mask=i >= tot)
    tl.atomic_or(mk_ptr + r * N + tl.where(valid, pos, 0), tl.full([N], 1, tl.int32) << j, mask=valid)
    tl.store(cnt_ptr + r, tot)


@triton.jit
def _dsa_pk_kernel(q_ptr, host_ptr, cache_ptr, spill_ptr, u_ptr, mk_ptr, res_ptr, cnt_ptr, m_ptr, l_ptr, acc_ptr,
                   stride_q_t, stride_q_h, H, R, sm_scale,
                   QL: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DN: tl.constexpr,
                   DR: tl.constexpr, DV: tl.constexpr, KS: tl.constexpr, QCHUNK: tl.constexpr = 1,
                   BD: tl.constexpr = 64, HG: tl.constexpr = 16, NHB: tl.constexpr = 1):
    """program (request, head group x latent slice, split): QL tokens x HG heads of the request vs 1/KS of its union
    rows (HG = H when QL * H fits the tile, else H is covered by H / HG head groups)"""
    r = tl.program_id(0)
    NDV: tl.constexpr = DN // DV
    dvb = tl.program_id(1) % NDV
    hb = tl.program_id(1) // NDV
    ks = tl.program_id(2)
    D: tl.constexpr = DN + DR
    m = tl.arange(0, BM)
    j = m // HG
    mm = j < QL
    qrow = q_ptr + (r * QL + j).to(tl.int64) * stride_q_t + (hb * HG + m % HG) * stride_q_h
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    dv = tl.arange(0, DV)
    if not QCHUNK:
        qn = tl.load(qrow[:, None] + dn[None, :], mask=mm[:, None], other=0.0)
        qr = tl.load(qrow[:, None] + DN + dr[None, :], mask=mm[:, None], other=0.0)
    bd = tl.arange(0, BD)
    cnt = tl.load(cnt_ptr + r)
    chunk = tl.cdiv(tl.cdiv(cnt, KS), BN) * BN
    lo = ks * chunk
    hi = tl.minimum(lo + chunk, cnt)
    e_max = tl.zeros([BM], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DV], dtype=tl.float32)
    for k0 in range(lo, hi, BN):
        k = k0 + tl.arange(0, BN)
        kin = k < hi
        h = tl.load(u_ptr + r * N + k, mask=kin, other=-1).to(tl.int64)
        valid = kin & (h >= 0) & (h < R)
        rr = tl.load(res_ptr + r * N + k, mask=valid, other=-1).to(tl.int64)
        base = tl.where(rr >= 0, cache_ptr + rr * D, tl.where(rr <= -2, spill_ptr + (-rr - 2) * D, host_ptr + h * D))
        if QCHUNK:   # q streamed from L1/L2 in BD-wide chunks: only acc and the score tile stay in registers
            qk = tl.zeros([BM, BN], dtype=tl.float32)
            for d0 in range(0, D, BD):
                dm = d0 + bd < D                     # D = 576 is not a multiple of BD = 128
                qa = tl.load(qrow[:, None] + d0 + bd[None, :], mask=mm[:, None] & dm[None, :], other=0.0)
                ka = tl.load(base[:, None] + d0 + bd[None, :], mask=valid[:, None] & dm[None, :],
                             other=0.0).to(tl.bfloat16)
                qk = tl.dot(qa, tl.trans(ka), qk)
            qk = qk * sm_scale
        else:
            kt = tl.load(base[:, None] + dn[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            krt = tl.load(base[:, None] + DN + dr[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            qk = (tl.dot(qn, tl.trans(kt)) + tl.dot(qr, tl.trans(krt))) * sm_scale
        bits = tl.load(mk_ptr + r * N + k, mask=valid, other=0)
        ok = (((bits[None, :] >> j[:, None]) & 1) != 0) & mm[:, None]
        qk = tl.where(ok, qk, -1.0e30)
        n_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp2(e_max - n_max)
        p = tl.where(ok, tl.exp2(qk - n_max[:, None]), 0.0)
        if DV == DN and not QCHUNK:
            v = kt
        else:
            v = tl.load(base[:, None] + dvb * DV + dv[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * re[:, None] + tl.dot(p.to(tl.bfloat16), v)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_max
    o = ((r * NHB + hb) * KS + ks) * BM + m
    tl.store(m_ptr + o, e_max)
    tl.store(l_ptr + o, e_sum)
    tl.store(acc_ptr + o[:, None] * DN + dvb * DV + dv[None, :], acc)


@triton.jit
def _dsa_pk_combine_kernel(m_ptr, l_ptr, acc_ptr, out_ptr, stride_o_t, stride_o_h, H, kv_scale,
                           QL: tl.constexpr, BM: tl.constexpr, DN: tl.constexpr, DV: tl.constexpr, KS: tl.constexpr,
                           HG: tl.constexpr = 16, NHB: tl.constexpr = 1):
    r = tl.program_id(0)
    NDV: tl.constexpr = DN // DV
    dvb = tl.program_id(1) % NDV
    hb = tl.program_id(1) // NDV
    m = tl.arange(0, BM)
    j = m // HG
    mm = j < QL
    dv = tl.arange(0, DV)
    M = tl.zeros([BM], dtype=tl.float32) - 1.0e30
    for ks in range(KS):
        M = tl.maximum(M, tl.load(m_ptr + ((r * NHB + hb) * KS + ks) * BM + m))
    L = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DV], dtype=tl.float32)
    for ks in range(KS):
        o = ((r * NHB + hb) * KS + ks) * BM + m
        w = tl.exp2(tl.load(m_ptr + o) - M)
        L += w * tl.load(l_ptr + o)
        acc += w[:, None] * tl.load(acc_ptr + o[:, None] * DN + dvb * DV + dv[None, :])
    acc = acc / tl.maximum(L, 1e-30)[:, None] * kv_scale
    t = (r * QL + j).to(tl.int64)
    tl.store(out_ptr + t[:, None] * stride_o_t + (hb * HG + m % HG)[:, None] * stride_o_h + dvb * DV + dv[None, :],
             acc.to(tl.bfloat16), mask=mm[:, None])


_PK = {}
_PK_SEEN = False


def _pk_bufs(nreq, N, BM, device):
    b = _PK.get(device)
    if b is None or b["u"].numel() < nreq * N or b["m"].numel() < nreq * _PK_KS * BM:
        n = nreq * N    # BM here = rows per request over all head groups
        b = _PK[device] = dict(u=torch.empty(n, dtype=torch.int32, device=device),
                               mk=torch.empty(n, dtype=torch.int32, device=device),
                               cnt=torch.empty(nreq, dtype=torch.int32, device=device),
                               m=torch.empty(nreq * _PK_KS * BM, device=device),
                               l=torch.empty(nreq * _PK_KS * BM, device=device),
                               acc=torch.empty(nreq * _PK_KS * BM * D_NOPE, device=device))
    return b


# ---- W-way set-associative row cache (ARC_OFFLOAD_WAYS > 1): row h lives in set h % S (S = C / W), any of its W lines
@triton.jit
def _sa_scan_kernel(idx_ptr, tag_ptr, pin_ptr, ht_ptr, ml_ptr, mcnt_ptr, n, S, R, HTM, mls_ptr,
                    W: tl.constexpr, PROBES: tl.constexpr, BLOCK: tl.constexpr, RES: tl.constexpr = 0):
    """BLOCK rows per program: hits pin their line for this step; each distinct missing row is appended once to the
    miss list (GPU hash set; inactive lanes CAS a dummy slot ht[HTM + 1] that never matches)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    v = m & (h >= 0) & (h < R)
    st = tl.where(v, h % S, 0)
    w = tl.arange(0, W)
    mt = (tl.load(tag_ptr + st[:, None] * W + w[None, :], mask=v[:, None], other=-2) == h[:, None]).to(tl.int32)
    hit = v & (tl.max(mt, 1) > 0)
    tl.store(pin_ptr + st * W + tl.argmax(mt, 1), 1, mask=hit)
    act = v & (~hit)
    slot = (h * 2654435761) & HTM
    lane = HTM + 1 + tl.arange(0, BLOCK)       # private dummy slots: inactive lanes never contend
    p = 0
    while (tl.max(act.to(tl.int32), 0) > 0) & (p < PROBES):
        ptr = tl.where(act, ht_ptr + slot, ht_ptr + lane)
        old = tl.atomic_cas(ptr, tl.where(act, h * 0 - 1, h * 0 - 5), h)
        new = act & (old == -1)
        k = tl.atomic_add(mcnt_ptr + i * 0, 1, mask=new)
        tl.store(ml_ptr + k, h, mask=new)
        if RES:
            tl.store(mls_ptr + k, slot, mask=new)
        act = act & (old != -1) & (old != h)
        slot = (slot + 1) & HTM
        p += 1


@triton.jit
def _sa_fill_kernel(ml_ptr, mcnt_ptr, pin_ptr, rr_ptr, cnt_ptr, tag_ptr, host_ptr, cache_ptr, S,
                    mls_ptr, htv_ptr, spill_ptr, scnt_ptr, CAP,
                    W: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr, BLOCK: tl.constexpr, RES: tl.constexpr = 0):
    """BLOCK distinct missing rows per program: each takes a ticket in its set and replaces the ticket-th unpinned way
    (round-robin from rr[set]); rows beyond the set's unpinned ways stay uncached (attention reads them from host)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < tl.load(mcnt_ptr)
    h = tl.load(ml_ptr + i, mask=m, other=0)
    st = h % S
    k = tl.atomic_add(cnt_ptr + st, 1, mask=m)
    w = tl.arange(0, W)
    wr = (w[None, :] + tl.load(rr_ptr + st, mask=m, other=0)[:, None]) % W
    free = (tl.load(pin_ptr + st[:, None] * W + wr, mask=m[:, None], other=1) == 0).to(tl.int32)
    sel = (free == 1) & (tl.cumsum(free, 1) == (k + 1)[:, None])
    ok = m & (tl.max(sel.to(tl.int32), 1) > 0)
    line = st * W + tl.sum(tl.where(sel, wr, 0), 1)
    D: tl.constexpr = DN + DR
    dn = tl.arange(0, DN)
    dr = DN + tl.arange(0, DR)
    tl.store(cache_ptr + line[:, None] * D + dn[None, :], tl.load(host_ptr + h[:, None] * D + dn[None, :], mask=ok[:, None]),
             mask=ok[:, None])
    tl.store(cache_ptr + line[:, None] * D + dr[None, :], tl.load(host_ptr + h[:, None] * D + dr[None, :], mask=ok[:, None]),
             mask=ok[:, None])
    tl.store(tag_ptr + line, h, mask=ok)
    if RES:   # no free way this step: copy into the spill buffer (or leave on host past CAP); note where in htv
        need = m & (~ok)
        sp = tl.atomic_add(scnt_ptr + i * 0, 1, mask=need)
        oks = need & (sp < CAP)
        sp64 = tl.where(oks, sp, 0).to(tl.int64)
        tl.store(spill_ptr + sp64[:, None] * D + dn[None, :],
                 tl.load(host_ptr + h[:, None] * D + dn[None, :], mask=oks[:, None]), mask=oks[:, None])
        tl.store(spill_ptr + sp64[:, None] * D + dr[None, :],
                 tl.load(host_ptr + h[:, None] * D + dr[None, :], mask=oks[:, None]), mask=oks[:, None])
        tl.store(htv_ptr + tl.load(mls_ptr + i, mask=need, other=0), tl.where(oks, -sp64 - 2, -1), mask=need)


@triton.jit
def _sa_advance_kernel(rr_ptr, cnt_ptr, S, W: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = s < S
    c = tl.load(cnt_ptr + s, mask=m, other=0)
    tl.store(rr_ptr + s, (tl.load(rr_ptr + s, mask=m, other=0) + tl.minimum(c, W)) % W, mask=m)
    tl.store(cnt_ptr + s, 0, mask=m)


@triton.jit
def _sa_resolve_kernel(idx_ptr, tag_ptr, ht_ptr, htv_ptr, res_ptr, n, S, R, HTM,
                       W: tl.constexpr, PROBES: tl.constexpr, BLOCK: tl.constexpr):
    """After the fill: res[i] = cache line of entry i (tag hit), else its spill row (-2 - k) or -1 (host) from htv."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    v = m & (h >= 0) & (h < R)
    st = tl.where(v, h % S, 0)
    w = tl.arange(0, W)
    mt = (tl.load(tag_ptr + st[:, None] * W + w[None, :], mask=v[:, None], other=-2) == h[:, None]).to(tl.int32)
    hit = v & (tl.max(mt, 1) > 0)
    res = tl.where(hit, st * W + tl.argmax(mt, 1), -1)
    act = v & (~hit)
    slot = (h * 2654435761) & HTM
    p = 0
    while (tl.max(act.to(tl.int32), 0) > 0) & (p < PROBES):
        cur = tl.load(ht_ptr + slot, mask=act, other=-1)
        found = act & (cur == h)
        res = tl.where(found, tl.load(htv_ptr + slot, mask=found, other=-1), res)
        act = act & (cur != h) & (cur != -1)
        slot = (slot + 1) & HTM
        p += 1
    tl.store(res_ptr + i, res.to(tl.int32), mask=m)


@triton.jit
def _refill_kernel(tag_ptr, host_ptr, cache_ptr, C, R, BLOCK: tl.constexpr):
    """mirror (re)sync: every cache line takes the host row its tag names (int64 words, 72 per 576 B row)"""
    ln = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = tl.load(tag_ptr + ln, mask=ln < C, other=-1)
    ok = (ln < C) & (h >= 0) & (h < R)
    c = tl.arange(0, 64)
    c2 = 64 + tl.arange(0, 8)
    src = host_ptr + tl.where(ok, h, 0)[:, None] * 72
    dst = cache_ptr + ln.to(tl.int64)[:, None] * 72
    tl.store(dst + c[None, :], tl.load(src + c[None, :], mask=ok[:, None]), mask=ok[:, None])
    tl.store(dst + c2[None, :], tl.load(src + c2[None, :], mask=ok[:, None]), mask=ok[:, None])


@triton.jit
def _wt_kernel(slot_ptr, res_ptr, n, host_ptr, cache_ptr, spill_ptr, R, BLOCK: tl.constexpr):
    """mirror write-through: new row slot[i] -> the line (res >= 0) / spill row (res <= -2) the source layer gave it"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(slot_ptr + i, mask=m, other=-1).to(tl.int64)
    d = tl.load(res_ptr + i, mask=m, other=-1).to(tl.int64)
    ok = m & (h >= 0) & (h < R) & (d != -1)
    c = tl.arange(0, 64)
    c2 = 64 + tl.arange(0, 8)
    src = host_ptr + tl.where(ok, h, 0)[:, None] * 72
    dl = tl.where(d >= 0, d, -d - 2)
    dst = tl.where((d >= 0)[:, None], cache_ptr + dl[:, None] * 72, spill_ptr + dl[:, None] * 72)
    tl.store(dst + c[None, :], tl.load(src + c[None, :], mask=ok[:, None]), mask=ok[:, None])
    tl.store(dst + c2[None, :], tl.load(src + c2[None, :], mask=ok[:, None]), mask=ok[:, None])


@triton.jit
def _sa_invalidate_kernel(slot_ptr, tag_ptr, S, R, W: tl.constexpr):
    """a (re)written host row must not be served from a stale cache line"""
    h = tl.load(slot_ptr + tl.program_id(0)).to(tl.int64)
    if (h >= 0) & (h < R):
        st = h % S
        w = tl.arange(0, W)
        t = tl.load(tag_ptr + st * W + w)
        tl.store(tag_ptr + st * W + w, h * 0 - 1, mask=t == h)


# ---------------------------------------------------------------- per-layer state
_WS = {}


_RS = {}
# ARC_OFFLOAD_SHARE_HT=1: one miss hash table (8 MiB) + miss list (up to T*topk int64, 16 MiB at 1024 tokens) per device
# instead of per layer: they are rebuilt from scratch in every _sa_prefill and the layers run back-to-back on one stream.
# Not with prefetch (it scans the next layer on a side stream while this layer's table is live).
_SHARE_HT = os.environ.get("ARC_OFFLOAD_SHARE_HT", "0") == "1" and not _PREFETCH
_SH = {}


def _shared_ht(device):
    b = _SH.setdefault(str(device), {})
    if b.get("ht") is None:
        b["ht"] = torch.full(((1 << 20) + 256,), -1, dtype=torch.int64, device=device)
    return b["ht"]


def _shared_ml(n, device):
    b = _SH.setdefault(str(device), {})
    if b.get("ml") is None or b["ml"].shape[0] < n:
        b["ml"] = torch.empty(n, dtype=torch.int64, device=device)
    return b["ml"]


def _res_bufs(n, device):
    """per-step resolve buffers shared by all offload layers (they run back-to-back): spill rows, spill counter,
    hash-slot value table, miss-list hash slots, per-entry resolution"""
    b = _RS.get(device)
    if b is None or b["res"].numel() < n:
        old = b
        b = _RS[device] = dict(
            spill=old["spill"] if old else _gpu_rows(_SPILL, device),
            scnt=old["scnt"] if old else torch.zeros(1, dtype=torch.int32, device=device),
            htv=old["htv"] if old else torch.full(((1 << 20) + 256,), -1, dtype=torch.int64, device=device),
            mls=torch.empty(n, dtype=torch.int64, device=device),
            res=torch.empty(n, dtype=torch.int32, device=device))
    return b


_CUDA_OK = None
_PKC = {}
_PKC_SEEN = False


def _pk_ok(g, topk, R):
    from . import offload_cuda
    return offload_cuda.pk_ok(g, topk, R)


def _pkc_bufs(n, nreq, device):
    """request-packed CUDA path: union rows / token masks / counts + split-K merge counters (zeroed once; the kernel
    leaves them zeroed), grow-only and shared by all offload layers"""
    b = _PKC.get(device)
    if b is None or b["u"].numel() < n or b["cnt"].numel() < nreq:
        n = max(n, b["u"].numel() if b else 0)
        nreq = max(nreq, b["cnt"].numel() if b else 0)
        b = _PKC[device] = dict(u=torch.empty(n, dtype=torch.int32, device=device),
                                mk=torch.empty(n, dtype=torch.uint8, device=device),
                                cnt=torch.empty(nreq, dtype=torch.int32, device=device),
                                sem=torch.zeros(nreq * 32, dtype=torch.int32, device=device))
    return b


def _cuda_ok():
    """the prebuilt CUDA kernel loads and the GPU has the shared memory it needs (sm_89+)"""
    global _CUDA_OK
    if _CUDA_OK is None:
        from . import offload_cuda
        _CUDA_OK = offload_cuda.available() and torch.cuda.get_device_capability() >= (8, 9)
        print(f"[offload] CUDA split kernel {'ON' if _CUDA_OK else 'unavailable: ' + str(offload_cuda._ERR)} "
              f"split={_CUDA_SPLIT} stages={_CUDA_STAGES}", flush=True)
    return _CUDA_OK


def _split_ws(n, device):
    """split-K partials (max, sum, acc), ONE grow-only buffer per device shared by all offload layers (they run
    back-to-back on one stream); per-layer/per-batch-size buffers cost GBs during CUDA-graph memory profiling"""
    ws = _WS.get(device)
    if ws is None or ws[0].numel() < n:
        ws = _WS[device] = (torch.empty(n, device=device), torch.empty(n, device=device),
                            torch.empty(n * D_NOPE, device=device))
    return ws



def _mem_available():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) * 1024
    return 0


class _HostReg:
    """cudaHostRegister'ed host buffer (GPU-visible through UVA), bypassing torch's caching pinned allocator
    (power-of-two rounding, never returned).
    With a `key`, all TP ranks of one engine share ONE buffer per layer: a MAP_SHARED file in ARC_OFFLOAD_SHM
    (default /dev/shm) named by the engine-core pid (common parent of the TP workers) -- every rank writes identical
    rows, so 1x instead of TPx host RAM. Fails fast instead of exhausting host RAM (pinned pages are resident)."""
    _created = []

    def __init__(self, nbytes, key=None):
        frac = float(os.environ.get("ARC_OFFLOAD_HOST_FRAC", "0.5"))
        avail = _mem_available()
        if avail and nbytes > frac * avail:
            raise RuntimeError(
                f"offload host buffer {nbytes / 2**30:.1f} GiB > {frac:.0%} of available host RAM "
                f"({avail / 2**30:.1f} GiB): the KV budget gives too many slots for host offload; lower "
                f"--gpu-memory-utilization / L4_GIB (or raise ARC_OFFLOAD_HOST_FRAC)")
        self.nbytes, self.path, self.mm = nbytes, None, None
        d = os.environ.get("ARC_OFFLOAD_SHM", "/dev/shm")
        shared_ok = key is not None and os.environ.get("ARC_OFFLOAD_SHARED", "1") == "1" and os.path.isdir(d)
        if shared_ok and os.environ.get("ARC_OFFLOAD_MEMFD") == "1":
            # anonymous shared memory (memfd) instead of a /dev/shm file: same 1x host RAM for all TP ranks, but not
            # bounded by the /dev/shm mount size (Kaggle 86 GiB, test pods 64 GiB) -- that bound, not host RAM, is
            # what capped the KV pool (host rows = all KV slots of the indexer group). /dev/shm only holds a tiny
            # rendezvous file ("pid fd" of the creating rank) that the other ranks open via /proc/<pid>/fd/<fd>.
            try:
                _cleanup_stale(d)
                self.mm = _memfd_shared(self, d, key, nbytes)
            except Exception as e:   # fall back to the /dev/shm file path below
                print(f"[arc] memfd offload buffer failed ({e!r}); using {d}", flush=True)
                self.mm = None
        if self.mm is not None:
            self.buf = torch.frombuffer(self.mm, dtype=torch.uint8)
            shared_ok = False
        elif shared_ok:
            _cleanup_stale(d)
            st = os.statvfs(d)
            # a too-small tmpfs (Docker default 64 MB) would SIGBUS on first touch: fall back to a private buffer
            shared_ok = st.f_bavail * st.f_frsize > nbytes * 1.05
            if not shared_ok:
                print(f"[arc] {d} too small for a {nbytes / 2**30:.1f} GiB shared offload buffer; using private memory",
                      flush=True)
        if shared_ok:
            import mmap, re as _re
            self.path = os.path.join(d, f"arcoff_{os.getppid()}_{_re.sub(r'[^A-Za-z0-9]+', '_', key)}_{nbytes}")
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if os.fstat(fd).st_size != nbytes:
                    os.ftruncate(fd, nbytes)
                self.mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
            self.buf = torch.frombuffer(self.mm, dtype=torch.uint8)
            _HostReg._created.append(self.path)
        elif self.mm is None:
            self.buf = torch.zeros(nbytes, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes}) failed: {err}")

    def __del__(self):
        try:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        except Exception:
            pass


def _memfd_shared(owner, d, key, nbytes):
    """ARC_OFFLOAD_MEMFD: map one memfd shared by all ranks of this engine (siblings under the engine-core pid).
    The first rank (under an flock) creates it and publishes "pid fd"; the others reopen /proc/<pid>/fd/<fd>.
    Every rank keeps its fd open, so the memory lives until the last rank exits."""
    import fcntl, mmap, re as _re
    base = os.path.join(d, f"arcoff_{os.getppid()}_{_re.sub(r'[^A-Za-z0-9]+', '_', key)}_{nbytes}")
    rv, lockp = base + ".memfd", base + ".lock"
    lf = os.open(lockp, os.O_RDWR | os.O_CREAT, 0o600)
    fd = None
    try:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            pid, n = open(rv).read().split()
            fd = os.open(f"/proc/{pid}/fd/{n}", os.O_RDWR)
            if os.fstat(fd).st_size != nbytes:
                os.close(fd)
                fd = None
        except (OSError, ValueError):
            fd = None
        if fd is None:
            fd = os.memfd_create("arcoff", 0)
            os.ftruncate(fd, nbytes)
            with open(rv + ".tmp", "w") as f:
                f.write(f"{os.getpid()} {fd}")
            os.replace(rv + ".tmp", rv)
            print(f"[arc] memfd offload buffer {nbytes / 2**30:.2f} GiB created ({key})", flush=True)
        _HostReg._created.extend([rv, lockp])
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        os.close(lf)
    owner._memfd = fd
    return mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)


def _cleanup_stale(d):
    """Remove shared offload files left by engines that no longer exist (crashes would otherwise pin RAM)."""
    import glob, re as _re
    for f in glob.glob(os.path.join(d, "arcoff_*")):
        m = _re.match(r"arcoff_(\d+)_", os.path.basename(f))
        if m and not os.path.exists(f"/proc/{m.group(1)}"):
            try:
                os.unlink(f)
            except OSError:
                pass


def _unlink_created():
    for f in _HostReg._created:
        try:
            os.unlink(f)   # mappings stay valid in processes that still hold them
        except OSError:
            pass


import atexit as _atexit
_atexit.register(_unlink_created)


def _registered_host(shape, dtype, key=None):
    n = 1
    for d in shape:
        n *= d
    reg = _HostReg(n * torch.empty((), dtype=dtype).element_size(), key)
    t = reg.buf.view(dtype).view(shape)
    t._arc_hostreg = reg   # keep registration alive with the tensor
    return t


class LayerOffload:
    _count = 0

    def __init__(self, num_host_rows: int, device, key=None):
        self._lid = LayerOffload._count
        LayerOffload._count += 1
        self._dump = None
        self.R = num_host_rows
        import re as _re
        m = _re.search(r"layers\.(\d+)\.", key or "")
        self._layer = int(m.group(1)) if m else -1      # backbone layer id (from the indexer cache prefix)
        # ARC_OFFLOAD_DUMP_LAYERS="1,17,...": dump these backbone layers (to path_L<layer>.npz; distinct sorted rows per
        # decode call + the step's stored slots), else the legacy first-layer dump
        dl = os.environ.get("ARC_OFFLOAD_DUMP_LAYERS", "")
        want = (self._layer in {int(x) for x in dl.split(",") if x}) if dl else self._lid == 0
        if _DUMP and want and torch.device(device).index in (0, None):
            import time
            path, after, calls, min_t = _DUMP.split(",")
            self._dump = dict(path=path + (f"_L{self._layer}" if dl else ""), t0=time.time() + float(after),
                              left=int(calls), min_t=int(min_t), ev=[], on=False, uniq=bool(dl))
        self.C = min(_CROWS_L.get(self._layer, CACHE_ROWS), num_host_rows)
        # ARC_OFFLOAD_AB feature 'rows': buffers sized for the larger of the per-layer size and ARC_OFFLOAD_CACHE_ROWS;
        # windows without 'rows' run at ARC_OFFLOAD_CACHE_ROWS (a size switch empties the cache, rows refetched on use)
        self._cmax = self.C
        self._cbase = min(CACHE_ROWS, num_host_rows)
        if _AB > 0 and self._cbase > self.C:
            self.C = self._cbase                   # allocate for the larger size; the first fill switches if needed
        # host rows: fp8 [R, 576], or compact uint8 [R, RB] (ARC_OFFLOAD_ROW_FMT; every host reader dequantizes)
        self.host = (_registered_host((self.R, _rf.RB), torch.uint8, key) if _rf.ON else
                     _registered_host((self.R, D_ROW), FP8, key))
        self.cache = _gpu_rows(self.C, device, zero=True)   # (ARC_OFFLOAD_CCACHE: compact lines)
        self.tag = torch.full((self.C,), -1, dtype=torch.int64, device=device)
        self.W = _WAYS if self.C % _WAYS == 0 else 1
        # per-line int64 scratch of the direct-mapped fill / write-through store; unused by the W-way cache with compact
        # host rows (ARC_OFFLOAD_ROW_FMT): ARC_OFFLOAD_WIN_LAZY=1 (default off) drops it (8 B/line, ~22 MB/GPU at
        # 2.7M lines -> funds ~36k more lines; bit-exact)
        _wl = _WIN_LAZY and self.W > 1 and _rf.ON
        self.win = torch.empty((1 if _wl else self.C,), dtype=torch.int64, device=device)
        self.stats = torch.zeros(2, dtype=torch.int64, device=device)
        if self.W > 1:   # set-associative state
            self.pin = torch.zeros(self.C, dtype=torch.int32, device=device)
            self.rr = torch.zeros(self.C // self.W, dtype=torch.int32, device=device)
            self.cnt = torch.zeros(self.C // self.W, dtype=torch.int32, device=device)
            self.ht = (_shared_ht(device) if _SHARE_HT else
                       torch.full(((1 << 20) + 256,), -1, dtype=torch.int64, device=device))   # + per-lane dummies
            self.mcnt = torch.zeros(1, dtype=torch.int32, device=device)
            self.ml = None
        self._next = None     # the offload layer that runs after this one in a decode step (learned at runtime)
        self._pf = None       # (event, rows) of a prefetch into this layer still in flight
        try:                  # this layer reuses the previous DSA layer's top-k (ARC_INDEX_SHARE 'S'): exact prefetch target
            from .index_share import source_prefix
            self._srck = source_prefix(key) if key is not None else None
            self._shared = self._srck is not None
        except Exception:
            self._shared = False
            self._srck = None
        # mirror state (ARC_OFFLOAD_MIRROR): as a source, _mon = mirror phase decided at store(), _gen = decode calls,
        # _mrec = this call's fill record; as an S layer, _mgen = source call its cache mirrors (None: needs a refill)
        self.key = key
        self._mon = None
        self._gen = 0
        self._mrec = None
        self._mgen = None
        self._mkey = None
        self._lcall = 0
        if _rf.ON:
            _rf.check(self)
        from . import link_gate   # ARC_OFFLOAD_AR_GATE: allocate the gate word now (before any CUDA-graph capture)
        link_gate.word(device)
        if key is not None:
            _REG[key] = self

    def _join(self):
        if self._pf is not None:
            # a prefetch recorded in another capture state (graph capture vs eager) is not joinable: drop it (captured
            # work never ran; eager work finished before the capture's warmup sync)
            if self._pf[2] == torch.cuda.is_current_stream_capturing():
                torch.cuda.current_stream().wait_event(self._pf[0])
            self._pf = None

    def store(self, kv_c_normed, k_pe, slots, kv_scale):
        self._join()
        n = min(kv_c_normed.shape[0], slots.shape[0])
        if n == 0:
            return
        kpe = k_pe.reshape(k_pe.shape[0], -1)
        if self._dump is not None and self._dump["on"]:
            self._dump["ev"].append(("s", slots[:n].int().cpu().numpy()))
        args = (1.0 / kv_scale, self.C, self.R, kv_c_normed.stride(0), kpe.stride(0))
        if _rf.ON:   # compact host row (W > 1: no cache write-through / line claim)
            _rf.store(self.host, kv_c_normed, kpe, slots, n, 1.0 / kv_scale, self.R)
        else:
            self.win.fill_(-1)
            _store_rows_kernel[(n,)](kv_c_normed, kpe, slots, self.host, self.win, *args, DN=D_NOPE, DR=D_ROPE)
        # (mirror: an S layer still invalidates its own tags -- unused while mirrored, current when it leaves the mirror;
        # store() is captured into the piecewise CUDA graphs, so it must not depend on the mirror state)
        if self.W > 1:   # no write-through: drop stale copies; new rows are fetched on first use
            # store() is captured into CUDA graphs: with a runtime size switch ('rows'), invalidate in both geometries
            for c in sorted({self._cbase - self._cbase % self.W, self._cmax} if _AB > 0 else {self.C}):
                _sa_invalidate_kernel[(n,)](slots, self.tag, c // self.W, self.R, W=self.W)
                if _FILLX:   # ARC_OFFLOAD_TWO_CHOICE: also the row's second set
                    from . import offload_fill
                    offload_fill.invalidate2(self, slots, n, c // self.W)
            return
        _store_cache_kernel[(n,)](kv_c_normed, kpe, slots, self.win, self.cache, self.tag, *args, DN=D_NOPE, DR=D_ROPE)

    def gather(self, rows, out):
        n = rows.shape[0]
        if _rf.ON:
            _rf.gather(self.host, rows, out, self.R)
            return
        if n:
            _gather_rows_kernel[(triton.cdiv(n, 32),)](self.host, rows, out, n, self.R, DN=D_NOPE, DR=D_ROPE, BLOCK=32)

    # ---- mirror of an index-shared layer's row cache (ARC_OFFLOAD_MIRROR)
    def _msrc(self):
        """the source F layer whose cache this S layer mirrors in this step, else None"""
        if self._srck is None or not _mirror_ok():
            return None
        src = _REG.get(self._srck)
        if (src is None or src is self or not src._mon or src.C != self.C or src.R != self.R or src.W != self.W
                or getattr(self, "_mdet", None) == src._mph):
            return None
        return src

    def _mirror_phase(self):
        """source layers, at the start of their decode fill: switch mirroring on / off from this step on"""
        if self._srck is not None or self.key is None:
            return
        on = ab_on("mirror") and _mirror_ok()
        if on == self._mon:
            return
        if _MASYNC:
            mjoin(self.cache.device)
        if self._mon is not None or on:
            for s in [o for o in _REG.values() if o._srck == self.key and o is not self]:
                if s.C != self.C or s.R != self.R or s.W != self.W:
                    continue
                if on:
                    s._mgen = None                          # refilled from these tags at its next mirrored call
                elif getattr(s, "_mdet", None) != getattr(self, "_mph", 0):   # (a detached S owns its tags already)
                    s._own_tags(self, refill=s._mgen != self._gen)
        self._mph = getattr(self, "_mph", 0) + 1
        self._mon = on
        print(f"[offload] mirror {'ON' if on else 'OFF'} for the S layers of {self.key}", flush=True)

    def _write_through(self, src, rec, slots):
        """MIRROR=2, S layer before attention (after joining its side copy): this step's new rows (written by this
        layer's store after the side copy may have read them) -> wherever the source's fill of this step put them"""
        n = slots.shape[0]
        if n == 0:
            return
        from . import offload_fill as of
        rb = rec["rb"]
        buf = _MSPILL.get(("wt", self.cache.device))
        if buf is None or buf.numel() < n:
            buf = _MSPILL[("wt", self.cache.device)] = torch.empty(max(n, 4096), dtype=torch.int32, device=self.cache.device)
        of._resolve_kernel[(triton.cdiv(n, 256),)](slots, src.tag, src.ht, rb["htv"], buf, n, self.C // self.W, self.R,
                                                   of.HTM, rec["ep"], buf, W=self.W, PROBES=16, BLOCK=256,
                                                   EP=int(of.EPOCH), P2=rec["p2"])
        if _rf.ON:
            _rf.write_through(slots, buf, n, self.host, self.cache, _mspill(self.cache.device), self.R)
            return
        _wt_kernel[(triton.cdiv(n, 32),)](slots, buf, n, self.host.view(torch.int64), self.cache.view(torch.int64),
                                          _mspill(self.cache.device).view(torch.int64), self.R, BLOCK=32)

    def _mirror_launch(self, rec):
        """MIRROR=2, source layer after its fill: start each in-sync S layer's copy of these misses on a side stream"""
        from . import offload_fill as of
        dev = self.cache.device
        for s in [o for o in _REG.values() if o._srck == self.key and o is not self]:
            if s._mgen != rec["gen"] - 1 or s._msrc() is not self:
                continue
            st = _side(dev)
            ev0 = torch.cuda.Event()
            ev0.record()
            sp = _mspill(dev)
            from . import offload_cuda
            cuda_copy = offload_cuda.overlap_available()
            with torch.cuda.stream(st):
                st.wait_event(ev0)
                if cuda_copy:   # CUDA copy with the max-smem carveout: co-runs with the attention / GEMM CTAs
                    offload_cuda.fill_copy(rec["ml"], rec["mcnt"], rec["cpd"], s.host, s.cache, sp, None, None,
                                           mirror_grid(dev, of._nsm(dev) * _MCTAS, of.OVL_RPW), st,
                                           perm=rec["perm"] if rec["ns"] > 0 else None, ns=rec["ns"], hfmt=_rf.FMTX,
                                           gate=_ar_gate(dev))
                elif _rf.ON:
                    _rf.copy(rec["ml"], rec["mcnt"], rec["cpd"], s.host, s.cache, sp, rec["perm"], rec["ns"],
                             max(1, of._nsm(dev) * _MCTAS), B=of.FILL_B, num_warps=of.FILL_WARPS)
                else:   # (a Triton copy here keeps the attention's 78 KB-smem CTAs off the SMs)
                    of._copy_kernel[(max(1, of._nsm(dev) * _MCTAS),)](
                        rec["ml"], rec["mcnt"], rec["cpd"], s.host.view(torch.int64), s.cache.view(torch.int64),
                        sp.view(torch.int64), rec["perm"], rec["ns"], B=of.FILL_B, PERM=int(rec["ns"] > 0),
                        num_warps=of.FILL_WARPS)
                ev1 = torch.cuda.Event()
                ev1.record(st)
            for t in (rec["ml"], rec["mcnt"], rec["cpd"], rec["perm"]):
                t.record_stream(st)
            _MPEND.setdefault(dev, []).append(ev1)
            s._mpf = rec["gen"]

    def _refill(self, tag):
        if _rf.ON:
            _rf.refill(self.host, tag, self.cache, self.R)
            return
        _refill_kernel[(triton.cdiv(self.C, 64),)](tag, self.host.view(torch.int64), self.cache.view(torch.int64),
                                                    self.C, self.R, BLOCK=64)

    def _own_tags(self, src, refill):
        """leave the mirror: take the source's tags as this cache's own (refill: its content first)"""
        if refill:
            self._refill(src.tag)
        self.tag.copy_(src.tag)
        self.pin.zero_()
        self.cnt.zero_()
        self._mgen = None

    def _mirror_fill(self, src, n, slots=None):
        """S layer: bring the source call's missing rows into this layer's cache / spill; returns the source's resolution"""
        rec = src._mrec
        if (rec is None or rec["gen"] != src._gen or rec["n"] != n or rec["mkey"] is None
                or rec["mkey"] != self._mkey):
            print(f"[offload] mirror mismatch at L{self._lid} (source call {None if rec is None else rec['gen']}/"
                  f"{src._gen}): detached until the next mirror phase", flush=True)
            self._own_tags(src, refill=True)     # the source's tags now include this step's allocations
            self._mdet = src._mph
            return None
        if self._mgen != rec["gen"] - 1:         # not in sync with the source's previous call: rebuild from its tags
            self._refill(src.tag)
        from . import offload_fill as of
        rb = rec["rb"]
        mjoin(self.cache.device)   # (overlapped fill: the source's copy may still be running)
        rb = dict(rb, lflag=None, sflag=None)   # this layer's copy completes before its attention: no waits
        if _MASYNC and getattr(self, "_mpf", None) == rec["gen"] and slots is not None:   # copied on the side stream
            self._write_through(src, rec, slots)
            self._mgen = rec["gen"]
            return dict(rb, spill=_mspill(self.cache.device))
        if rec["ns"] > 0:
            rec["perm"].record_stream(torch.cuda.current_stream())   # (made on the side stream when overlapped)
        if _rf.ON:
            _rf.copy(rec["ml"], rec["mcnt"], rec["cpd"], self.host, self.cache, rb["spill"], rec["perm"], rec["ns"],
                     of.copy_grid(self.cache.device, n), B=of.FILL_B, num_warps=of.FILL_WARPS)
            self._mgen = rec["gen"]
            return rb
        of._copy_kernel[(of.copy_grid(self.cache.device, n),)](
            rec["ml"], rec["mcnt"], rec["cpd"], self.host.view(torch.int64), self.cache.view(torch.int64),
            rb["spill"].view(torch.int64), rec["perm"], rec["ns"], B=of.FILL_B, PERM=int(rec["ns"] > 0),
            num_warps=of.FILL_WARPS)
        self._mgen = rec["gen"]
        return rb

    def _lstats(self, T, n, rb, src):
        self._lcall += 1
        if self._lcall % _LSTATS:
            return
        o = src if src is not None else self
        print(f"[offl] L{self._layer} C={self.C} T={T} n={n} hit={int(o.pin.sum())} miss={int(o.mcnt)} "
              f"spill={int(rb['scnt']) if rb is not None else -1} cap={spill_cap()} mir={int(src is not None)}",
              flush=True)

    def _size_phase(self):
        """ARC_OFFLOAD_AB 'rows': switch between the per-layer size and ARC_OFFLOAD_CACHE_ROWS"""
        if _AB <= 0 or self._cmax == self._cbase or self.W <= 1:
            return
        want = self._cmax if ab_on("rows") else self._cbase
        if want % self.W:
            want -= want % self.W
        if want == self.C:
            return
        mjoin(self.cache.device)
        for o in [self] + [o for o in _REG.values() if o._srck is not None and o._srck == self.key and o is not self]:
            if o._cmax != self._cmax or o.W != self.W:
                continue
            o.C = want
            o.tag.fill_(-1)
            o.pin.zero_()
            o.rr.zero_()
            o.cnt.zero_()
            if getattr(o, "_lflag", None) is not None:
                o._lflag.zero_()
            o._mgen = None
        if self._srck is None:
            print(f"[offload] L{self._layer} cache rows -> {want}", flush=True)

    def _fill_rows(self, flat, T, slots=None):
        """W-way cache: this decode call's rows into cache / spill, resolved per entry (mirror of the source layer's
        fill for index-shared layers under ARC_OFFLOAD_MIRROR). slots: this step's stored rows (MIRROR=2)."""
        self._size_phase()
        if _MIRROR:
            self._mirror_phase()          # source layers: mirror on / off from this step (eager, before the fill)
        src = self._msrc() if _MIRROR else None
        rb = self._mirror_fill(src, flat.numel(), slots) if src is not None else None
        if rb is None:
            src = None
            rb = self._sa_prefill(flat)
        if _LSTATS:
            self._lstats(T, flat.numel(), rb, src)
        return rb

    def attend(self, q, idx, sm_scale, kv_scale, num_heads, ql=1, mkey=None, slots=None, out=None):
        """q [T, H, 576] bf16; idx [T, TOPK] host rows (-1 invalid) -> out [T, H, 512]. ql: decode tokens per request
        (consecutive rows of q/idx), used by the request-packed path. mkey: identity of idx's inputs (metadata,
        top-k buffer, decode tokens) -- equal keys within one step mean identical rows (mirror check)."""
        self._mkey = mkey
        self._topk = idx.shape[-1]
        T, H, _ = q.shape
        if self._dump is not None:
            self._record(idx, T, slots)
        if (_CUDA_PK and ql > 1 and T and H == 32 and self.W > 1 and _RESOLVE and idx.dtype == torch.int32
                and idx.stride(1) == 1 and q.stride(2) == 1 and q.data_ptr() % 16 == 0 and q.stride(0) % 8 == 0
                and q.stride(1) % 8 == 0 and _cuda_ok() and _pk_ok(min(ql, _CUDA_PK_QL or ql), idx.shape[1], self.R)):
            return self._attend_cuda_pk(q, idx, sm_scale, kv_scale, min(ql, _CUDA_PK_QL or ql))[:, :num_heads]
        if (_PACKED and ql > 1 and T % ql == 0 and 64 % ql == 0 and H % min(H, 64 // ql) == 0 and self.W > 1 and _RESOLVE
                and D_NOPE % _PK_DV == 0 and (ql * idx.shape[1]) & (ql * idx.shape[1] - 1) == 0):
            return self._attend_packed(q, idx, sm_scale, kv_scale, ql)[:, :num_heads]
        if out is None or out.shape != (T, H, D_NOPE) or out.dtype != q.dtype:   # (out: ARC_OFFLOAD_PF_OUT slice)
            out = torch.empty((T, H, D_NOPE), dtype=q.dtype, device=q.device)
        if T:
            BH = min(_SPLIT_BH, triton.next_power_of_2(H))   # 32: one program covers all heads -> each row read once/token
            if _STATS:
                self._stat(idx)                       # hit rate of the cache as the step finds it
            rb = None
            if self.W > 1:
                self._join()
                rb = self._fill_rows(idx.reshape(-1), T, slots)
                if _PREFETCH:
                    self._prefetch_next(idx.reshape(-1))
                if _STATS and self._ncall % _STATS == 0 and self._lid < 3:   # true fill volume of the sampled call
                    print(f"[offfill] L{self._lid} T={T} misses={int(self.mcnt)} "
                          f"spill={int(rb['scnt']) if rb is not None else -1}", flush=True)
            elif _PREFILL:
                self._prefill(idx.reshape(-1))
            TOPK = idx.shape[1]
            cuda_path = (_CUDA and rb is not None and (H == 32 or (_HCHUNK and H % 32 == 0))
                         and TOPK % (_CUDA_SPLIT * 32) == 0
                         and idx.dtype == torch.int32 and q.stride(2) == 1 and idx.stride(1) == 1 and _cuda_ok())
            lf = rb.get("lflag") if rb is not None else None
            sf = rb.get("sflag") if rb is not None else None
            if rb is not None and rb.get("ovl") and not cuda_path:
                mjoin(q.device)        # overlapped fill but no CUDA attention here: finish the copy, drop the marks
                r = rb["res"][:idx.numel()]
                r.sub_(torch.where(r >= (1 << 30), 1 << 30, 0).to(r.dtype))
                lf = sf = None
            if _rf.ON and not cuda_path:
                raise RuntimeError(f"ARC_OFFLOAD_ROW_FMT: decode needs the CUDA split attention (T={T} H={H} "
                                   f"TOPK={TOPK} idx {idx.dtype})")
            if cuda_path:
                from . import offload_cuda
                _te = _oa2_time_begin(q.device)   # ARC_OFFLOAD_ATTN2_TIME (diagnostics)
                _arm = "v1"
                for h0 in range(0, H, 32):   # (H > 32: ARC_OFFLOAD_CUDA_HCHUNK head slices)
                    qh = q if H == 32 else q[:, h0:h0 + 32]
                    oh = out if H == 32 else out[:, h0:h0 + 32]
                    if _CUDA_SPLIT == 1 and _oa2_ok(T, idx, rb["res"], qh):   # ARC_OFFLOAD_ATTN2 (offload_cuda2.py)
                        from . import offload_cuda2
                        f8 = ab_on("attn2f8") if _AB > 0 else None   # None: ARC_OFFLOAD_ATTN2 = 2 -> FP8 P.V
                        _arm = "pv8" if (f8 if f8 is not None else _OA2 == 2) else "pv16"
                        offload_cuda2.attend(qh, self.host, self.cache, rb["spill"], idx, rb["res"],
                                             sm_scale * 1.4426950408889634 * kv_scale, self.R, oh, kv_scale,
                                             lflag=lf, sflag=sf, hfmt=_rf.FMTX, pvf8=f8)
                        offload_cuda2.compare(qh, self.host, self.cache, rb["spill"], idx, rb["res"],
                                              sm_scale * 1.4426950408889634 * kv_scale, self.R, oh, kv_scale,
                                              hfmt=_rf.FMT, pvf8=f8)
                    elif _rf.CC:
                        raise RuntimeError(f"ARC_OFFLOAD_CCACHE: compact GPU rows need the ATTN2 kernel 5 path "
                                           f"(refused at T={T} topk={TOPK})")
                    elif _CUDA_SPLIT == 1:   # the kernel writes the normalized output itself
                        offload_cuda.split_partials(qh, self.host, self.cache, rb["spill"], idx, rb["res"], None, 1,
                                                    sm_scale * 1.4426950408889634 * kv_scale, self.R,
                                                    stages=_CUDA_STAGES, out=oh, kv_scale=kv_scale, lflag=lf, sflag=sf,
                                                    hfmt=_rf.FMT)
                    else:
                        ws = _split_ws(T * _CUDA_SPLIT * 32, q.device)
                        offload_cuda.split_partials(qh, self.host, self.cache, rb["spill"], idx, rb["res"], ws,
                                                    _CUDA_SPLIT, sm_scale * 1.4426950408889634 * kv_scale, self.R,
                                                    stages=_CUDA_STAGES, lflag=lf, sflag=sf, hfmt=_rf.FMT)
                        _combine_kernel[(T, 1)](ws[0], ws[1], ws[2], oh, oh.stride(0), oh.stride(1), 32, kv_scale,
                                                SPLIT=_CUDA_SPLIT, BLOCK_H=32, HP=32, DN=D_NOPE, num_warps=4)
                if _te is not None:
                    from . import offload_cuda2
                    offload_cuda2.time_end(_te, _arm, T)
            elif _SPLIT > 1 and TOPK % (_SPLIT * 16) == 0:
                HB = triton.cdiv(H, BH)
                HP = HB * BH
                ws = _split_ws(T * _SPLIT * HP, q.device)
                dvs = _SPLIT_DVS if (rb is not None or _ROWMAJOR == 2) else 1
                _sparse_mla_offload_split_kernel[(T, HB, _SPLIT * dvs)](
                    q, self.host, self.cache, self.tag, idx, ws[0], ws[1], ws[2], self.C, self.R,
                    q.stride(0), q.stride(1), idx.stride(0), H, sm_scale * 1.4426950408889634 * kv_scale,
                    rb["res"] if rb else idx, rb["spill"] if rb else self.cache,
                    TOPK=TOPK, SPLIT=_SPLIT, BLOCK_H=BH, BLOCK_N=_SPLIT_BN, HP=HP, DN=D_NOPE, DR=D_ROPE,
                    W=self.W, ROWMAJOR=_ROWMAJOR, RES=int(rb is not None), DVS=dvs, CD=tl.float16 if _SPLIT_F16 else tl.bfloat16, num_warps=_SPLIT_WARPS,
                    num_stages=_SPLIT_STAGES)
                _combine_kernel[(T, HB)](ws[0], ws[1], ws[2], out, out.stride(0), out.stride(1), H, kv_scale,
                                         SPLIT=_SPLIT, BLOCK_H=BH, HP=HP, DN=D_NOPE, num_warps=4)
            else:
                _sparse_mla_offload_kernel[(T, triton.cdiv(H, BH))](
                    q, self.host, self.cache, self.tag, idx, out, self.C, self.R,
                    q.stride(0), q.stride(1), out.stride(0), out.stride(1), idx.stride(0),
                    H, sm_scale * 1.4426950408889634 * kv_scale, kv_scale,
                    TOPK=TOPK, BLOCK_H=BH, BLOCK_N=16, DN=D_NOPE, DR=D_ROPE, W=self.W, num_warps=8, num_stages=1)
            if not _PREFILL and self.W == 1:
                self._fill(idx.reshape(-1))
        return out[:, :num_heads]

    def _attend_packed(self, q, idx, sm_scale, kv_scale, ql):
        T, H, _ = q.shape
        global _PK_SEEN
        if not _PK_SEEN:
            _PK_SEEN = True
            print(f"[offload] request-packed decode attention: T={T} ql={ql} H={H} topk={idx.shape[1]}", flush=True)
        out = torch.empty((T, H, D_NOPE), dtype=q.dtype, device=q.device)
        if _STATS:
            self._stat(idx)
        nreq, TOPK = T // ql, idx.shape[1]
        N = ql * TOPK
        HG = min(H, 64 // ql)                         # heads per tile; H / HG head groups
        NHB = H // HG
        BM = max(16, triton.next_power_of_2(ql * HG))
        x = idx.reshape(nreq, N)
        x = torch.where((x >= 0) & (x < self.R), x, 2147483647)
        srt, perm = torch.sort(x, dim=1)
        b = _pk_bufs(nreq, N, BM * NHB, q.device)
        u, mk, cnt = b["u"][:nreq * N], b["mk"][:nreq * N], b["cnt"]
        mk.zero_()
        _union_kernel[(nreq,)](srt, perm, u, mk, cnt, N=N, TOPK=TOPK, num_warps=8)
        rb = self._sa_prefill(u)                     # cache fill + resolve on the distinct rows only
        NDV = D_NOPE // _PK_DV
        _dsa_pk_kernel[(nreq, NDV * NHB, _PK_KS)](
            q, self.host, self.cache, rb["spill"], u, mk, rb["res"], cnt, b["m"], b["l"], b["acc"],
            q.stride(0), q.stride(1), H, self.R, sm_scale * 1.4426950408889634 * kv_scale,
            QL=ql, N=N, BM=BM, BN=_PK_BN, DN=D_NOPE, DR=D_ROPE, DV=_PK_DV, KS=_PK_KS, QCHUNK=_PK_QCHUNK,
            BD=_PK_BD, HG=HG, NHB=NHB, num_warps=_PK_WARPS, num_stages=_PK_STAGES)
        _dsa_pk_combine_kernel[(nreq, NDV * NHB)](b["m"], b["l"], b["acc"], out, out.stride(0), out.stride(1), H,
                                                  kv_scale, QL=ql, BM=BM, DN=D_NOPE, DV=_PK_DV, KS=_PK_KS, HG=HG,
                                                  NHB=NHB, num_warps=4)
        return out

    def _attend_cuda_pk(self, q, idx, sm_scale, kv_scale, g):
        from . import offload_cuda
        T, H, _ = q.shape
        TOPK = idx.shape[1]
        global _PKC_SEEN
        if not _PKC_SEEN:
            _PKC_SEEN = True
            print(f"[offload] CUDA request-packed decode attention: T={T} tokens/request={g} topk={TOPK} "
                  f"ks={_CUDA_PK_KS} cfg={_CUDA_PK_CFG}", flush=True)
        if _STATS:
            self._stat(idx)
        nreq = (T + g - 1) // g
        nmax = g * TOPK
        b = _pkc_bufs(nreq * nmax, nreq, q.device)
        u = b["u"][:nreq * nmax]
        self._join()
        offload_cuda.pk_union(idx, g, None, g, nreq, self.R, u, b["mk"], b["cnt"], nmax)
        rb = self._sa_prefill(u)                     # cache fill + resolve on the distinct rows only
        if _PREFETCH:
            self._prefetch_next(idx.reshape(-1))
        out = torch.empty((T, H, D_NOPE), dtype=q.dtype, device=q.device)
        ws = _split_ws(T * _CUDA_PK_KS * 32, q.device) if _CUDA_PK_KS > 1 else None
        offload_cuda.pk_attend(q, self.host, self.cache, rb["spill"], u, b["mk"], rb["res"], b["cnt"], None, g, g, nreq,
                               nmax, ws, _CUDA_PK_KS, sm_scale * 1.4426950408889634 * kv_scale, self.R,
                               cfg=_CUDA_PK_CFG, out=out, kv_scale=kv_scale, sem=b["sem"])
        return out

    def _record(self, idx, T, slots=None):
        import time
        import numpy as np
        d = self._dump
        if not d["on"]:
            if T < d["min_t"] or time.time() < d["t0"]:
                return
            d["on"] = True
            print(f"[offdump] start T={T} C={self.C} W={self.W} R={self.R}", flush=True)
        if d["uniq"]:   # store() is captured into CUDA graphs: record this step's slots here
            if slots is not None:
                d["ev"].append(("s", slots.int().cpu().numpy()))
            f = idx.reshape(-1)
            d["ev"].append(("a", torch.unique(f[f >= 0]).int().cpu().numpy()))
        else:
            d["ev"].append(("a", idx.int().cpu().numpy()))
        d["left"] -= 1
        if d["left"] <= 0:
            kinds = np.array([1 if k == "a" else 0 for k, _ in d["ev"]], dtype=np.int8)
            np.savez_compressed(d["path"], kinds=kinds, C=self.C, W=self.W, R=self.R, T=T,
                     **{f"e{i}": a for i, (_, a) in enumerate(d["ev"])})
            print(f"[offdump] saved {len(d['ev'])} events to {d['path']}.npz", flush=True)
            self._dump = None

    def _stat(self, idx):
        """ARC_OFFLOAD_STATS=N: every N decode calls log this layer's row-cache hit rate (host sync; diagnostics)."""
        self._ncall = getattr(self, "_ncall", 0) + 1
        if self._ncall % _STATS:
            return
        f = idx.reshape(-1)
        f = f[(f >= 0) & (f < self.R)].long()
        if f.numel():
            hit = float(self._cached(f).float().mean())
            print(f"[offstats] C={self.C} rows={f.numel()} uniq={int(torch.unique(f).numel())} hit={hit:.3f}", flush=True)

    def _prefetch_next(self, flat):
        try:
            from vllm.forward_context import get_forward_context
            fc = id(get_forward_context())
        except Exception:
            return
        prev, pfc = _LAST.get(flat.device, (None, None))
        if (prev is not None and pfc == fc and prev is not self and prev._next is None and prev._lid < self._lid
                and prev.W > 1):
            prev._next = self       # learn the order from consecutive layers of one forward (never into the draft model)
            if self._shared and not _PF_LOGGED:
                _PF_LOGGED.append(1)
                print(f"[offload] prefetch: layer {prev._lid} warms index-shared layer {self._lid}", flush=True)
        _LAST[flat.device] = (self, fc)
        nxt = self._next
        if nxt is None or not nxt._shared or nxt.C != self.C or nxt.R != self.R:
            return     # only index-shared layers: a guessed row set evicts useful lines (local A/B: own-top-k layer +46% misses)
        st = _PF_STREAM.get(flat.device)
        if st is None:
            st = _PF_STREAM[flat.device] = torch.cuda.Stream(flat.device)
        ev0 = torch.cuda.Event()
        ev0.record()
        with torch.cuda.stream(st):
            st.wait_event(ev0)
            nxt._sa_warm(flat)
            ev1 = torch.cuda.Event()
            ev1.record(st)
        nxt._pf = (ev1, flat, torch.cuda.is_current_stream_capturing())   # rows stay alive until the join

    def _sa_warm(self, flat):
        """scan + fill without spill/resolve (rows that find no free way stay uncached); this layer's buffers only"""
        n = flat.shape[0]
        if self.ml is None or self.ml.shape[0] < n:
            self.ml = torch.empty(n, dtype=torch.int64, device=flat.device)
        S = self.C // self.W
        self.pin.zero_(); self.ht.fill_(-1); self.ht[1 << 20:] = -7; self.mcnt.zero_()
        HTM = (1 << 20) - 1
        _sa_scan_kernel[(triton.cdiv(n, 256),)](flat, self.tag, self.pin, self.ht, self.ml, self.mcnt, n, S, self.R, HTM,
                                                 self.ml, W=self.W, PROBES=16, BLOCK=256, RES=0)
        _sa_fill_kernel[(triton.cdiv(n, 64),)](self.ml, self.mcnt, self.pin, self.rr, self.cnt, self.tag, self.host,
                                                self.cache, S, self.ml, self.ht, self.cache, self.mcnt, 0,
                                                W=self.W, DN=D_NOPE, DR=D_ROPE, BLOCK=64, RES=0)
        _sa_advance_kernel[(triton.cdiv(S, 1024),)](self.rr, self.cnt, S, W=self.W, BLOCK=1024)

    def _sa_prefill(self, flat):
        if _FILLX:   # ARC_OFFLOAD_HT_EPOCH / ARC_OFFLOAD_FILL_SPLIT (offload_fill.py; same results)
            from . import offload_fill
            return offload_fill.sa_prefill(self, flat)
        n = flat.shape[0]
        if _SHARE_HT:
            self.ml = _shared_ml(n, flat.device)
        elif self.ml is None or self.ml.shape[0] < n:
            self.ml = torch.empty(n, dtype=torch.int64, device=flat.device)
        S = self.C // self.W
        self.pin.zero_(); self.ht.fill_(-1); self.ht[1 << 20:] = -7; self.mcnt.zero_()
        HTM = (1 << 20) - 1                 # power-of-two table mask; slot HTM + 1 is the dummy (inactive CAS lanes)
        rb = _res_bufs(n, flat.device) if _RESOLVE else None
        if rb is not None:
            rb["scnt"].zero_()
        mls = rb["mls"] if rb else self.ml
        _sa_scan_kernel[(triton.cdiv(n, 256),)](flat, self.tag, self.pin, self.ht, self.ml, self.mcnt, n, S, self.R, HTM,
                                                 mls, W=self.W, PROBES=16, BLOCK=256, RES=int(rb is not None))
        _sa_fill_kernel[(triton.cdiv(n, 64),)](self.ml, self.mcnt, self.pin, self.rr, self.cnt, self.tag, self.host,
                                                self.cache, S, mls, rb["htv"] if rb else self.ht,
                                                rb["spill"] if rb else self.cache, rb["scnt"] if rb else self.mcnt,
                                                spill_cap(), W=self.W, DN=D_NOPE, DR=D_ROPE, BLOCK=64,
                                                RES=int(rb is not None))
        _sa_advance_kernel[(triton.cdiv(S, 1024),)](self.rr, self.cnt, S, W=self.W, BLOCK=1024)
        if rb is not None:
            _sa_resolve_kernel[(triton.cdiv(n, 256),)](flat, self.tag, self.ht, rb["htv"], rb["res"], n, S, self.R,
                                                        HTM, W=self.W, PROBES=16, BLOCK=256)
        return rb

    def _cached(self, f):
        """bool per row id: currently in the cache"""
        S = self.C // self.W
        return (self.tag.view(S, self.W)[f % S] == f[:, None]).any(1)

    def _prefill(self, flat):
        """bring this step's missing rows into the cache BEFORE attention (one deduplicated bulk host gather);
        attention then reads from GPU except rows that lost a same-line conflict"""
        n = flat.shape[0]
        self.win.fill_(-1)
        _claim_kernel[(triton.cdiv(n, 1024),)](flat, self.tag, self.win, n, self.C, self.R, BLOCK=1024)
        _prefill_kernel[(n,)](flat, self.win, self.tag, self.host, self.cache, n, self.C, self.R, DN=D_NOPE, DR=D_ROPE)

    def _fill(self, flat):
        n = flat.shape[0]
        self.win.fill_(-1)
        B = 1024
        _propose_kernel[(triton.cdiv(n, B),)](flat, self.tag, self.win, n, self.C, self.R, BLOCK=B)
        _fill_kernel[(n,)](flat, self.win, self.tag, self.host, self.cache, n, self.C, self.R, DN=D_NOPE, DR=D_ROPE)

    def hit_rate(self, idx):
        """Diagnostics only (host sync)."""
        f = idx.reshape(-1)
        f = f[f >= 0]
        return float(self._cached(f.long()).float().mean()) if f.numel() else 1.0


# ---------------------------------------------------------------- host-resident vocab embeddings (ARC_EMBED_HOST=1)
@triton.jit
def _embed_gather_kernel(ids_ptr, w_ptr, out_ptr, start, end, D, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    tok = tl.load(ids_ptr + t).to(tl.int64)
    ok = (tok >= start) & (tok < end)
    row = tl.where(ok, tok - start, 0)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    v = tl.load(w_ptr + row * D + c, mask=m & ok, other=0.0)
    tl.store(out_ptr + t.to(tl.int64) * D + c, v, mask=m)


def host_embeddings(model):
    """Move every VocabParallelEmbedding (not LM heads) of `model` to host memory; lookups read rows over UVA.
    Only org-vocab rows are supported (no added/LoRA vocab)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    from vllm.distributed import tensor_model_parallel_all_reduce
    moved = 0
    for name, mod in model.named_modules():
        if type(mod) is not VocabParallelEmbedding or not mod.weight.is_cuda:
            continue
        w = mod.weight.data
        host = _registered_host(tuple(w.shape), w.dtype)
        host.copy_(w)
        moved += w.numel() * w.element_size()
        mod.weight = torch.nn.Parameter(host, requires_grad=False)
        si = mod.shard_indices
        start, end, D, dev = si.org_vocab_start_index, si.org_vocab_end_index, w.shape[1], w.device

        def fwd(input_, mod=mod, host=host, start=start, end=end, D=D, dev=dev):
            ids = input_.reshape(-1)
            out = torch.empty((ids.shape[0], D), dtype=host.dtype, device=dev)
            if ids.shape[0]:
                _embed_gather_kernel[(ids.shape[0], triton.cdiv(D, 1024))](ids, host, out, start, end, D, BLOCK=1024)
            out = out.view(*input_.shape, D)
            return tensor_model_parallel_all_reduce(out) if mod.tp_size > 1 else out
        mod.forward = fwd
        del w
    torch.cuda.empty_cache()
    return moved


# ---------------------------------------------------------------- FP8 SWA prefill gather (ARC_SWA_FP8=1)
@triton.jit
def _gather_dequant_kernel(src_ptr, dst_ptr, bt_ptr, cu_ptr, t2s_ptr, starts_ptr, scale_ptr,
                           stride_blk, stride_tok, stride_bt, stride_dst, BS, Dw,
                           HAS_STARTS: tl.constexpr, BLOCK_D: tl.constexpr):
    i = tl.program_id(0)
    dblk = tl.program_id(1)
    seq = tl.load(t2s_ptr + i).to(tl.int64)
    pos = i - tl.load(cu_ptr + seq).to(tl.int64)
    if HAS_STARTS:
        pos += tl.load(starts_ptr + seq).to(tl.int64)
    blk = tl.load(bt_ptr + seq * stride_bt + pos // BS).to(tl.int64)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    m = d < Dw
    x = tl.load(src_ptr + blk * stride_blk + (pos % BS) * stride_tok + d, mask=m, other=0.0).to(tl.float32)
    tl.store(dst_ptr + i * stride_dst + d, (x * tl.load(scale_ptr)).to(dst_ptr.dtype.element_ty), mask=m)


def install_fp8_gather():
    """vLLM's gather_and_maybe_dequant_cache only supports head_dim 320/512/576; Dots3 SWA rows are 1088 wide."""
    from vllm import _custom_ops as ops
    orig = ops.gather_and_maybe_dequant_cache

    def gather(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale,
               seq_starts=None):
        Dw = src_cache.shape[-1]
        if Dw in (320, 512, 576) or not str(kv_cache_dtype).startswith("fp8"):
            return orig(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale,
                        seq_starts)
        src = src_cache.view(torch.float8_e4m3fn) if src_cache.dtype == torch.uint8 else src_cache
        if src.dim() == 4:
            src = src.squeeze(2)
        n = int(num_tokens)
        if n == 0:
            return
        BLOCK_D = 256
        _gather_dequant_kernel[(n, triton.cdiv(Dw, BLOCK_D))](
            src, dst, block_table, cu_seq_lens, token_to_seq,
            seq_starts if seq_starts is not None else cu_seq_lens, scale,
            src.stride(0), src.stride(1), block_table.stride(0), dst.stride(0), src.shape[1], Dw,
            HAS_STARTS=seq_starts is not None, BLOCK_D=BLOCK_D)
    ops.gather_and_maybe_dequant_cache = gather
