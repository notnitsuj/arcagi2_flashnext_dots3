"""Alternative scan / fill / resolve for the W-way offload row cache (sparse_offload.LayerOffload._sa_prefill).
Enabled per flag (default off; all variants produce the same cache state, spill rows and resolution as the default):

  ARC_OFFLOAD_HT_EPOCH=1   epoch-tagged miss hash table: entries are (epoch << 32 | row); a slot whose epoch is not the
                           current call's counts as empty, so the 8 MiB table is never reset (the default path memsets
                           it every layer call: 13 x 8 MiB of writes per decode step). The epoch counter lives in device
                           memory next to the table (keyed by the table's address, so layers that share one table
                           share its epoch) and is bumped by a 1-program kernel, so this is CUDA-graph safe.
  ARC_OFFLOAD_FILL_SPLIT=1 split the fill into a slot-allocation kernel (same ticket / round-robin / spill logic) and a
                           persistent PCIe gather kernel (int64 words, ARC_OFFLOAD_FILL_B rows per iteration,
                           ARC_OFFLOAD_FILL_WARPS warps, ARC_OFFLOAD_FILL_CTAS CTAs per SM), so the host loads are not
                           serialized behind the per-set atomics and no CTA is launched per 64 entries of the (mostly
                           hit) entry list. ARC_OFFLOAD_FILL_SORT=N: the copy visits the first N misses in ascending
                           host-row order (torch.sort over a fixed N-entry window; locality for the GPU TLB / page
                           walks over the multi-GB host buffers).
  ARC_OFFLOAD_TWO_CHOICE=1 two-choice set placement (needs ARC_OFFLOAD_RESOLVE=1, implies the split fill): a missing row
                           that finds no free (unpinned) way in its set h % S takes one in a second, hashed set; lookups
                           check the second set only for entries that miss the first. On the real-L4 decode trace
                           (run 27, 48 seqs, 131072 x 8) misses/call 5469 -> 2353 and spills 3851 -> 423
                           (row_cache_sim: W16 gives 3572). Stores invalidate both sets (sparse_offload.store hook).
                           The prefetch warm path (ARC_OFFLOAD_PREFETCH) still places by the first set only (a row can
                           then be cached twice; lookups stay correct).
"""
import os

import torch
import triton
import triton.language as tl

EPOCH = os.environ.get("ARC_OFFLOAD_HT_EPOCH", "0") == "1"
SPLIT = os.environ.get("ARC_OFFLOAD_FILL_SPLIT", "0") == "1"
TWO = os.environ.get("ARC_OFFLOAD_TWO_CHOICE", "0") == "1"
FILL_B = int(os.environ.get("ARC_OFFLOAD_FILL_B", "32"))
FILL_WARPS = int(os.environ.get("ARC_OFFLOAD_FILL_WARPS", "4"))
FILL_CTAS = int(os.environ.get("ARC_OFFLOAD_FILL_CTAS", "8"))
SORT = int(os.environ.get("ARC_OFFLOAD_FILL_SORT", "0"))     # >0: sort window (misses) of the split copy
SORT32 = os.environ.get("ARC_OFFLOAD_FILL_SORT32", "0") == "1"   # sort the window on int32 keys
# ARC_OFFLOAD_OVERLAP=1 (needs the split fill, resolve and the CUDA attention): the fill copy (csrc fill_copy_kernel,
# miss-list order ~ token order, no address sort) runs on a side stream concurrently with resolve + attention; alloc
# marks each filled line / spill row pending, the copy clears the mark after writing the row, and the attention waits
# for the mark of rows still pending (resolve tags them) before loading them. Same rows, same order: bit-exact.
OVERLAP = os.environ.get("ARC_OFFLOAD_OVERLAP", "0") == "1"
OVL_GRID = int(os.environ.get("ARC_OFFLOAD_OVL_GRID", "0"))   # copy CTAs (4 warps; 0 = one per SM): fit beside the attention
OVL_RPW = int(os.environ.get("ARC_OFFLOAD_OVL_RPW", "2"))     # rows per warp per iteration (1 | 2 | 4)
OVL_SORT = int(os.environ.get("ARC_OFFLOAD_OVL_SORT", "65536"))   # copy-order sort window (misses)
_FT = {}
_OKEY = {}


@triton.jit
def _okey_kernel(ml_ptr, mls_ptr, ft_ptr, mcnt_ptr, key_ptr, ns, NSM, K32: tl.constexpr, BLOCK: tl.constexpr):
    """overlapped-fill copy order key of miss i: (attention wave of its first token, host row); past mcnt: max"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < ns
    live = m & (i < tl.load(mcnt_ptr))
    v = tl.load(ft_ptr + tl.load(mls_ptr + i, mask=live, other=0), mask=live, other=0)
    wave = (2147483647 - (v & 0xFFFFFFFF)) // NSM
    h = tl.load(ml_ptr + i, mask=live, other=0)
    if K32:   # rows < 2^27, waves < 16
        key = tl.where(live, (tl.minimum(wave, 15) << 27) | h, 2147483647).to(tl.int32)
    else:
        key = tl.where(live, (wave << 32) | h, 9223372036854775807)
    tl.store(key_ptr + i, key, mask=m)


def _ftok(dev):
    t = _FT.get(dev)
    if t is None:
        t = _FT[dev] = torch.zeros((1 << HT_BITS) + 256, dtype=torch.int64, device=dev)
    return t
_OVOK = [None]


def _ovl_ok():
    if _OVOK[0] is None:
        from . import offload_cuda, sparse_offload as so
        _OVOK[0] = bool(offload_cuda.overlap_available() and so._CUDA and so._CUDA_STAGES in (2, 3)
                        and not so._CUDA_PK and not so._PACKED and so._cuda_ok())
        print(f"[offload] overlapped fill {'ON' if _OVOK[0] else 'unavailable (needs the CUDA 4-warp attention)'}",
              flush=True)
    return _OVOK[0]


def _flags(lo, dev):
    """pending marks: per layer [C] (cache lines), per device [spill rows]; all zero between calls"""
    from . import sparse_offload as so
    if getattr(lo, "_lflag", None) is None:
        lo._lflag = torch.zeros(lo.cache.shape[0], dtype=torch.int32, device=dev)
    sf = _SFL.get(dev)
    if sf is None:
        sf = _SFL[dev] = torch.zeros(so._SPILL, dtype=torch.int32, device=dev)
    return lo._lflag, sf


_SFL = {}
HT_BITS = 20
HTM = (1 << HT_BITS) - 1
_EP = {}      # ht.data_ptr() -> int64[1] epoch counter (shared tables share it)
_CPD = {}     # device -> int64 copy destinations (grow-only, shared by all layers)
_NSM = {}


def enabled():
    return EPOCH or SPLIT or TWO


def _epoch(ht):
    ep = _EP.get(ht.data_ptr())
    if ep is None:
        ht[HTM + 1:] = -7          # per-lane dummy CAS slots (never match the -5 compare value)
        ep = _EP[ht.data_ptr()] = torch.zeros(1, dtype=torch.int64, device=ht.device)
    return ep


def _nsm(dev):
    n = _NSM.get(dev)
    if n is None:
        n = _NSM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


@triton.jit
def _begin_kernel(pin_ptr, C, ep_ptr, mcnt_ptr, scnt_ptr, EP: tl.constexpr, RES: tl.constexpr, BLOCK: tl.constexpr):
    """zero this call's pins; program 0 also bumps the table epoch and zeroes the miss / spill counters"""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(pin_ptr + o, tl.zeros([BLOCK], dtype=tl.int32), mask=o < C)
    if tl.program_id(0) == 0:
        if EP:
            tl.store(ep_ptr, tl.load(ep_ptr) + 1)
        tl.store(mcnt_ptr, 0)
        if RES:
            tl.store(scnt_ptr, 0)


@triton.jit
def _set2(h, S):
    """second candidate set of row h (two-choice placement): independent of h % S"""
    return ((h * 2654435761) >> 12) % S


@triton.jit
def _scan_kernel(idx_ptr, tag_ptr, pin_ptr, ht_ptr, ml_ptr, mcnt_ptr, n, S, R, HTM, mls_ptr, ep_ptr, ftok_ptr, TOPK,
                 W: tl.constexpr, PROBES: tl.constexpr, BLOCK: tl.constexpr, RES: tl.constexpr, EP: tl.constexpr,
                 P2: tl.constexpr, FT: tl.constexpr = 0):
    """sparse_offload._sa_scan_kernel; EP: epoch-tagged table (a slot holding another epoch's key is free);
    P2: a row may also live in its second set (hit there pins that line).
    FT (with EP, ARC_OFFLOAD_OVERLAP): ftok[slot] = max over the row's missing entries of (epoch << 32 | 2^31-1 - token)
    = its first token in this call (token = entry // TOPK), the copy-order key"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    v = m & (h >= 0) & (h < R)
    st = tl.where(v, h % S, 0)
    w = tl.arange(0, W)
    mt = (tl.load(tag_ptr + st[:, None] * W + w[None, :], mask=v[:, None], other=-2) == h[:, None]).to(tl.int32)
    hit = v & (tl.max(mt, 1) > 0)
    tl.store(pin_ptr + st * W + tl.argmax(mt, 1), 1, mask=hit)
    if P2:
        v2 = v & (~hit)
        s2 = tl.where(v2, _set2(h, S), 0)
        mt2 = (tl.load(tag_ptr + s2[:, None] * W + w[None, :], mask=v2[:, None], other=-2) == h[:, None]).to(tl.int32)
        hit2 = v2 & (tl.max(mt2, 1) > 0)
        tl.store(pin_ptr + s2 * W + tl.argmax(mt2, 1), 1, mask=hit2)
        hit = hit | hit2
    act = v & (~hit)
    if EP:
        ep = tl.load(ep_ptr)
        ek = (ep << 32) | tl.where(v, h, 0)
    else:
        ep = 0
        ek = h
    slot = (h * 2654435761) & HTM
    lane = HTM + 1 + tl.arange(0, BLOCK)
    p = 0
    while (tl.max(act.to(tl.int32), 0) > 0) & (p < PROBES):
        cur = tl.load(ht_ptr + slot, mask=act, other=0, volatile=True)
        dup = act & (cur == ek)
        if EP:
            stale = act & ((cur >> 32) != ep)
        else:
            stale = act & (cur == -1)
        old = tl.atomic_cas(tl.where(stale, ht_ptr + slot, ht_ptr + lane), tl.where(stale, cur, h * 0 - 5),
                            tl.where(stale, ek, h * 0 - 5))
        new = stale & (old == cur)
        k = tl.atomic_add(mcnt_ptr + i * 0, 1, mask=new)
        tl.store(ml_ptr + k, h, mask=new)
        if RES:
            tl.store(mls_ptr + k, slot, mask=new)
        lost = stale & (~new)
        dup2 = lost & (old == ek)
        if FT:
            fv = (ep << 32) | (2147483647 - (i // TOPK)).to(tl.int64)
            tl.atomic_max(ftok_ptr + slot, fv, mask=dup | new | dup2)
        if EP:
            taken = (old >> 32) == ep
        else:
            taken = old != -1
        adv = (act & (~stale) & (~dup)) | (lost & taken & (old != ek))
        act = act & (~dup) & (~new) & (~dup2)
        slot = tl.where(adv, (slot + 1) & HTM, slot)       # lost a race for a free slot: re-read the same slot
        p += 1


@triton.jit
def _resolve_kernel(idx_ptr, tag_ptr, ht_ptr, htv_ptr, res_ptr, n, S, R, HTM, ep_ptr, lflag_ptr,
                    W: tl.constexpr, PROBES: tl.constexpr, BLOCK: tl.constexpr, EP: tl.constexpr, P2: tl.constexpr,
                    FLAG: tl.constexpr = 0):
    """FLAG (ARC_OFFLOAD_OVERLAP): a line still pending in lflag (being filled by the overlapped copy) resolves to
    line + 2^30, the attention's cue to wait for its flag"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    h = tl.load(idx_ptr + i, mask=m, other=-1).to(tl.int64)
    v = m & (h >= 0) & (h < R)
    st = tl.where(v, h % S, 0)
    w = tl.arange(0, W)
    mt = (tl.load(tag_ptr + st[:, None] * W + w[None, :], mask=v[:, None], other=-2) == h[:, None]).to(tl.int32)
    hit = v & (tl.max(mt, 1) > 0)
    res = tl.where(hit, st * W + tl.argmax(mt, 1), -1)
    if P2:
        v2 = v & (~hit)
        s2 = tl.where(v2, _set2(h, S), 0)
        mt2 = (tl.load(tag_ptr + s2[:, None] * W + w[None, :], mask=v2[:, None], other=-2) == h[:, None]).to(tl.int32)
        hit2 = v2 & (tl.max(mt2, 1) > 0)
        res = tl.where(hit2, s2 * W + tl.argmax(mt2, 1), res)
        hit = hit | hit2
    if FLAG:
        pend = tl.load(lflag_ptr + tl.where(hit, res, 0), mask=hit, other=0)
        res = tl.where(hit & (pend != 0), res + (1 << 30), res)
    act = v & (~hit)
    if EP:
        ep = tl.load(ep_ptr)
        ek = (ep << 32) | tl.where(v, h, 0)
    else:
        ep = 0
        ek = h
    slot = (h * 2654435761) & HTM
    p = 0
    while (tl.max(act.to(tl.int32), 0) > 0) & (p < PROBES):
        cur = tl.load(ht_ptr + slot, mask=act, other=0)
        found = act & (cur == ek)
        res = tl.where(found, tl.load(htv_ptr + slot, mask=found, other=-1), res)
        if EP:
            act = act & (cur != ek) & ((cur >> 32) == ep)
        else:
            act = act & (cur != ek) & (cur != -1)
        slot = (slot + 1) & HTM
        p += 1
    tl.store(res_ptr + i, res.to(tl.int32), mask=m)


@triton.jit
def _alloc_kernel(ml_ptr, mcnt_ptr, pin_ptr, rr_ptr, cnt_ptr, tag_ptr, cpd_ptr, S, mls_ptr, htv_ptr, scnt_ptr, CAP,
                  lflag_ptr, sflag_ptr, W: tl.constexpr, BLOCK: tl.constexpr, RES: tl.constexpr, P2: tl.constexpr,
                  FLAG: tl.constexpr = 0):
    """sparse_offload._sa_fill_kernel without the data movement: picks each missing row's line (or spill row) and
    records it in cpd (line >= 0, spill row k as -2 - k, -1 = stays on host). P2: a row that finds no free way in its
    first set takes a ticket in its second set."""
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
    if P2:
        m2 = m & (~ok)
        s2 = _set2(h, S)
        k2 = tl.atomic_add(cnt_ptr + s2, 1, mask=m2)
        wr2 = (w[None, :] + tl.load(rr_ptr + s2, mask=m2, other=0)[:, None]) % W
        free2 = (tl.load(pin_ptr + s2[:, None] * W + wr2, mask=m2[:, None], other=1) == 0).to(tl.int32)
        sel2 = (free2 == 1) & (tl.cumsum(free2, 1) == (k2 + 1)[:, None])
        ok2 = m2 & (tl.max(sel2.to(tl.int32), 1) > 0)
        line = tl.where(ok2, s2 * W + tl.sum(tl.where(sel2, wr2, 0), 1), line)
        ok = ok | ok2
    tl.store(tag_ptr + line, h, mask=ok)
    if FLAG:   # pending until the overlapped copy has written the row
        tl.store(lflag_ptr + line, tl.full([BLOCK], 1, tl.int32), mask=ok)
    d = tl.where(ok, line, -1)
    if RES:
        need = m & (~ok)
        sp = tl.atomic_add(scnt_ptr + i * 0, 1, mask=need)
        oks = need & (sp < CAP)
        sp64 = tl.where(oks, sp, 0).to(tl.int64)
        tl.store(htv_ptr + tl.load(mls_ptr + i, mask=need, other=0), tl.where(oks, -sp64 - 2, -1), mask=need)
        if FLAG:
            tl.store(sflag_ptr + sp64, tl.full([BLOCK], 1, tl.int32), mask=oks)
        d = tl.where(oks, -sp64 - 2, d)
    tl.store(cpd_ptr + i, d, mask=m)


@triton.jit
def _invalidate2_kernel(slot_ptr, tag_ptr, S, R, W: tl.constexpr):
    """store(): a rewritten host row must not stay cached in its second set either"""
    h = tl.load(slot_ptr + tl.program_id(0)).to(tl.int64)
    if (h >= 0) & (h < R):
        s2 = _set2(h, S)
        w = tl.arange(0, W)
        t = tl.load(tag_ptr + s2 * W + w)
        tl.store(tag_ptr + s2 * W + w, h * 0 - 1, mask=t == h)


def invalidate2(lo, slots, n, S=None):
    if TWO and n:
        _invalidate2_kernel[(n,)](slots, lo.tag, S if S is not None else lo.C // lo.W, lo.R, W=lo.W)


@triton.jit
def _copy_kernel(ml_ptr, mcnt_ptr, cpd_ptr, host_ptr, cache_ptr, spill_ptr, perm_ptr, NS, B: tl.constexpr,
                 PERM: tl.constexpr):
    """persistent PCIe gather of the allocated rows: host [R, 72] int64 words -> cache / spill lines.
    PERM: the first NS miss-list entries are visited in perm (ascending host row) order"""
    n = tl.load(mcnt_ptr)
    c = tl.arange(0, 64)
    c2 = 64 + tl.arange(0, 8)
    for s in range(tl.program_id(0) * B, n, tl.num_programs(0) * B):
        i = s + tl.arange(0, B)
        m = i < n
        if PERM:
            i = tl.where(i < NS, tl.load(perm_ptr + i, mask=m & (i < NS), other=0), i)
        h = tl.load(ml_ptr + i, mask=m, other=0).to(tl.int64)
        d = tl.load(cpd_ptr + i, mask=m, other=-1)
        ok = m & (d != -1)
        src = host_ptr + h[:, None] * 72
        a = tl.load(src + c[None, :], mask=ok[:, None])
        b = tl.load(src + c2[None, :], mask=ok[:, None])
        dl = tl.where(d >= 0, d, -d - 2)
        dst = tl.where((d >= 0)[:, None], cache_ptr + dl[:, None] * 72, spill_ptr + dl[:, None] * 72)
        tl.store(dst + c[None, :], a, mask=ok[:, None])
        tl.store(dst + c2[None, :], b, mask=ok[:, None])


@triton.jit
def _gather_kernel(rows_ptr, n_ptr, host_ptr, out_ptr, B: tl.constexpr):
    """bench reference: the copy kernel's loop on a plain row list (out row i <- host row rows[i])"""
    n = tl.load(n_ptr)
    c = tl.arange(0, 64)
    c2 = 64 + tl.arange(0, 8)
    for s in range(tl.program_id(0) * B, n, tl.num_programs(0) * B):
        i = s + tl.arange(0, B)
        m = i < n
        h = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
        src = host_ptr + h[:, None] * 72
        a = tl.load(src + c[None, :], mask=m[:, None])
        b = tl.load(src + c2[None, :], mask=m[:, None])
        dst = out_ptr + i.to(tl.int64)[:, None] * 72
        tl.store(dst + c[None, :], a, mask=m[:, None])
        tl.store(dst + c2[None, :], b, mask=m[:, None])


_AR = {}


def _ar(n, dev):
    a = _AR.get(dev)
    if a is None or a.shape[0] < n:
        a = _AR[dev] = torch.arange(max(n, 1 << 16), device=dev)
    return a[:n]


def copy_grid(dev, n):
    return max(1, min(triton.cdiv(n, FILL_B), _nsm(dev) * FILL_CTAS))


def sa_prefill(lo, flat):
    """LayerOffload._sa_prefill with the EPOCH / SPLIT / TWO options (same resolved rows as the default)."""
    from . import sparse_offload as so
    n = flat.shape[0]
    dev = flat.device
    if so._MASYNC or OVERLAP:
        so.mjoin(dev)       # a pending side-stream copy still reads the shared miss list / destinations / flags
    if getattr(so, "_SHARE_HT", False):      # ARC_OFFLOAD_SHARE_HT: one table / miss list per device (shared epoch)
        lo.ml = so._shared_ml(n, dev)
    elif lo.ml is None or lo.ml.shape[0] < n:
        lo.ml = torch.empty(n, dtype=torch.int64, device=dev)
    S = lo.C // lo.W
    rb = so._res_bufs(n, dev) if so._RESOLVE else None
    p2 = int(TWO and rb is not None)
    mls = rb["mls"] if rb else lo.ml
    ep = _epoch(lo.ht) if EPOCH else lo.mcnt
    if not EPOCH:
        lo.ht.fill_(-1)
        lo.ht[HTM + 1:] = -7
    BZ = 4096
    _begin_kernel[(triton.cdiv(lo.C, BZ),)](lo.pin, lo.C, ep, lo.mcnt, rb["scnt"] if rb else lo.mcnt,
                                            EP=int(EPOCH), RES=int(rb is not None), BLOCK=BZ)
    ovl = OVERLAP and rb is not None and EPOCH and (SPLIT or p2) and _ovl_ok() and so.ab_on("ovl")
    ft = _ftok(dev) if ovl else mls
    _scan_kernel[(triton.cdiv(n, 256),)](flat, lo.tag, lo.pin, lo.ht, lo.ml, lo.mcnt, n, S, lo.R, HTM, mls, ep, ft,
                                         max(1, getattr(lo, "_topk", 1)), W=lo.W, PROBES=16, BLOCK=256,
                                         RES=int(rb is not None), EP=int(EPOCH), P2=p2, FT=int(ovl))
    htv = rb["htv"] if rb else lo.ht
    spill = rb["spill"] if rb else lo.cache
    scnt = rb["scnt"] if rb else lo.mcnt
    if SPLIT or p2:
        cpd = _CPD.get(dev)
        if cpd is None or cpd.shape[0] < n:
            cpd = _CPD[dev] = torch.empty(max(n, cpd.shape[0] if cpd is not None else 0), dtype=torch.int64,
                                          device=dev)
        if ovl:
            lf, sf = _flags(lo, dev)
        else:
            lf = sf = cpd
        _alloc_kernel[(triton.cdiv(n, 256),)](lo.ml, lo.mcnt, lo.pin, lo.rr, lo.cnt, lo.tag, cpd, S, mls, htv, scnt,
                                              so.spill_cap(), lf, sf, W=lo.W, BLOCK=256, RES=int(rb is not None), P2=p2,
                                              FLAG=int(ovl))
        perm, ns = cpd, 0
        if ovl:   # copy order (side stream, overlapping resolve): attention wave of the row's first token, then host
            # address; the copy itself starts after resolve (below), concurrently with the attention
            st = so._side(dev)
            ev0 = torch.cuda.Event()
            ev0.record()
            with torch.cuda.stream(st):
                st.wait_event(ev0)
                ons = min(n, OVL_SORT)
                key = _OKEY.get((dev, ons))
                if key is None:
                    key = _OKEY[(dev, ons)] = torch.empty(ons, dtype=torch.int64 if lo.R >= (1 << 27) else torch.int32,
                                                          device=dev)
                _okey_kernel[(triton.cdiv(ons, 1024),)](lo.ml, mls, ft, lo.mcnt, key, ons, _nsm(dev),
                                                        K32=int(key.dtype == torch.int32), BLOCK=1024)
                operm = torch.sort(key).indices
        elif SORT:   # visit the first SORT misses in host-address order (GPU TLB / page-walk locality); fixed shapes
            ns = min(n, SORT)
            ar = _ar(ns, dev)
            if SORT32:   # 32-bit keys (row ids < 2^31): half the radix passes, same visit order
                key = torch.where(ar < lo.mcnt, lo.ml[:ns].to(torch.int32), torch.iinfo(torch.int32).max)
            else:
                key = torch.where(ar < lo.mcnt, lo.ml[:ns], torch.iinfo(torch.int64).max)
            perm = torch.sort(key).indices
        if not ovl and so._rf.ON:   # compact host rows: dequantizing copy (same order / destinations)
            so._rf.copy(lo.ml, lo.mcnt, cpd, lo.host, lo.cache, spill, perm, ns, copy_grid(dev, n), B=FILL_B,
                        num_warps=FILL_WARPS)
        elif not ovl:
            _copy_kernel[(copy_grid(dev, n),)](lo.ml, lo.mcnt, cpd, lo.host.view(torch.int64),
                                               lo.cache.view(torch.int64), spill.view(torch.int64), perm, ns, B=FILL_B,
                                               PERM=int(ns > 0), num_warps=FILL_WARPS)
        if so._MIRROR and getattr(lo, "_mon", None) and lo._srck is None:   # index-shared layers replay this fill
            lo._mrec = dict(gen=lo._gen + 1, n=n, mkey=lo._mkey, ml=lo.ml, mcnt=lo.mcnt, cpd=cpd,
                            perm=operm if ovl else perm, ns=ons if ovl else ns, rb=rb, ep=ep, p2=p2)
            if so._MASYNC and not ovl:
                lo._mirror_launch(lo._mrec)
    else:
        so._sa_fill_kernel[(triton.cdiv(n, 64),)](lo.ml, lo.mcnt, lo.pin, lo.rr, lo.cnt, lo.tag, lo.host, lo.cache, S,
                                                   mls, htv, spill, scnt, so.spill_cap(), W=lo.W, DN=so.D_NOPE,
                                                   DR=so.D_ROPE, BLOCK=64, RES=int(rb is not None))
    so._sa_advance_kernel[(triton.cdiv(S, 1024),)](lo.rr, lo.cnt, S, W=lo.W, BLOCK=1024)
    if hasattr(lo, "_gen"):
        lo._gen += 1
    if rb is not None:
        fl = lo._lflag if ovl else rb["res"]
        _resolve_kernel[(triton.cdiv(n, 256),)](flat, lo.tag, lo.ht, rb["htv"], rb["res"], n, S, lo.R, HTM, ep, fl,
                                                W=lo.W, PROBES=16, BLOCK=256, EP=int(EPOCH), P2=p2, FLAG=int(ovl))
        if ovl:   # copy on the side stream, concurrently with the attention (which waits on the flags)
            from . import offload_cuda
            st = so._side(dev)
            ev_res = torch.cuda.Event()
            ev_res.record()          # after resolve: a concurrent copy slows the latency-bound resolve badly
            with torch.cuda.stream(st):
                st.wait_event(ev_res)
                offload_cuda.fill_copy(lo.ml, lo.mcnt, cpd, lo.host, lo.cache, spill, lf, sf,
                                       so.side_grid(dev, OVL_GRID or _nsm(dev), OVL_RPW), st, perm=operm, ns=ons,
                                       hfmt=so._rf.FMTX)
                ev1 = torch.cuda.Event()
                ev1.record(st)
            so._MPEND.setdefault(dev, []).append(ev1)
            if so._MASYNC and getattr(lo, "_mrec", None) is not None and lo._mrec["gen"] == lo._gen:
                lo._mirror_launch(lo._mrec)     # the S layers' copy follows on the side stream
            return dict(rb, lflag=lo._lflag, sflag=sf, ovl=True)
    return rb
