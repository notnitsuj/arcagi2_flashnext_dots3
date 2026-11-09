"""Expert-stationary low-bit MoE GEMM for the pair layout (ARC_LOWBIT2=1, default off; see lowbit_moe.py).

The pair kernel (_lowbit_moe_gemm_tp) runs one CTA per (16-token block, N tile): an expert with 40 routed tokens is
3 blocks, so its weight tile is loaded (L2), dequantized and fed to the MMA three times. At decode batch sizes
(M = 256-384 tokens, top-8 of 144 experts, skewed routing) that is ~1.5x the dequant work of one pass per expert.

Here a work item is a run of up to CMAX consecutive 16-token blocks of one expert (built on the GPU from
moe_align_block_size's expert_ids, so it is CUDA-graph safe): one CTA per (item, N tile) loads and dequantizes each
weight group once and multiplies it with all 16*nb tokens (tl.dot N dimension 16 / 32 / 64). Items are ordered by
size (largest first) so the long ones start early. Per output element the arithmetic is the same instruction
sequence as the pair kernel (same k16 MMAs chained over j, same per-group scale step), so results are bit-identical.

EVICT=1 marks the streamed weight words / scales / zeros evict_first so the activations (x, c1, c2, c3) stay in L2.
"""
import torch
import triton
import triton.language as tl

from .lowbit_moe import _DQ3


@triton.jit
def _items_kernel(eid_ptr, npp_ptr, items_ptr, count_ptr, CMAX: tl.constexpr, BLOCK: tl.constexpr):
    """expert_ids per 16-token block -> items (b << 3 | nb), largest nb first; count[0] = number of items."""
    b = tl.arange(0, BLOCK)
    nblk = tl.load(npp_ptr) // 16
    valid = b < nblk
    e = tl.load(eid_ptr + b, mask=valid, other=-1)
    ep = tl.load(eid_ptr + b - 1, mask=valid & (b > 0), other=-2)
    en = tl.load(eid_ptr + b + 1, mask=(b + 1) < nblk, other=-3)
    start = valid & (e != ep)
    end = valid & (e != en)
    s = tl.associative_scan(tl.where(start, b, 0), 0, _max_op)
    big = BLOCK * 4
    t = tl.where(end, b, big)
    endp = tl.associative_scan(t, 0, _min_op, reverse=True)
    idx = b - s
    lead = valid & (e >= 0) & ((idx % CMAX) == 0)
    nb = tl.minimum(endp - b + 1, CMAX)
    base = 0
    for c in tl.static_range(CMAX):
        cls = CMAX - c
        m = lead & (nb == cls)
        mi = m.to(tl.int32)
        pos = base + tl.cumsum(mi, axis=0) - 1
        tl.store(items_ptr + pos, b * 8 + cls, mask=m)
        base += tl.sum(mi, axis=0)
    tl.store(count_ptr, base)


@triton.jit
def _min_op(a, b):
    return tl.minimum(a, b)


@triton.jit
def _max_op(a, b):
    return tl.maximum(a, b)


@triton.jit
def _body(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, topk_w_ptr, sorted_ids_ptr,
          e, row0, nb, pid_n, bits_e, off2_e, off1_e, offs_e,
          N, K, num_valid, stride_xm, stride_om, R2, R1, RS,
          TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, COMBINE: tl.constexpr, EVICT: tl.constexpr, PROBE: tl.constexpr):
    EV: tl.constexpr = "evict_first" if EVICT > 0 else ""
    EVS: tl.constexpr = "evict_last" if EVICT > 1 else ""
    offs_m = row0 + tl.arange(0, BM)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    mmask = (tok < num_valid) & (offs_m < row0 + nb * 16)
    tok = tl.where(mmask, tok, 0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    rs = offs_e + offs_n
    xrow = tok // TOPK if X_PER_TOKEN else tok
    KG = K // 128
    x_base = x_ptr + xrow[None, :] * stride_xm
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    w16 = tl.arange(0, 16)
    zero = tl.zeros((BN, 16), dtype=tl.int32)
    w8i = tl.arange(0, 8)
    w4i = tl.arange(0, 4)
    r2 = off2_e + offs_n
    r1 = off1_e + offs_n
    for g in range(0, KG):
        w8 = tl.load(p2_ptr + (g * R2 + r2[:, None]) * 8 + w8i[None, :], mask=nmask[:, None], other=0,
                     eviction_policy=EV)
        wd = tl.reshape(tl.join(w8, w8), (BN, 16))
        if bits_e == 3:
            h4 = tl.load(p1_ptr + (g * R1 + r1[:, None]) * 4 + w4i[None, :], mask=nmask[:, None], other=0,
                         eviction_policy=EV)
            h8 = tl.reshape(tl.join(h4, h4), (BN, 8))
            hb = tl.reshape(tl.join(h8, h8), (BN, 16))
        else:
            hb = zero
        accg = tl.zeros((BN, BM), dtype=tl.float32)
        xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
        if PROBE == 1:   # memory probe: same loads, no dequant / MMA
            accg += tl.sum((wd ^ hb).to(tl.float32), axis=1)[:, None]
        else:
            for j in tl.static_range(8):
                at = tl.load(x_base + (g * 128 + j * 16 + w16)[:, None], mask=mmask[None, :], other=0.0)
                sh = zero + 2 * j
                rot = zero + ((j + 30 + 8 * ((w16 // 2) % 2)) % 32)[None, :]
                q = tl.inline_asm_elementwise(_DQ3, "=r,r,r,r,r,r,r,r,r", [wd, hb, sh, rot], dtype=tl.float16,
                                              is_pure=True, pack=2)
                accg += tl.dot(q, at)
        sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0, eviction_policy=EV).to(tl.float32)
        zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0, eviction_policy=EV).to(tl.float32)
        acc += (accg - xs[None, :] * (zz[:, None] + 1024.0)) * sc[:, None]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[None, :]
    if COMBINE > 0:
        tl.atomic_add(out_ptr + (tok // COMBINE)[None, :] * stride_om + offs_n[:, None], acc,
                      mask=mmask[None, :] & nmask[:, None], sem="relaxed")
    else:
        tl.store(out_ptr + tok[None, :] * stride_om + offs_n[:, None], acc.to(tl.bfloat16),
                 mask=mmask[None, :] & nmask[:, None], eviction_policy=EVS)


@triton.jit
def _lowbit2_gemm(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr,
                  bits_ptr, off2_ptr, off1_ptr, offs_ptr,
                  topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, items_ptr, count_ptr,
                  N, K, num_valid, stride_xm, stride_om, R2, R1, RS, NT,
                  TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr,
                  BN: tl.constexpr, CMAX: tl.constexpr, COMBINE: tl.constexpr = 0, EVICT: tl.constexpr = 0,
                  PROBE: tl.constexpr = 0, BMX: tl.constexpr = 0):
    pid = tl.program_id(0)
    it_i = pid // NT
    pid_n = pid % NT
    if it_i >= tl.load(count_ptr):
        return
    it = tl.load(items_ptr + it_i)
    b = it >> 3
    nb = it & 7
    e = tl.load(expert_ids_ptr + b)
    bits_e = tl.load(bits_ptr + e)
    off2_e = tl.load(off2_ptr + e).to(tl.int64)
    off1_e = tl.load(off1_ptr + e).to(tl.int64)
    offs_e = tl.load(offs_ptr + e).to(tl.int64)
    row0 = b * 16
    if BMX > 0:   # single tile width (no branch on nb): items of up to BMX // 16 blocks, masked
        _body(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, topk_w_ptr, sorted_ids_ptr,
              e, row0, nb, pid_n, bits_e, off2_e, off1_e, offs_e, N, K, num_valid, stride_xm, stride_om, R2, R1, RS,
              TOPK, X_PER_TOKEN, MUL_W, BMX, BN, COMBINE, EVICT, PROBE)
    else:
        if nb == 1:
            _body(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, topk_w_ptr, sorted_ids_ptr,
                  e, row0, nb, pid_n, bits_e, off2_e, off1_e, offs_e, N, K, num_valid, stride_xm, stride_om, R2, R1, RS,
                  TOPK, X_PER_TOKEN, MUL_W, 16, BN, COMBINE, EVICT, PROBE)
        elif nb == 2:
            _body(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, topk_w_ptr, sorted_ids_ptr,
                  e, row0, nb, pid_n, bits_e, off2_e, off1_e, offs_e, N, K, num_valid, stride_xm, stride_om, R2, R1, RS,
                  TOPK, X_PER_TOKEN, MUL_W, 32, BN, COMBINE, EVICT, PROBE)
        else:
            if CMAX > 2:
                _body(x_ptr, xs_ptr, out_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, topk_w_ptr, sorted_ids_ptr,
                      e, row0, nb, pid_n, bits_e, off2_e, off1_e, offs_e, N, K, num_valid, stride_xm, stride_om, R2, R1,
                      RS, TOPK, X_PER_TOKEN, MUL_W, 64, BN, COMBINE, EVICT, PROBE)



_DISCARD = tl.constexpr("discard.global.L2 [$1], 128; mov.b32 $0, 0;")


@triton.jit
def _combine_kernel(c3_ptr, out_ptr, N, TOPK: tl.constexpr, BLOCK: tl.constexpr, DISCARD: tl.constexpr):
    """out[t] = bf16(sum_k c3[t*TOPK + k]) with torch's order for c3.view(M, 8, N).sum(1) (bf16 in, fp32 acc):
    (((v0 + v4) + (v1 + v5)) + (v2 + v6)) + (v3 + v7)  -> bit-identical. DISCARD=1: afterwards the c3 lines are
    dropped from L2 without write-back (discard.global.L2): c3 is dead, so its 2*M*8*N bytes never reach DRAM if the
    down GEMM's stores stayed in L2."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = cols < N
    base = c3_ptr + t * TOPK * N + cols
    if TOPK == 8:
        v0 = tl.load(base + 0 * N, mask=cm, other=0.0).to(tl.float32)
        v1 = tl.load(base + 1 * N, mask=cm, other=0.0).to(tl.float32)
        v2 = tl.load(base + 2 * N, mask=cm, other=0.0).to(tl.float32)
        v3 = tl.load(base + 3 * N, mask=cm, other=0.0).to(tl.float32)
        v4 = tl.load(base + 4 * N, mask=cm, other=0.0).to(tl.float32)
        v5 = tl.load(base + 5 * N, mask=cm, other=0.0).to(tl.float32)
        v6 = tl.load(base + 6 * N, mask=cm, other=0.0).to(tl.float32)
        v7 = tl.load(base + 7 * N, mask=cm, other=0.0).to(tl.float32)
        acc = (((v0 + v4) + (v1 + v5)) + (v2 + v6)) + (v3 + v7)
        tl.store(out_ptr + t * N + cols, acc.to(tl.bfloat16), mask=cm)
        if DISCARD:
            tl.debug_barrier()   # every thread of the CTA has its c3 values before any line is dropped
            ln = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK // 64) * 64     # one 128-byte line per 64 bf16
            for k in tl.static_range(8):
                lp = c3_ptr + t * TOPK * N + k * N + ln
                tl.inline_asm_elementwise(_DISCARD, "=r,l", [lp], dtype=tl.int32, is_pure=False, pack=1)


def combine(c3, M, topk, discard=0, block=1024):
    """c3 bf16 [M*topk, N] -> bf16 [M, N] (== c3.view(M, topk, N).sum(1) bitwise; topk 8 only)."""
    N = c3.shape[1]
    assert topk == 8 and c3.is_contiguous() and N % block == 0 and block % 64 == 0
    out = torch.empty(M, N, dtype=torch.bfloat16, device=c3.device)
    _combine_kernel[(M, triton.cdiv(N, block))](c3, out, N, TOPK=topk, BLOCK=block, DISCARD=discard,
                                                 num_warps=4)
    return out


def make_items(expert_ids, npp, cmax):
    """Work items for _lowbit2_gemm from moe_align_block_size(block_size=16) outputs (GPU only, graph safe)."""
    nbmax = expert_ids.shape[0]
    items = torch.empty(nbmax, dtype=torch.int32, device=expert_ids.device)
    count = torch.empty(1, dtype=torch.int32, device=expert_ids.device)
    _items_kernel[(1,)](expert_ids, npp, items, count, CMAX=cmax, BLOCK=triton.next_power_of_2(max(nbmax, 16)),
                        num_warps=4)
    return items, count


def gemm2(P, x, out, topk_w, sorted_ids, expert_ids, items, count, topk, x_per_token, mul_w, cfg, xs=None,
          combine=0, evict=0, probe=0):
    """Same contract as LowbitProj.__call__ (pair layout, block 16 alignment), expert-stationary items.
    cfg = (block_n, cmax, num_warps, num_stages)."""
    bn, cmax, nw, ns = cfg
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if xs is None:
        xs = x.view(x.shape[0], -1, 128).float().sum(-1)
    nt = triton.cdiv(P.N, bn)
    grid = (items.shape[0] * nt,)
    _lowbit2_gemm[grid](x, xs, out, P.p2, P.p1, P.sc, P.zp, P.bits, P.off2, P.off1, P.offs,
                        topk_w, sorted_ids, expert_ids, items, count,
                        P.N, P.K, topk_w.numel(), x.stride(0), out.stride(0), P.R2, P.R1, P.RS, nt,
                        TOPK=topk, X_PER_TOKEN=x_per_token, MUL_W=mul_w, BN=bn, CMAX=cmax, COMBINE=combine,
                        EVICT=evict, PROBE=probe, num_warps=nw, num_stages=ns)



_CFG = None


def layer_cfg(M):
    """Per-layer pipeline for M tokens from ARC_LOWBIT2_TUNE (json "moe2": {M: cfg | null}, written by
    bench_lowbit2.py on the target GPU; the smallest tuned M >= M decides; None = keep LowbitMoE's path).
    cfg = {"gu": ["old", bm, bn, warps, stages, evict] | ["new", bn, cmax, warps, stages, evict], "dn": same,
           "comb": 0 torch sum | 1 fused combine kernel | 2 fused combine + L2 discard of c3}."""
    global _CFG
    if _CFG is None:
        import json, os
        f = os.environ.get("ARC_LOWBIT2_TUNE")
        d = json.load(open(f)) if f and os.path.exists(f) else {}
        _CFG = sorted((int(m), c) for m, c in d.get("moe2", {}).items())
    for m, c in _CFG:
        if m >= M:
            return c
    return None


def _proj(P, cfg, x, out, topk_w, al16, items, topk, x_per_token, mul_w, xs, combine=0):
    if cfg[0] == "old":
        bm, bn, nw, ns, ev = cfg[1:]
        sid, eid, npp = al16[:3] if bm == 16 else al16[3](bm)
        P(x, out, topk_w, sid, eid, npp, topk, x_per_token, mul_w, bm, bn, nw, ns, xs=xs, combine=combine, evict=ev)
    else:
        bn, cm, nw, ns, ev = cfg[1:]
        gemm2(P, x, out, topk_w, al16[0], al16[1], *items(cm), topk, x_per_token, mul_w, (bn, cm, nw, ns), xs=xs,
              combine=combine, evict=ev)


def moe_forward(moe, x, topk_w, topk_ids, cfg, combine=False):
    """LowbitMoE.__call__ with the per-M pipeline cfg (see layer_cfg); bit-identical to it unless combine=True
    (ARC_LOWBIT_COMBINE fp32 atomics, then as LowbitMoE with combine)."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M = x.shape[0]
    topk = topk_ids.shape[1]
    sid, eid, npp = moe_align_block_size(topk_ids, 16, moe.E)
    other = {}
    al16 = (sid, eid, npp, lambda bm: other.setdefault(bm, moe_align_block_size(topk_ids, bm, moe.E)))
    its = {}
    items = lambda cm: its.setdefault(cm, make_items(eid, npp, cm))
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    xs = x.view(M, -1, 128).float().sum(-1)
    c1 = torch.empty(M * topk, moe.gu.N, dtype=torch.bfloat16, device=x.device)
    _proj(moe.gu, cfg["gu"], x, c1, topk_w.reshape(-1), al16, items, topk, True, False, xs)
    c2 = torch.empty(M * topk, moe.gu.N // 2, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(c2, c1)
    c2h = c2.to(torch.float16)
    xs2 = c2h.view(M * topk, -1, 128).float().sum(-1)
    if combine:
        c3 = torch.zeros(M, moe.dn.N, dtype=torch.float32, device=x.device)
        _proj(moe.dn, cfg["dn"], c2h, c3, topk_w.reshape(-1).float(), al16, items, 1, False, True, xs2, combine=topk)
        return c3.to(torch.bfloat16)
    c3 = torch.empty(M * topk, moe.dn.N, dtype=torch.bfloat16, device=x.device)
    _proj(moe.dn, cfg["dn"], c2h, c3, topk_w.reshape(-1).float(), al16, items, 1, False, True, xs2)
    if cfg.get("comb", 0) and topk == 8:
        return combine_(c3, M, topk, discard=int(cfg["comb"] == 2))
    return c3.view(M, topk, moe.dn.N).sum(1)


combine_ = combine
