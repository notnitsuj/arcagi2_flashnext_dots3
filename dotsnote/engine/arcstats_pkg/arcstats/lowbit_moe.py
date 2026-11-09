"""Mixed-precision (2/3/4-bit, per expert) MoE for Dots3 routed experts: one Triton GEMM launch per projection for all
experts of a layer, whatever their bit width.

Planar group layout (G = 128 input columns = one quantization group). For a projection W [E][N_out, K] (y = x @ W^T),
each row stores per group:
  bits 2/3:  p2  int16 x16 words: field j (2 bits, j<8) of word w = column j*16 + w
  bits 3:    p1  uint8 x16 words: bit j of byte w           = 3rd bit of column j*16 + w
  bits 4:    p4  int16 x32 words: field j (4 bits, j<4) of word w = column j*32 + w
codes are turned into fp16 without int->float conversions: (q | 0x6400) bitcast to fp16 is exactly 1024 + q (the
offset is folded into the zero point). Field j of all words is a contiguous column slice: every field is one tensor-core dot against a contiguous slice
of x, with no in-register reshape. Dequantization is factored out of the inner loop:
  sum_k x_k (q_k - z) s = s * (sum_k x_k q_k - z * sum_k x_k)        (per row, per group)
scale fp16 [rows, K/G], zero uint8 [rows, K/G]. Planes are stored group-major ([KG, rows, words]; scales/zeros [KG, rows]) so a tile load is one contiguous block.
Experts are concatenated along the row axis of each plane;
off4/off2/off1/offs [E] give each expert's first row. bits [E] int32 selects the path.
Token routing uses vLLM's moe_align_block_size (tokens sorted by expert, padded to BLOCK_M).
"""
import torch
import triton
import triton.language as tl

G = 128
import os  # noqa: E402
SWAP = os.environ.get("ARC_LOWBIT_SWAP") == "1"   # transposed kernel (planar layout)
PAIRS = os.environ.get("ARC_LOWBIT_PAIRS") == "1"  # pair layout + PTX lop3 dequant, transposed kernel (2/3-bit)


_PF = int(os.environ.get("ARC_LOWBIT_PF", "0"))   # pair kernel: prefetch the next K group (see _lowbit_moe_gemm_tp)

@triton.jit
def _lowbit_moe_gemm(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr,
                     bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr,
                     topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr,
                     N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS,
                     TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr = 0):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(num_post_pad_ptr):
        return
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    mmask = tok < num_valid
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    bits = tl.load(bits_ptr + e)
    rs = tl.load(offs_ptr + e).to(tl.int64) + offs_n
    xrow = tok // TOPK if X_PER_TOKEN else tok
    KG = K // 128
    x_base = x_ptr + xrow[:, None] * stride_xm
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w16 = tl.arange(0, 16)
    w32 = tl.arange(0, 32)
    if bits == 4:
        r4 = tl.load(off4_ptr + e).to(tl.int64) + offs_n
        for g in range(0, KG):
            wd = tl.load(p4_ptr + (g * R4 + r4[:, None]) * 32 + w32[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            accg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
            for j in tl.static_range(4):
                a = tl.load(x_base + (g * 128 + j * 32 + w32)[None, :], mask=mmask[:, None], other=0.0)
                q = (((wd >> (4 * j)) & 0xF) | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)   # 1024 + q
                accg += tl.dot(a, tl.trans(q))
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[:, None] * (zz[None, :] + 1024.0)) * sc[None, :]
    else:
        r2 = tl.load(off2_ptr + e).to(tl.int64) + offs_n
        r1 = tl.load(off1_ptr + e).to(tl.int64) + offs_n
        for g in range(0, KG):
            wd = tl.load(p2_ptr + (g * R2 + r2[:, None]) * 16 + w16[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            if bits == 3:
                hb = tl.load(p1_ptr + (g * R1 + r1[:, None]) * 16 + w16[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            else:
                hb = tl.zeros((BLOCK_N, 16), dtype=tl.int32)
            accg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
            for j in tl.static_range(8):
                a = tl.load(x_base + (g * 128 + j * 16 + w16)[None, :], mask=mmask[:, None], other=0.0)
                q = (((wd >> (2 * j)) & 0x3) | (((hb >> j) & 0x1) << 2) | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)
                accg += tl.dot(a, tl.trans(q))
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[:, None] * (zz[None, :] + 1024.0)) * sc[None, :]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[:, None]
    if COMBINE > 0:   # fused top-k sum: fp32 atomics into out[token] (out zeroed fp32 [M, N])
        tl.atomic_add(out_ptr + (tok // COMBINE)[:, None] * stride_om + offs_n[None, :], acc,
                      mask=mmask[:, None] & nmask[None, :], sem="relaxed")
    else:
        tl.store(out_ptr + tok[:, None] * stride_om + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _lowbit_moe_gemm_t(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr,
                       bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr,
                       topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr,
                       N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS,
                       TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr = 0):
    """Same contract as _lowbit_moe_gemm, computed transposed (out^T = W x^T): the dequantized weight tile [BLOCK_N, 16]
    is the MMA A operand in its natural K-contiguous layout and tokens are the MMA N dimension."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(num_post_pad_ptr):
        return
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    mmask = tok < num_valid
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    bits = tl.load(bits_ptr + e)
    rs = tl.load(offs_ptr + e).to(tl.int64) + offs_n
    xrow = tok // TOPK if X_PER_TOKEN else tok
    KG = K // 128
    x_base = x_ptr + xrow[None, :] * stride_xm          # [1, BM]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    w16 = tl.arange(0, 16)
    w32 = tl.arange(0, 32)
    if bits == 4:
        r4 = tl.load(off4_ptr + e).to(tl.int64) + offs_n
        for g in range(0, KG):
            wd = tl.load(p4_ptr + (g * R4 + r4[:, None]) * 32 + w32[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            accg = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
            for j in tl.static_range(4):
                at = tl.load(x_base + (g * 128 + j * 32 + w32)[:, None], mask=mmask[None, :], other=0.0)   # [32, BM]
                q = (((wd >> (4 * j)) & 0xF) | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)
                accg += tl.dot(q, at)
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[None, :] * (zz[:, None] + 1024.0)) * sc[:, None]
    else:
        r2 = tl.load(off2_ptr + e).to(tl.int64) + offs_n
        r1 = tl.load(off1_ptr + e).to(tl.int64) + offs_n
        for g in range(0, KG):
            wd = tl.load(p2_ptr + (g * R2 + r2[:, None]) * 16 + w16[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            if bits == 3:
                hb = tl.load(p1_ptr + (g * R1 + r1[:, None]) * 16 + w16[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            else:
                hb = tl.zeros((BLOCK_N, 16), dtype=tl.int32)
            accg = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
            for j in tl.static_range(8):
                at = tl.load(x_base + (g * 128 + j * 16 + w16)[:, None], mask=mmask[None, :], other=0.0)   # [16, BM]
                q = (((wd >> (2 * j)) & 0x3) | (((hb >> j) & 0x1) << 2) | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)
                accg += tl.dot(q, at)
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[None, :] * (zz[:, None] + 1024.0)) * sc[:, None]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[None, :]
    if COMBINE > 0:
        tl.atomic_add(out_ptr + (tok // COMBINE)[None, :] * stride_om + offs_n[:, None], acc,
                      mask=mmask[None, :] & nmask[:, None], sem="relaxed")
    else:
        tl.store(out_ptr + tok[None, :] * stride_om + offs_n[:, None], acc.to(tl.bfloat16),
                 mask=mmask[None, :] & nmask[:, None])


# Pair layout (2/3-bit; same bytes as planar), per row and group of 128 columns, column c = j*16 + 2p + lane:
#   p2: int32 x8, word p: bits [2j, 2j+2) = lane-0 code bits 0-1, bits [16+2j, 16+2j+2) = lane-1 code bits 0-1
#   p1: int32 x4, word p//2: bit 8*(p%2) + j = lane-0 bit 2, bit 16 + 8*(p%2) + j = lane-1 bit 2
# so one shift + one rotate + two lop3 turn a word pair into the two fp16 values (1024 + q) of adjacent columns.
_DQ3 = tl.constexpr("{ .reg .b32 a, b, u; shr.b32 a, $1, $5; shf.r.wrap.b32 b, $3, $3, $7; "
        "lop3.b32 u, a, b, 0x00030003, 0xE4; lop3.b32 $0, u, 0x00070007, 0x64006400, 0xEA; }")


@triton.jit
def _lowbit_moe_gemm_tp(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr,
                        bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr,
                        topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr,
                        N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS,
                        TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr = 0,
                        PF: tl.constexpr = 0, EVICT: tl.constexpr = 0):
    """Transposed GEMM (see _lowbit_moe_gemm_t) on the pair layout: 2/3-bit experts only (4-bit: planar kernels).
    PF=1: the next K group's packed words are loaded (into registers) before the current group is dequantized, so the
    weight stream has one group in flight while the dequant + MMA of the previous one runs.
    EVICT>=1: weight words / scales / zeros are loaded evict_first (activations stay in L2); EVICT=2: the output is
    also stored evict_last (lowbit2.py: the c3 -> fused combine round trip stays in L2). Same arithmetic."""
    EV: tl.constexpr = "evict_first" if EVICT > 0 else ""
    EVS: tl.constexpr = "evict_last" if EVICT > 1 else ""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(num_post_pad_ptr):
        return
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m).to(tl.int64)
    mmask = tok < num_valid
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    bits = tl.load(bits_ptr + e)
    rs = tl.load(offs_ptr + e).to(tl.int64) + offs_n
    xrow = tok // TOPK if X_PER_TOKEN else tok
    KG = K // 128
    x_base = x_ptr + xrow[None, :] * stride_xm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    w16 = tl.arange(0, 16)
    zero = tl.zeros((BLOCK_N, 16), dtype=tl.int32)
    w8i = tl.arange(0, 8)
    w4i = tl.arange(0, 4)
    r2 = tl.load(off2_ptr + e).to(tl.int64) + offs_n
    r1 = tl.load(off1_ptr + e).to(tl.int64) + offs_n
    hm = nmask[:, None] & (bits == 3)
    if PF:
        w8n = tl.load(p2_ptr + r2[:, None] * 8 + w8i[None, :], mask=nmask[:, None], other=0, eviction_policy=EV)
        h4n = tl.load(p1_ptr + r1[:, None] * 4 + w4i[None, :], mask=hm, other=0, eviction_policy=EV)
    for g in range(0, KG):
        if PF:
            w8 = w8n
            h4 = h4n
            more = g + 1 < KG
            w8n = tl.load(p2_ptr + ((g + 1) * R2 + r2[:, None]) * 8 + w8i[None, :], mask=nmask[:, None] & more, other=0, eviction_policy=EV)
            h4n = tl.load(p1_ptr + ((g + 1) * R1 + r1[:, None]) * 4 + w4i[None, :], mask=hm & more, other=0, eviction_policy=EV)
            wd = tl.reshape(tl.join(w8, w8), (BLOCK_N, 16))
            h8 = tl.reshape(tl.join(h4, h4), (BLOCK_N, 8))
            hb = tl.reshape(tl.join(h8, h8), (BLOCK_N, 16))                  # 2-bit experts: h4 = 0 -> hb = 0
        else:
            w8 = tl.load(p2_ptr + (g * R2 + r2[:, None]) * 8 + w8i[None, :], mask=nmask[:, None], other=0, eviction_policy=EV)
            wd = tl.reshape(tl.join(w8, w8), (BLOCK_N, 16))                     # column w <- word w // 2
            if bits == 3:
                h4 = tl.load(p1_ptr + (g * R1 + r1[:, None]) * 4 + w4i[None, :], mask=nmask[:, None], other=0, eviction_policy=EV)
                h8 = tl.reshape(tl.join(h4, h4), (BLOCK_N, 8))
                hb = tl.reshape(tl.join(h8, h8), (BLOCK_N, 16))                  # column w <- word w // 4
            else:
                hb = zero
        accg = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
        for j in tl.static_range(8):
            at = tl.load(x_base + (g * 128 + j * 16 + w16)[:, None], mask=mmask[None, :], other=0.0)   # [16, BM]
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


def _wrap32(w):
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def _pairs_from_codes(q):
    """q int [R, KG, 8 (j), 16 (w)] 3-bit or 2-bit codes -> (p2 int32 [R, KG, 8], p1 int32 [R, KG, 4]) pair layout."""
    q = q.to(torch.int64).view(*q.shape[:2], 8, 8, 2)                  # [R, KG, j, p, lane]
    sh = (torch.arange(8, device=q.device) * 2).view(1, 1, 8, 1)
    p2 = (((q[..., 0] & 3) << sh) | ((q[..., 1] & 3) << (sh + 16))).sum(2)            # [R, KG, p]
    jj = torch.arange(8, device=q.device).view(1, 1, 8, 1)
    hb = ((((q[..., 0] >> 2) & 1) << jj) | (((q[..., 1] >> 2) & 1) << (jj + 16))).sum(2)  # [R, KG, p]
    p1 = hb[..., 0::2] | (hb[..., 1::2] << 8)                                          # [R, KG, 4]
    return _wrap32(p2), _wrap32(p1)


def _codes_from_planar(p2, p1, KG):
    """planar p2 int16 [R, KG*16], p1 uint8 [R, KG*16] or None -> codes int32 [R, KG, 8, 16]."""
    R = p2.shape[0]
    w = p2.view(R, KG, 1, 16).to(torch.int32) & 0xFFFF
    j = torch.arange(8, device=p2.device).view(1, 1, 8, 1)
    q = (w >> (2 * j)) & 3
    if p1 is not None:
        h = p1.view(R, KG, 1, 16).to(torch.int32) & 0xFF
        q = q | (((h >> j) & 1) << 2)
    return q


def _pack_planar(q, bits):
    """q int [R, K] codes -> dict of planes in the planar group layout (see module docstring)."""
    R, K = q.shape
    q = q.to(torch.int32).view(R, K // 128, 128)
    out = {}
    if bits == 4:
        f = q.view(R, K // 128, 4, 32)                                  # [R, g, field j, word w]
        w = sum(((f[:, :, j, :] & 0xF) << (4 * j)) for j in range(4))
        out["p4"] = w.view(R, K // 4).to(torch.int32).to(torch.int16) if False else _to_i16(w).view(R, K // 4)
    else:
        f = q.view(R, K // 128, 8, 16)
        w = sum(((f[:, :, j, :] & 0x3) << (2 * j)) for j in range(8))
        out["p2"] = _to_i16(w).view(R, K // 8)
        if bits == 3:
            h = sum((((f[:, :, j, :] >> 2) & 0x1) << j) for j in range(8))
            out["p1"] = h.to(torch.uint8).view(R, K // 8)
    return out


def _to_i16(w):
    w = w.to(torch.int32)
    return torch.where(w >= 32768, w - 65536, w).to(torch.int16)


def _to_pairs(obj, bits, KG, off2, off1, n_out):
    """ARC_LOWBIT_PAIRS=1: convert the planar group-major planes [KG, rows, words] of a 2/3-bit-only projection to the
    pair layout (same rows and offsets, same bytes), one expert at a time."""
    obj.pairs = False
    if not PAIRS or any(b == 4 for b in bits) or obj.R2 <= 1:
        return
    dev = obj.p2.device
    lo = torch.empty(KG, obj.R2, 8, dtype=torch.int32, device=dev)
    hi = torch.zeros(KG, max(1, obj.R1), 4, dtype=torch.int32, device=dev)
    for e, b in enumerate(bits):
        if b not in (2, 3):
            continue
        r2 = slice(off2[e], off2[e] + n_out)
        p2 = obj.p2[:, r2].transpose(0, 1).reshape(n_out, KG * 16)
        p1 = obj.p1[:, off1[e]:off1[e] + n_out].transpose(0, 1).reshape(n_out, KG * 16) if b == 3 else None
        a, h = _pairs_from_codes(_codes_from_planar(p2, p1, KG))
        lo[:, r2] = a.transpose(0, 1)
        if b == 3:
            hi[:, off1[e]:off1[e] + n_out] = h.transpose(0, 1)
    obj.p2, obj.p1, obj.pairs = lo, hi, True


class LowbitProj:
    """One projection (gate_up or down) of all experts in a layer, on one device."""

    def __init__(self, bits, codes, scales, zeros, device):
        """bits: list[int] per expert (0 = absent); codes[e]: uint8 [N_out, K] (or None); scales/zeros [N_out, K/G]."""
        E = len(bits)
        n_out = next(c.shape[0] for c in codes if c is not None)
        self.K = next(c.shape[1] for c in codes if c is not None)
        self.N = n_out
        p4, p2, p1, sc, zp = [], [], [], [], []
        off4, off2, off1, offs = [0] * E, [0] * E, [0] * E, [0] * E
        n4 = n2 = n1 = ns = 0
        for e in range(E):
            if not bits[e]:
                continue
            pk = _pack_planar(codes[e], bits[e])
            if bits[e] == 4:
                p4.append(pk["p4"]); off4[e] = n4; n4 += n_out
            else:
                p2.append(pk["p2"]); off2[e] = n2; n2 += n_out
                if bits[e] == 3:
                    p1.append(pk["p1"]); off1[e] = n1; n1 += n_out
            sc.append(scales[e].half()); zp.append(zeros[e].to(torch.uint8)); offs[e] = ns; ns += n_out
        KG = self.K // G

        def gmajor(parts, words, dtype):   # [rows, KG*words] per expert -> [KG, rows_total, words]
            if not parts:
                return torch.zeros(1, 1, 1, dtype=dtype).to(device), 1
            t = torch.cat(parts).view(-1, KG, words).transpose(0, 1).contiguous()
            return t.to(device), t.shape[1]
        self.p4, self.R4 = gmajor(p4, 32, torch.int16)
        self.p2, self.R2 = gmajor(p2, 16, torch.int16)
        self.p1, self.R1 = gmajor(p1, 16, torch.uint8)
        _to_pairs(self, bits, KG, off2, off1, n_out)
        self.sc = torch.cat(sc).t().contiguous().to(device)      # [KG, rows_total]
        self.zp = torch.cat(zp).t().contiguous().to(device)
        self.RS = self.sc.shape[1]
        t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
        self.bits, self.off4, self.off2, self.off1, self.offs = t(bits), t(off4), t(off2), t(off1), t(offs)

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.p4, self.p2, self.p1, self.sc, self.zp))

    def __call__(self, x, out, topk_w, sorted_ids, expert_ids, num_post_pad, topk, x_per_token, mul_w,
                 block_m, block_n=64, num_warps=4, num_stages=2, xs=None, combine=0, evict=0):
        """x: fp16 [rows, K]; xs: fp32 [rows, K/G] per-group row sums of x (computed here if None)."""
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        if xs is None:
            xs = x.view(x.shape[0], -1, G).float().sum(-1)
        grid = (sorted_ids.shape[0] // block_m, triton.cdiv(self.N, block_n))
        kern = _lowbit_moe_gemm_tp if getattr(self, "pairs", False) else (_lowbit_moe_gemm_t if SWAP else _lowbit_moe_gemm)
        kern[grid](x, xs, out, self.p4, self.p2, self.p1, self.sc, self.zp,
                               self.bits, self.off4, self.off2, self.off1, self.offs,
                               topk_w, sorted_ids, expert_ids, num_post_pad,
                               self.N, self.K, topk_w.numel(), x.stride(0), out.stride(0),
                               self.R4, self.R2, self.R1, self.RS,
                               TOPK=topk, X_PER_TOKEN=x_per_token, MUL_W=mul_w,
                               BLOCK_M=block_m, BLOCK_N=block_n, COMBINE=combine,
                               num_warps=num_warps, num_stages=num_stages,
                               **({"PF": _PF, "EVICT": evict} if kern is _lowbit_moe_gemm_tp else {}))


class LowbitMoE:
    """Routed experts of one layer: gate_up (stacked, N_out = 2*I) and down (N_out = H)."""

    def __init__(self, gate_up: LowbitProj, down: LowbitProj, num_experts: int, block_m: int = 16):
        self.gu, self.dn, self.E, self.block_m = gate_up, down, num_experts, block_m

    def __call__(self, x, topk_w, topk_ids):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        M, H = x.shape
        topk = topk_ids.shape[1]
        if _LB2 and M <= _LB2_MAXM and getattr(self.gu, "pairs", False) and getattr(self.dn, "pairs", False):
            from . import lowbit2   # per-M tuned pipeline, bit-identical (ARC_LOWBIT2=1, run_l4 lowbit2_gate)
            cfg = lowbit2.layer_cfg(M)
            if cfg:
                return lowbit2.moe_forward(self, x, topk_w, topk_ids, cfg, combine=_COMBINE)
        cg, cd = _tile("gate_up", M, self.block_m), _tile("down", M, self.block_m)
        al = {}
        for bm in {cg[0], cd[0]}:
            al[bm] = moe_align_block_size(topk_ids, bm, self.E)
        c1 = torch.empty(M * topk, self.gu.N, dtype=torch.bfloat16, device=x.device)
        self.gu(x, c1, topk_w.reshape(-1), *al[cg[0]], topk, True, False, *cg)
        c2 = torch.empty(M * topk, self.gu.N // 2, dtype=torch.bfloat16, device=x.device)
        torch.ops._C.silu_and_mul(c2, c1)
        if _COMBINE:
            c3 = torch.zeros(M, self.dn.N, dtype=torch.float32, device=x.device)
            self.dn(c2, c3, topk_w.reshape(-1).float(), *al[cd[0]], 1, False, True, *cd, combine=topk)
            return c3.to(torch.bfloat16)
        c3 = torch.empty(M * topk, self.dn.N, dtype=torch.bfloat16, device=x.device)
        self.dn(c2, c3, topk_w.reshape(-1).float(), *al[cd[0]], 1, False, True, *cd)
        return c3.view(M, topk, self.dn.N).sum(1)


_TUNE = None
_COMBINE = __import__("os").environ.get("ARC_LOWBIT_COMBINE") == "1"   # fused top-k combine (fp32 atomics; not bit-exact)
_FUSE = __import__("os").environ.get("ARC_LOWBIT_FUSE") == "1"     # lowbit_fuse.py small-kernel fusion
_FUSE_MAXM = int(__import__("os").environ.get("ARC_LOWBIT_FUSE_MAXM", "512"))
_RDUMP = bool(__import__("os").environ.get("ARC_ROUTE_DUMP"))   # routing dump for tile tuning (diagnostics)
_LB2 = __import__("os").environ.get("ARC_LOWBIT2") == "1"           # lowbit2.py expert-stationary kernels
_LB2_MAXM = int(__import__("os").environ.get("ARC_LOWBIT2_MAXM", "512"))   # decode batches only (prefill: pair kernel)


def _tile(proj, M, block_m):
    """(block_m, block_n, num_warps, num_stages) for this projection at M tokens: from ARC_LOWBIT_TUNE (json
    {proj: {M: [bm, bn, warps, stages]}}, written by tune_lowbit.py on the target GPU; smallest tuned M >= M, else
    the largest), default (block_m, 64, 4, 2)."""
    global _TUNE
    if _TUNE is None:
        import json, os
        f = os.environ.get("ARC_LOWBIT_TUNE")
        _TUNE = {k: sorted((int(m), tuple(c)) for m, c in v.items()) for k, v in json.load(open(f)).items()} \
            if f and os.path.exists(f) else {}
    t = _TUNE.get(proj)
    if not t:
        return (block_m, 64, 4, 2)
    for m, c in t:
        if m >= M:
            return c
    return t[-1][1]


def proj_from_packed(experts, which, rows, gslice, device):
    """Build a LowbitProj from packed per-expert planes (output of gptq_experts.py --pack), taking `rows` (index tensor
    of output rows) and quantization groups `gslice` (slice over K/G) for this TP rank."""
    obj = LowbitProj.__new__(LowbitProj)
    p4, p2, p1, sc, zp = [], [], [], [], []
    E = len(experts)
    off4, off2, off1, offs, bits = [0] * E, [0] * E, [0] * E, [0] * E, [0] * E
    n4 = n2 = n1 = ns = 0
    for i, ex in enumerate(experts):
        b = ex["bits"]; bits[i] = b
        pl, s_, z_ = ex[which], ex[which + "_s"], ex[which + "_z"]
        # slice on the GPU: CPU fancy-indexing of the planes in several TP workers at once oversubscribes the host
        pl = {kk: vv.to(device, non_blocking=True) for kk, vv in pl.items()}
        s_, z_ = s_.to(device), z_.to(device)
        rows = rows.to(device)
        n_out = len(rows)
        KG = s_.shape[1]
        if b == 4:
            w = pl["p4"].view(pl["p4"].shape[0], KG, 32)[rows][:, gslice]
            p4.append(w.reshape(n_out, -1)); off4[i] = n4; n4 += n_out
        else:
            w = pl["p2"].view(pl["p2"].shape[0], KG, 16)[rows][:, gslice]
            p2.append(w.reshape(n_out, -1)); off2[i] = n2; n2 += n_out
            if b == 3:
                h = pl["p1"].view(pl["p1"].shape[0], KG, 16)[rows][:, gslice]
                p1.append(h.reshape(n_out, -1)); off1[i] = n1; n1 += n_out
        sc.append(s_[rows][:, gslice].half()); zp.append(z_[rows][:, gslice]); offs[i] = ns; ns += n_out
    kg = sc[0].shape[1]
    obj.N, obj.K = len(rows), kg * G

    def gmajor(parts, words, dtype):
        if not parts:
            return torch.zeros(1, 1, 1, dtype=dtype, device=device), 1
        t = torch.cat(parts).view(-1, kg, words).transpose(0, 1).contiguous()
        return t.to(device), t.shape[1]
    obj.p4, obj.R4 = gmajor(p4, 32, torch.int16)
    obj.p2, obj.R2 = gmajor(p2, 16, torch.int16)
    obj.p1, obj.R1 = gmajor(p1, 16, torch.uint8)
    _to_pairs(obj, bits, kg, off2, off1, len(rows))
    obj.sc = torch.cat(sc).t().contiguous().to(device)
    obj.zp = torch.cat(zp).t().contiguous().to(device)
    obj.RS = obj.sc.shape[1]
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    obj.bits, obj.off4, obj.off2, obj.off1, obj.offs = t(bits), t(off4), t(off2), t(off1), t(offs)
    return obj


class ArcLowbitExperts(torch.nn.Module):
    """Drop-in for vLLM's FusedMoE in Dots3 MoE layers of an `arc_lowbit` checkpoint: sigmoid routing with the
    e_score_correction_bias (noaux_tc, renormalized top-k), then the mixed-bit kernels on this TP rank's shard.
    Output is the TP-partial sum (the Dots3 MoE layer all-reduces after adding the shared expert)."""

    def __init__(self, *, gate, e_score_correction_bias, top_k, renormalize, routed_scaling_factor,
                 num_experts, intermediate_size, prefix, lowbit_dir, block_m=16, **_):
        super().__init__()
        import re
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        self.gate, self.top_k, self.renormalize = gate, top_k, renormalize
        self.scale = routed_scaling_factor
        self.bias = e_score_correction_bias
        self.num_experts, self.block_m = num_experts, block_m
        self.layer = int(re.search(r"layers\.(\d+)\.", prefix).group(1))
        self.lowbit_path = f"{lowbit_dir}/lowbit-L{self.layer:02d}.pt"
        self.tp_rank, self.tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        self.I = intermediate_size
        self.moe = None

    def _random_experts(self, device):
        """ARC_LOWBIT_RANDOM=1: random packed planes with the real shapes/bit mix (first ARC_LOWBIT_RANDOM_3B experts
        3-bit, rest 2-bit), generated on the GPU -- for speed measurements without expert weights."""
        import os
        n3 = int(os.environ.get("ARC_LOWBIT_RANDOM_3B", "96"))
        H, I = int(os.environ.get("ARC_LOWBIT_HIDDEN", "5120")), self.I
        out = []
        for e in range(self.num_experts):
            b = 3 if e < n3 else 2
            def mk(n, k):
                pl = {"p2": torch.randint(-32768, 32767, (n, k // 8), dtype=torch.int16, device=device)}
                if b == 3:
                    pl["p1"] = torch.randint(0, 255, (n, k // 8), dtype=torch.uint8, device=device)
                return pl, (torch.rand(n, k // G, device=device) * 0.002).half(), \
                    torch.randint(0, 2 ** b, (n, k // G), dtype=torch.uint8, device=device)
            gu, gus, guz = mk(2 * I, H)
            d, ds, dz = mk(H, I)
            out.append({"bits": b, "gu": gu, "gu_s": gus, "gu_z": guz, "d": d, "d_s": ds, "d_z": dz})
        return out

    def load(self, device):
        import os
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        if os.environ.get("ARC_LOWBIT_RANDOM") == "1":
            experts = self._random_experts(device)
        else:
            if os.environ.get("ARC_STAGE_DIR"):   # local checkpoint staging (run_l4 prefetch "stage", default off)
                from .stage_wait import wait_staged
                wait_staged(self.lowbit_path)
            d = torch.load(self.lowbit_path, map_location="cpu", weights_only=False)
            if os.environ.get("ARC_DELTA_FILE"):   # fine-tuned expert scales overlay (arcstats.delta_overlay)
                from .delta_overlay import apply_lowbit
                apply_lowbit(self.layer, d)
            experts = [d[e] for e in d["experts"]]
        assert len(experts) == self.num_experts, (len(experts), self.num_experts)
        I, r, tp = self.I, self.tp_rank, self.tp_size
        Is = I // tp
        gu_rows = torch.cat([torch.arange(r * Is, (r + 1) * Is), I + torch.arange(r * Is, (r + 1) * Is)])
        if experts[0]["gu_s"].is_cuda:
            gu_rows = gu_rows.to(experts[0]["gu_s"].device)
        H = experts[0]["d"]["p2" if "p2" in experts[0]["d"] else "p4"].shape[0]
        kg_d = experts[0]["d_s"].shape[1]
        gpr = kg_d // tp
        gu = proj_from_packed(experts, "gu", gu_rows, slice(None), device)
        dn = proj_from_packed(experts, "d", torch.arange(H, device=experts[0]["d_s"].device),
                              slice(r * gpr, (r + 1) * gpr), device)
        self.moe = LowbitMoE(gu, dn, self.num_experts, self.block_m)
        return gu.nbytes() + dn.nbytes()

    def _route_dump(self, ids):
        """ARC_ROUTE_DUMP=path,min_m,max_calls (diagnostics, default off; eager steps only, TP rank 0): top-k expert
        ids of every MoE call with M >= min_m, saved per layer to path_L<layer>.npz (ids int16 [sum M, topk], m [calls])
        -- real decode routing for tile tuning (dots3/tune_lowbit_routes.py)."""
        import os
        import numpy as np
        if self.tp_rank != 0 or torch.cuda.is_current_stream_capturing():
            return
        st = getattr(self, "_rd", None)
        if st is None:
            path, mm, mc = os.environ["ARC_ROUTE_DUMP"].split(",")
            st = self._rd = dict(path=f"{path}_L{self.layer:02d}.npz", mm=int(mm), left=int(mc), ids=[], m=[])
        if st["left"] <= 0 or ids.shape[0] < st["mm"]:
            return
        st["ids"].append(ids.to(torch.int16).cpu().numpy()); st["m"].append(ids.shape[0]); st["left"] -= 1
        if st["left"] == 0 or len(st["m"]) % 200 == 0:
            np.savez(st["path"], ids=np.concatenate(st["ids"]), m=np.array(st["m"], dtype=np.int32))

    def forward(self, hidden_states, router_logits=None):
        logits = self.gate(hidden_states)
        logits = logits[0] if isinstance(logits, tuple) else logits
        if _FUSE and hidden_states.shape[0] <= _FUSE_MAXM and getattr(self.moe.gu, "pairs", False):
            from . import lowbit_fuse   # fused routing / prep / silu kernels (ARC_LOWBIT_FUSE=1, not bit-exact)
            w, ids = lowbit_fuse.route(logits, self.bias, self.top_k, self.renormalize, self.scale)
            if _RDUMP:
                self._route_dump(ids)
            return lowbit_fuse.moe_forward(self.moe, hidden_states, w, ids)
        scores = torch.sigmoid(logits.float())
        sel = scores + self.bias.float() if self.bias is not None else scores
        ids = torch.topk(sel, self.top_k, dim=-1).indices
        w = scores.gather(1, ids)
        if self.renormalize:
            w = w / w.sum(-1, keepdim=True)
        w = w * self.scale
        if _RDUMP:
            self._route_dump(ids)
        return self.moe(hidden_states, w, ids.to(torch.int32))
