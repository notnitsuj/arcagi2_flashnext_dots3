"""Compact host rows for the DSA KV offload (ARC_OFFLOAD_ROW_FMT, default fp8 = today's 576 B rows, untouched).

Only the HOST copy (pinned / UVA backing store, the bytes that cross PCIe on every row-cache miss and prefill gather)
changes format. Every host reader dequantizes on the fly into today's GPU format (576 B fp8 e4m3 rows: 512 latent +
64 rope, kv_scale 1), so the row cache, spill rows, staged prefill rows and the attention MMA paths are unchanged.

  ARC_OFFLOAD_ROW_FMT = fp8 | e2m3 | e2m2 | e2m1  [+ b16 | b32 block suffix, default b32]
  latent: 512 values as sign + E2M{M} minifloat (bias 1, max 7.5 / 7 / 6) with a power-of-two scale 2^k per BLK values
          (MX-style; k int8). Dequantized values mant * 2^(k + e) have <= 4 significant bits, so they are EXACT e4m3
          values whenever they are in e4m3's range: the GPU row equals the host value, no double rounding (values below
          e4m3's subnormal step 2^-9 round as today's fp8 store would). E2M3 keeps e4m3's 3-bit mantissa for every value
          within 4 binades of its block max.
  rope:   the 64 rope values stay e4m3 bytes (bit-identical to today).
  row bytes (16 B aligned): [lo nibble plane 256 B | hi bit plane (M=3: 2 bits/value 128 B, M=2: 1 bit 64 B, M=1: -) |
          int8 block exponents 512/BLK B | rope 64 B]  -> e2m3b32 464 B (-19.4%), e2m3b16 480, e2m2b32 400 (-30.6%),
          e2m2b16 416, e2m1b32 336 (-41.7%), e2m1b16 352 (-38.9%).
  value e (0..511): code = lo nibble (byte e/2, low nibble for even e) | hi bits << 4 (M=3: byte 256 + e/4, bits
          2 (e%4); M=2: byte 256 + e/8, bit e%8); code bit M+2 = sign, magnitude index i = code & (2^(M+2)-1):
          e2 = i >> M, m = i & (2^M-1), value = (e2 ? 2^M + m : m) * 2^(max(e2,1) - 1 - M + k).
Host readers: store (quantize), offload_fill copy (Triton), csrc fill_copy / attention fetch-or-wait / direct host
rows (CUDA, fmt code FMT), mirror refill / write-through, prefill gather. Needs the W-way resolved cache, the split fill
and the CUDA split attention (the v12+ serving stack); sparse_offload refuses other paths.
"""
import os

import torch
import triton
import triton.language as tl

_SPEC = os.environ.get("ARC_OFFLOAD_ROW_FMT", "fp8").strip().lower() or "fp8"


def parse(spec):
    """-> (M, BLK) or None for fp8"""
    if spec in ("fp8", "e4m3", "0", "off"):
        return None
    s = spec
    blk = 32
    if "b" in s[1:]:
        s, b = s.rsplit("b", 1)
        blk = int(b)
    M = {"e2m3": 3, "e2m2": 2, "e2m1": 1}.get(s)
    if M is None or blk not in (16, 32):
        raise ValueError(f"ARC_OFFLOAD_ROW_FMT={spec!r}: expected fp8 | e2m3 | e2m2 | e2m1 [b16|b32]")
    return M, blk


def layout(M, BLK):
    """(hi plane bytes, scale offset, rope offset, row bytes)"""
    hb = {3: 128, 2: 64, 1: 0}[M]
    soff = 256 + hb
    roff = soff + 512 // BLK
    return hb, soff, roff, roff + 64


_P = parse(_SPEC)
ON = _P is not None
M, BLK = _P if ON else (0, 0)
HB, SOFF, ROFF, RB = layout(M, BLK) if ON else (0, 0, 0, 576)
FMT = (M | (BLK << 4)) if ON else 0      # csrc fmt code (0 = fp8 rows)
GMAX = {3: 7.5, 2: 7.0, 1: 6.0}.get(M, 0.0)
if ON:
    print(f"[offload] host row format {_SPEC}: E2M{M} latent, 2^k per {BLK}, {RB} B/row (fp8 576 B, "
          f"{100.0 * (RB / 576 - 1):+.1f}%)", flush=True)
# ARC_OFFLOAD_CCACHE=1 (default off): the GPU row cache lines and spill rows keep the compact
# host row bytes too (e2m2b32 only: 400 B instead of 576 B per line, so ~1.42x the lines in the same memory). Every
# writer copies the host row as it is; the decode attention (ARC_OFFLOAD_ATTN2 kernel 5) gathers compact rows and
# dequantizes them in shared memory to the same e4m3 bytes, so outputs are bit-identical to the fp8 GPU rows.
CC = ON and os.environ.get("ARC_OFFLOAD_CCACHE", "0") == "1"
if CC and (M, BLK) != (2, 32):
    raise RuntimeError(f"ARC_OFFLOAD_CCACHE=1 needs ARC_OFFLOAD_ROW_FMT=e2m2 (b32), not {_SPEC}")
if os.environ.get("ARC_OFFLOAD_CCACHE", "0") == "1" and not ON:
    raise RuntimeError("ARC_OFFLOAD_CCACHE=1 needs compact host rows (ARC_OFFLOAD_ROW_FMT=e2m2)")
LB = RB if CC else 576                  # bytes per GPU row (cache line / spill row)
FMTX = (FMT | (1 << 16)) if CC else FMT  # csrc fmt code with the compact-GPU-row bit
if CC:
    print(f"[offload] ARC_OFFLOAD_CCACHE: compact GPU row cache, {LB} B/line", flush=True)


# ---------------------------------------------------------------- Triton kernels
@triton.jit
def _quant(x, M: tl.constexpr, BLK: tl.constexpr):
    """x [512] fp32 (clamped to +-448) -> codes [512] int32, block exponents k [512 // BLK] int32"""
    NB: tl.constexpr = 512 // BLK
    GM: tl.constexpr = 7.5 if M == 3 else (7.0 if M == 2 else 6.0)
    ax = tl.abs(x)
    am = tl.max(tl.reshape(ax, (NB, BLK)), 1)
    r = am / GM
    rb = r.to(tl.int32, bitcast=True)
    e = ((rb >> 23) & 255) - 127
    k = e + ((rb & 0x7FFFFF) != 0).to(tl.int32)          # smallest k with 2^k >= amax / gmax (exact)
    k = tl.where(e < -126, -126, tl.minimum(k, 127))      # zero / denormal blocks: codes 0
    kx = tl.reshape(tl.broadcast_to(k[:, None], (NB, BLK)), (512,))
    inv = ((127 - kx) << 23).to(tl.float32, bitcast=True)   # 2^-k (k >= -126 -> exponent field <= 253)
    y = ax * inv
    ONE: tl.constexpr = 1 << M
    sub = tl.floor(y * ONE + 0.5).to(tl.int32)                        # y < 1: subnormal step 2^-M
    e2 = (y >= 2.0).to(tl.int32) + (y >= 4.0).to(tl.int32)
    stp = ((e2 - M + 127) << 23).to(tl.float32, bitcast=True)        # 2^(e2 - M)
    mant = tl.floor(y / stp + 0.5).to(tl.int32)                       # in [2^M, 2^(M+1)] (carry -> next binade)
    nrm = ((e2 + 1) << M) + mant - ONE
    mag = tl.minimum(tl.where(y < 1.0, sub, nrm), (1 << (M + 2)) - 1)
    code = tl.where((x < 0) & (mag > 0), mag | (1 << (M + 2)), mag)
    return code, k


@triton.jit
def _store_kernel(kvc_ptr, kpe_ptr, slot_ptr, host_ptr, inv_scale, R, stride_kvc, stride_kpe,
                  M: tl.constexpr, BLK: tl.constexpr, SOFF: tl.constexpr, ROFF: tl.constexpr, RB: tl.constexpr):
    """one new token -> its compact host row (store(); write-through of the cache is not used with W > 1)"""
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if (slot < 0) | (slot >= R):
        return
    x = tl.clamp(tl.load(kvc_ptr + t * stride_kvc + tl.arange(0, 512)).to(tl.float32) * inv_scale, -448.0, 448.0)
    code, k = _quant(x, M, BLK)
    base = host_ptr + slot * RB
    lo = tl.reshape(code & 15, (64, 8)).to(tl.uint32) << (4 * tl.arange(0, 8))[None, :].to(tl.uint32)
    tl.store(base.to(tl.pointer_type(tl.uint32), bitcast=True) + tl.arange(0, 64), tl.sum(lo, 1))
    if M == 3:
        hi = tl.reshape(code >> 4, (32, 16)).to(tl.uint32) << (2 * tl.arange(0, 16))[None, :].to(tl.uint32)
        tl.store((base + 256).to(tl.pointer_type(tl.uint32), bitcast=True) + tl.arange(0, 32), tl.sum(hi, 1))
    if M == 2:
        hi = tl.reshape(code >> 4, (16, 32)).to(tl.uint32) << tl.arange(0, 32)[None, :].to(tl.uint32)
        tl.store((base + 256).to(tl.pointer_type(tl.uint32), bitcast=True) + tl.arange(0, 16), tl.sum(hi, 1))
    NB: tl.constexpr = 512 // BLK
    tl.store((base + SOFF).to(tl.pointer_type(tl.int8), bitcast=True) + tl.arange(0, NB), k.to(tl.int8))
    b = tl.clamp(tl.load(kpe_ptr + t * stride_kpe + tl.arange(0, 64)).to(tl.float32) * inv_scale, -448.0, 448.0)
    tl.store((base + ROFF).to(tl.pointer_type(tl.float8e4nv), bitcast=True) + tl.arange(0, 64), b.to(tl.float8e4nv))


@triton.jit
def _dq_rows(host_ptr, h, ok, M: tl.constexpr, BLK: tl.constexpr, SOFF: tl.constexpr, RB: tl.constexpr,
             B: tl.constexpr):
    """compact host rows h [B] (int64; ok mask) -> latent e4m3 bytes packed 4 per int32 [B, 128]"""
    base = host_ptr + h * RB
    w = tl.load((base[:, None] + 4 * tl.arange(0, 64)[None, :]).to(tl.pointer_type(tl.int32), bitcast=True),
                mask=ok[:, None], other=0)
    code = tl.reshape((w[:, :, None] >> (4 * tl.arange(0, 8))[None, None, :]) & 15, (B, 512))
    if M == 3:
        hw = tl.load((base[:, None] + 256 + 4 * tl.arange(0, 32)[None, :]).to(tl.pointer_type(tl.int32), bitcast=True),
                     mask=ok[:, None], other=0)
        code = code | (tl.reshape((hw[:, :, None] >> (2 * tl.arange(0, 16))[None, None, :]) & 3, (B, 512)) << 4)
    if M == 2:
        hw = tl.load((base[:, None] + 256 + 4 * tl.arange(0, 16)[None, :]).to(tl.pointer_type(tl.int32), bitcast=True),
                     mask=ok[:, None], other=0)
        code = code | (tl.reshape((hw[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1, (B, 512)) << 4)
    NB: tl.constexpr = 512 // BLK
    k = tl.load((base[:, None] + SOFF + tl.arange(0, NB)[None, :]).to(tl.pointer_type(tl.int8), bitcast=True),
                mask=ok[:, None], other=0).to(tl.int32)
    kx = tl.reshape(tl.broadcast_to(k[:, :, None], (B, NB, BLK)), (B, 512))
    mag = code & ((1 << (M + 2)) - 1)
    e2 = mag >> M
    mm = mag & ((1 << M) - 1)
    mant = tl.where(e2 > 0, mm + (1 << M), mm)
    ex = tl.minimum(tl.maximum(tl.maximum(e2, 1) - 1 - M + kx, -126), 127)
    v = mant.to(tl.float32) * ((ex + 127) << 23).to(tl.float32, bitcast=True)
    v = tl.clamp(tl.where(((code >> (M + 2)) & 1) != 0, -v, v), -448.0, 448.0)
    u8 = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.uint32)
    return tl.sum(tl.reshape(u8, (B, 128, 4)) << (8 * tl.arange(0, 4))[None, None, :].to(tl.uint32), 2)


@triton.jit
def _put_rows(host_ptr, h, ok, dst_ptr, dl, M: tl.constexpr, BLK: tl.constexpr, SOFF: tl.constexpr,
              ROFF: tl.constexpr, RB: tl.constexpr, B: tl.constexpr):
    """dequantize host rows h [B] into GPU fp8 rows dst_ptr (uint32 view) + dl [B] * 144 words"""
    lat = _dq_rows(host_ptr, h, ok, M, BLK, SOFF, RB, B)
    d = dst_ptr + dl.to(tl.int64)[:, None] * 144
    tl.store(d + tl.arange(0, 128)[None, :], lat, mask=ok[:, None])
    rope = tl.load((host_ptr + h * RB + ROFF)[:, None].to(tl.pointer_type(tl.uint32), bitcast=True) +
                   tl.arange(0, 16)[None, :], mask=ok[:, None])
    tl.store(d + 128 + tl.arange(0, 16)[None, :], rope, mask=ok[:, None])


@triton.jit
def _raw_rows(host_ptr, h, ok, dst_ptr, dl, RB: tl.constexpr, B: tl.constexpr):
    """compact host rows h [B] -> compact GPU rows dst_ptr (uint32 view) + dl [B] * RB / 4 words, bytes as they are"""
    w = tl.arange(0, 128)
    m = ok[:, None] & (w[None, :] < RB // 4)
    x = tl.load((host_ptr + h * RB)[:, None].to(tl.pointer_type(tl.uint32), bitcast=True) + w[None, :], mask=m)
    tl.store(dst_ptr + dl.to(tl.int64)[:, None] * (RB // 4) + w[None, :], x, mask=m)


@triton.jit
def _copy_kernel(ml_ptr, mcnt_ptr, cpd_ptr, host_ptr, cache_ptr, spill_ptr, perm_ptr, NS, B: tl.constexpr,
                 PERM: tl.constexpr, M: tl.constexpr, BLK: tl.constexpr, SOFF: tl.constexpr, ROFF: tl.constexpr,
                 RB: tl.constexpr, RAW: tl.constexpr):
    """offload_fill._copy_kernel for compact host rows (same visit order and destinations)"""
    n = tl.load(mcnt_ptr)
    for s in range(tl.program_id(0) * B, n, tl.num_programs(0) * B):
        i = s + tl.arange(0, B)
        m = i < n
        if PERM:
            i = tl.where(i < NS, tl.load(perm_ptr + i, mask=m & (i < NS), other=0), i)
        h = tl.load(ml_ptr + i, mask=m, other=0).to(tl.int64)
        d = tl.load(cpd_ptr + i, mask=m, other=-1)
        ok = m & (d != -1)
        dl = tl.abs(d + 1) - 1                     # line d (>= 0) / spill row -d - 2 (d <= -2)
        if RAW:   # ARC_OFFLOAD_CCACHE: compact GPU rows, plain copy
            _raw_rows(host_ptr, h, ok & (d >= 0), cache_ptr, dl, RB, B)
            _raw_rows(host_ptr, h, ok & (d < 0), spill_ptr, dl, RB, B)
        else:
            lat = _dq_rows(host_ptr, h, ok, M, BLK, SOFF, RB, B)
            rope = tl.load((host_ptr + h * RB + ROFF)[:, None].to(tl.pointer_type(tl.uint32), bitcast=True) +
                           tl.arange(0, 16)[None, :], mask=ok[:, None])
            mc = (ok & (d >= 0))[:, None]
            ms = (ok & (d < 0))[:, None]
            o = dl[:, None] * 144
            tl.store(cache_ptr + o + tl.arange(0, 128)[None, :], lat, mask=mc)
            tl.store(cache_ptr + o + 128 + tl.arange(0, 16)[None, :], rope, mask=mc)
            tl.store(spill_ptr + o + tl.arange(0, 128)[None, :], lat, mask=ms)
            tl.store(spill_ptr + o + 128 + tl.arange(0, 16)[None, :], rope, mask=ms)


@triton.jit
def _rows_kernel(src_ptr, n, host_ptr, dst_ptr, alt_ptr, R, MODE: tl.constexpr, M: tl.constexpr, BLK: tl.constexpr,
                 SOFF: tl.constexpr, ROFF: tl.constexpr, RB: tl.constexpr, B: tl.constexpr, RAW: tl.constexpr = 0):
    """MODE 0 gather: dst row i <- host row src[i] (invalid rows -> zeros, like _gather_rows_kernel)
    MODE 1 refill: cache line i <- host row tag[i] (tag = src, n = C; invalid: untouched)"""
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    h = tl.load(src_ptr + i, mask=m, other=-1).to(tl.int64)
    ok = m & (h >= 0) & (h < R)
    hs = tl.where(ok, h, 0)
    if MODE == 0:
        lat = _dq_rows(host_ptr, hs, ok, M, BLK, SOFF, RB, B)
        rope = tl.load((host_ptr + hs * RB + ROFF)[:, None].to(tl.pointer_type(tl.uint32), bitcast=True) +
                       tl.arange(0, 16)[None, :], mask=ok[:, None], other=0)
        d = dst_ptr + i.to(tl.int64)[:, None] * 144
        tl.store(d + tl.arange(0, 128)[None, :], tl.where(ok[:, None], lat, 0), mask=m[:, None])
        tl.store(d + 128 + tl.arange(0, 16)[None, :], rope, mask=m[:, None])
    elif RAW:
        _raw_rows(host_ptr, hs, ok, dst_ptr, i, RB, B)
    else:
        _put_rows(host_ptr, hs, ok, dst_ptr, i, M, BLK, SOFF, ROFF, RB, B)


@triton.jit
def _wt_kernel(slot_ptr, res_ptr, n, host_ptr, cache_ptr, spill_ptr, R, M: tl.constexpr, BLK: tl.constexpr,
               SOFF: tl.constexpr, ROFF: tl.constexpr, RB: tl.constexpr, B: tl.constexpr, RAW: tl.constexpr):
    """sparse_offload._wt_kernel for compact host rows"""
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    h = tl.load(slot_ptr + i, mask=m, other=-1).to(tl.int64)
    d = tl.load(res_ptr + i, mask=m, other=-1).to(tl.int64)
    ok = m & (h >= 0) & (h < R) & (d != -1)
    hs = tl.where(ok, h, 0)
    dl = tl.abs(d + 1) - 1
    if RAW:
        _raw_rows(host_ptr, hs, ok & (d >= 0), cache_ptr, dl, RB, B)
        _raw_rows(host_ptr, hs, ok & (d < 0), spill_ptr, dl, RB, B)
        return
    lat = _dq_rows(host_ptr, hs, ok, M, BLK, SOFF, RB, B)
    rope = tl.load((host_ptr + hs * RB + ROFF)[:, None].to(tl.pointer_type(tl.uint32), bitcast=True) +
                   tl.arange(0, 16)[None, :], mask=ok[:, None])
    mc = (ok & (d >= 0))[:, None]
    ms = (ok & (d < 0))[:, None]
    o = dl[:, None] * 144
    tl.store(cache_ptr + o + tl.arange(0, 128)[None, :], lat, mask=mc)
    tl.store(cache_ptr + o + 128 + tl.arange(0, 16)[None, :], rope, mask=mc)
    tl.store(spill_ptr + o + tl.arange(0, 128)[None, :], lat, mask=ms)
    tl.store(spill_ptr + o + 128 + tl.arange(0, 16)[None, :], rope, mask=ms)


# ---------------------------------------------------------------- host API (sparse_offload / offload_fill call these)
_KW = None


def _kw():
    global _KW
    if _KW is None:
        _KW = dict(M=M, BLK=BLK, SOFF=SOFF, ROFF=ROFF, RB=RB)
    return _KW


def _u32(t):
    return t.view(torch.uint8).view(torch.int32) if t.dtype != torch.int32 else t


def check(lo):
    """the serving paths that understand compact host rows; raise for anything else"""
    from . import sparse_offload as so, offload_fill as of, offload_cuda
    why = []
    if lo.W <= 1:
        why.append("W-way cache (ARC_OFFLOAD_WAYS > 1, C % W == 0)")
    if not so._RESOLVE:
        why.append("ARC_OFFLOAD_RESOLVE=1")
    if not (of.SPLIT or of.TWO):
        why.append("the split fill (ARC_OFFLOAD_FILL_SPLIT=1 or ARC_OFFLOAD_TWO_CHOICE=1)")
    if not so._CUDA:
        why.append("the CUDA split attention (ARC_OFFLOAD_CUDA=1)")
    if so._PACKED or so._CUDA_PK or so._PREFETCH:
        why.append("no ARC_OFFLOAD_PACKED / ARC_OFFLOAD_CUDA_PK / ARC_OFFLOAD_PREFETCH")
    if CC:
        from . import offload_cuda2 as oc2
        if not (oc2.MODE > 0 and oc2.KERN == 5 and oc2.ST == 4 and oc2.available() and so._CUDA_SPLIT == 1
                and not oc2._CMP_N):
            why.append("ARC_OFFLOAD_CCACHE: the decode attention ARC_OFFLOAD_ATTN2=1|2 kernel 5, ST 4, CUDA split 1, "
                       "no ARC_OFFLOAD_ATTN2_CMP")
        if so._AB > 0:
            why.append("ARC_OFFLOAD_CCACHE: no ARC_OFFLOAD_AB (an 'attn2'-less window would need the v1 kernel)")
    if not offload_cuda.hostfmt_available():
        why.append(f"an offload_attn.so with the compact-row entry points (rebuild: dots3/build_offload_attn.sh)")
    if why:
        raise RuntimeError(f"ARC_OFFLOAD_ROW_FMT={_SPEC} needs " + "; ".join(why))


def store(host, kv_c_normed, kpe, slots, n, inv_scale, R):
    _store_kernel[(n,)](kv_c_normed, kpe, slots, host, inv_scale, R, kv_c_normed.stride(0), kpe.stride(0),
                        num_warps=4, **_kw())


def copy(ml, mcnt, cpd, host, cache, spill, perm, ns, grid, B=32, num_warps=4):
    _copy_kernel[(grid,)](ml, mcnt, cpd, host, _u32(cache), _u32(spill), perm, ns, B=B, PERM=int(ns > 0),
                          num_warps=num_warps, RAW=int(CC), **_kw())


def gather(host, rows, out, R):
    n = rows.shape[0]
    if n:
        _rows_kernel[(triton.cdiv(n, 32),)](rows, n, host, _u32(out), out, R, MODE=0, B=32, num_warps=4, **_kw())


def refill(host, tag, cache, R):
    C = tag.shape[0]
    _rows_kernel[(triton.cdiv(C, 32),)](tag, C, host, _u32(cache), cache, R, MODE=1, B=32, num_warps=4, RAW=int(CC),
                                        **_kw())


def write_through(slots, res, n, host, cache, spill, R):
    _wt_kernel[(triton.cdiv(n, 32),)](slots, res, n, host, _u32(cache), _u32(spill), R, B=32, num_warps=4, RAW=int(CC),
                                      **_kw())


# ---------------------------------------------------------------- torch reference (tests / analysis)
def ref_quant(x, M_=None, BLK_=None):
    """x [N, 512] float -> (codes int32 [N, 512], k int32 [N, 512 // BLK]) with the store kernel's math"""
    M_, BLK_ = M_ or M, BLK_ or BLK
    gm = {3: 7.5, 2: 7.0, 1: 6.0}[M_]
    x = x.float().clamp(-448, 448)
    N = x.shape[0]
    ax = x.abs().view(N, -1, BLK_)
    r = ax.amax(-1) / gm
    rb = r.view(torch.int32)
    e = ((rb >> 23) & 255) - 127
    k = e + ((rb & 0x7FFFFF) != 0).int()
    k = torch.where(e < -126, torch.full_like(k, -126), k.clamp(max=127))
    y = (ax * torch.exp2(-k.float())[..., None]).view(N, 512)
    one = 1 << M_
    sub = torch.floor(y * one + 0.5).int()
    e2 = (y >= 2).int() + (y >= 4).int()
    mant = torch.floor(y / torch.exp2((e2 - M_).float()) + 0.5).int()
    nrm = ((e2 + 1) << M_) + mant - one
    mag = torch.where(y < 1, sub, nrm).clamp(max=(1 << (M_ + 2)) - 1)
    code = torch.where((x < 0) & (mag > 0), mag | (1 << (M_ + 2)), mag)
    return code, k


def ref_dequant(code, k, M_=None, BLK_=None):
    """-> float32 [N, 512] values before the e4m3 cast (exact e4m3 values when in range)"""
    M_, BLK_ = M_ or M, BLK_ or BLK
    mag = code & ((1 << (M_ + 2)) - 1)
    e2 = mag >> M_
    mm = mag & ((1 << M_) - 1)
    mant = torch.where(e2 > 0, mm + (1 << M_), mm)
    ex = (e2.clamp(min=1) - 1 - M_ + k.repeat_interleave(BLK_, 1)).clamp(-126, 127)
    v = mant.float() * torch.exp2(ex.float())
    return torch.where(((code >> (M_ + 2)) & 1) != 0, -v, v).clamp(-448, 448)


def ref_pack(code, k, rope_f8, M_=None, BLK_=None):
    """torch: codes / exponents / rope e4m3 [N, 64] -> compact rows uint8 [N, RB]"""
    M_, BLK_ = M_ or M, BLK_ or BLK
    hb, soff, roff, rb = layout(M_, BLK_)
    N = code.shape[0]
    out = torch.zeros((N, rb), dtype=torch.uint8, device=code.device)
    lo = (code & 15).view(N, 256, 2)
    out[:, :256] = (lo[..., 0] | (lo[..., 1] << 4)).to(torch.uint8)
    hi = code >> 4
    if M_ == 3:
        h = hi.view(N, 128, 4)
        out[:, 256:384] = (h[..., 0] | (h[..., 1] << 2) | (h[..., 2] << 4) | (h[..., 3] << 6)).to(torch.uint8)
    if M_ == 2:
        h = hi.view(N, 64, 8)
        out[:, 256:320] = sum(h[..., j] << j for j in range(8)).to(torch.uint8)
    out[:, soff:roff] = k.to(torch.int8).view(torch.uint8)
    out[:, roff:] = rope_f8.view(torch.uint8)
    return out
