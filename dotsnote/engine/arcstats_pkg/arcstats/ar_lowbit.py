"""Sub-int6 host all-reduce payloads (default off, needs ARC_HOST_AR=1 ARC_HOST_AR_Q8=1).

The packed two-shot kernel of host_ar.py (_host_ar_qb_kernel: same phases, flags, per-block epochs, slices, slot
layout and summation order) extended with:

  * 4-bit payloads: int4 (codes -7..7) or e2m1 / FP4 (levels 0, .5, 1, 1.5, 2, 3, 4, 6), per-128 fp32 absmax scales;
    one int32 nibble word per 8 elements (B = 4: no high-bit plane). 5 / 6 bit as in host_ar.py.
  * Cross-layer error feedback (EF): within ONE forward pass the residual stream is a plain sum of the all-reduce
    outputs (h_L = emb + sum_l AR_l(x_l), every AR over the same [T, H] token rows), so the quantization error E_l of
    all-reduce l can be cancelled exactly in the residual stream by adding -E_l to the next all-reduce's inputs.
    Every rank keeps a bf16 carry c [n] = minus its own share of E_l (peer quarters: x' - dq(Q(x')) of what it sent;
    own quarter: acc - dq(Q(acc)) of the requantized sum) and adds it to its input of all-reduce l+1 before
    quantizing. The residual stream then carries only the LAST all-reduce's error instead of the sum of all ~93,
    plus the (uncorrected) nonlinear effect of each error on the next sublayer. The carry never crosses forward passes
    (decode tokens differ per step, so step-to-step EF would inject another token's error): a hook on the main
    model's and the MTP predictor's forward resets the chain, and only same-sized hidden-state all-reduces (x.shape[-1]
    == hidden) join it.
  * Optional per-group Hadamard rotation (HAD): codes are sent in the rotated domain (128-point Walsh-Hadamard via one
    fp16 tensor-core dot per tile; the inverse of integer codes is exact), spreading outlier channels over the group.
  * Per-site bits: main-model attention / MLP(MoE) / embedding all-reduces and the MTP draft all-reduces (which only
    change acceptance, never correctness: verification is exact) each get their own peer / reduced widths.

All ranks still produce bit-identical outputs (everyone dequantizes the same reduced codes). CUDA-graph capturable:
the role / chain position of each call is fixed at capture time (the breakable-graph capture runs the Python forward).

Env (all optional, default off):
  ARC_AR_LB=1                         enable
  ARC_AR_LB_MAIN=B/RB[f]              main-model all-reduces: peer bits / reduced bits, 'f' suffix on a width = e2m1
                                      (4 bit only). Default 6/6 (= ARC_HOST_AR_BITS 6). e.g. 4f/6, 4/4, 5/5
  ARC_AR_LB_ATTN=B/RB, ARC_AR_LB_MLP=B/RB, ARC_AR_LB_EMB=B/RB   per-kind overrides inside the main model
  ARC_AR_LB_DRAFT=B/RB                MTP draft all-reduces (default = MAIN)
  ARC_AR_LB_EF=1                      cross-layer error feedback (main and draft chains)
  ARC_AR_LB_HAD=1|8                   Hadamard rotation (all chain all-reduces; reduced width must not be 16): 1 = 128-point
                                      per scale group (fp16 tensor-core dot; its ~40 KB of shared memory per CTA made the
                                      all-reduce CTAs wait for SM space next to the offload kernels in serving: 2.6x slower
                                      steps on L4), 8 = 8-point in registers (each thread's 8 consecutive elements, no
                                      shared memory), s = 128-point as 7 radix-2 stages written with
                                      reshape/permute/split/join (no tensor-core dot; Triton's layout conversions only)
  ARC_AR_LB_E8=1                      UE8M0 scales: one byte (power of two >= absmax / qmax) per 128 values instead of fp32
                                      (-0.023 B per element; costs up to one bit of step size per group)
  ARC_AR_LB_MAXNREG=64                registers per thread (0 = compiler choice). EF raises the kernel to 128 registers x 512
                                      threads = a whole SM's register file, so its CTAs cannot share an SM with the
                                      ARC_OFFLOAD_AR_GATE-paused mirror copy CTAs (which spin until the all-reduce ends):
                                      serving steps collapsed 2.6-3.7x under Kaggle-like all-reduce times. 64 (int6
                                      kernel: 59) restores co-residency without local-memory spills.
  ARC_AR_LB_LOG=1                     log the per-forward chain lengths of the first forwards
  ARC_AR_LB_CTL=file.json             TEST ONLY (eager servers, --enforce-eager): re-read {"main": "B/RB", "attn": ..,
                                      "mlp": .., "emb": .., "draft": .., "ef": 0/1, "had": 0/1} at every main forward
                                      start when the file changed (change it only while the server is idle: every rank
                                      must see the same config for the same all-reduce)
"""
import os

import torch
import triton
import triton.language as tl

from . import host_ar as H
from .host_ar import _wait, _emu_io, _emu_words, _split8, _join8

EMU_W = H.EMU_WORDS

ON = os.environ.get("ARC_AR_LB") == "1"
EF = os.environ.get("ARC_AR_LB_EF") == "1"
MAXNREG = int(os.environ.get("ARC_AR_LB_MAXNREG", "64"))   # register cap: co-residency with the gated mirror copy
E8S = os.environ.get("ARC_AR_LB_E8") == "1"   # UE8M0 one-byte power-of-two scales instead of fp32
HAD = {"1": 1, "8": 2, "s": 3}.get(os.environ.get("ARC_AR_LB_HAD", ""), 0)
LOG = os.environ.get("ARC_AR_LB_LOG", "1") == "1"


def _w(s):
    """'4f' -> (4, 1); '6' -> (6, 0); '16' -> (16, 0)"""
    s = s.strip()
    f = s.endswith("f")
    b = int(s[:-1] if f else s)
    assert (b in (4, 5, 6, 8) and (not f or b == 4)) or (b == 16 and not f), s
    return b, int(f)


def _pair(s, dflt):
    if not s:
        return dflt
    a, _, b = s.partition("/")
    pb = _w(a)
    return pb + (_w(b) if b else pb)


_MAIN = _pair(os.environ.get("ARC_AR_LB_MAIN"), (6, 0, 6, 0))
CFG = {("main", "attn"): _pair(os.environ.get("ARC_AR_LB_ATTN"), _MAIN),
       ("main", "mlp"): _pair(os.environ.get("ARC_AR_LB_MLP"), _MAIN),
       ("main", "emb"): _pair(os.environ.get("ARC_AR_LB_EMB"), _MAIN)}
_DR = _pair(os.environ.get("ARC_AR_LB_DRAFT"), _MAIN)
for k in ("attn", "mlp", "emb"):
    CFG[("draft", k)] = _DR
for (_, _), (b, f, rb, rf) in CFG.items():
    assert b != 16 and b != 8, "peer payload must be 4..6 bit (8 = use the q8 kernel)"
    assert not (HAD and rb == 16), "ARC_AR_LB_HAD needs a quantized reduced quarter"

CTL = os.environ.get("ARC_AR_LB_CTL")
_CTL_M = [None]


def _reload():
    """ARC_AR_LB_CTL: switch widths / EF / HAD at runtime (eager test servers only)"""
    global EF, HAD
    try:
        m = os.stat(CTL).st_mtime_ns
    except OSError:
        return
    if m == _CTL_M[0]:
        return
    _CTL_M[0] = m
    import json
    c = json.load(open(CTL))
    mn = _pair(c.get("main"), (6, 0, 6, 0))
    for k in ("attn", "mlp", "emb"):
        CFG[("main", k)] = _pair(c.get(k), mn)
        CFG[("draft", k)] = _pair(c.get("draft"), mn)
    global E8S
    E8S = bool(c.get("e8", 0))
    EF, HAD = bool(c.get("ef", 0)), {"0": 0, "1": 1, "8": 2, "s": 3}[str(c.get("had", 0))]
    print(f"[ar_lowbit] ctl reload: {c}", flush=True)


# chain state: role ('main' / 'draft' / None = outside a hooked forward), position, element count of the chain
_ST = {"role": None, "pos": 0, "n": 0}
_HID = int(os.environ.get("ARC_AR_LB_HIDDEN", "5120"))
STATS = {"chain": 0, "other": 0}
_LOGGED = []
_CY = {}


# ------------------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _lv(c, FP4: tl.constexpr):
    """signed code (int32) -> level (fp32): int: c; e2m1: sign(c) * [0, .5, 1, 1.5, 2, 3, 4, 6][|c|]"""
    if FP4:
        m = tl.abs(c)
        mf = m.to(tl.float32)
        L = tl.where(m <= 4, mf * 0.5, tl.where(m == 7, 6.0, mf - 2.0))
        return tl.where(c < 0, -L, L)
    else:
        return c.to(tl.float32)


@triton.jit
def _qz(v, G: tl.constexpr, BLOCK: tl.constexpr, B: tl.constexpr, FP4: tl.constexpr, E8: tl.constexpr = False):
    """fp32 [BLOCK] -> (offset-binary codes int32 [BLOCK] in [1, 2^B - 1], fp32 scale [BLOCK // G]); per-G absmax.
    int: qmax 2^(B-1) - 1, round half away (host_ar._qb); e2m1: scale absmax / 6, nearest level."""
    O: tl.constexpr = 1 << (B - 1)
    v2 = tl.reshape(v, (BLOCK // G, G))
    if FP4:
        sc = tl.maximum(tl.max(tl.abs(v2), axis=1), 1e-30) / 6.0
        if E8:   # UE8M0: next power of two >= the absmax scale (no clipping)
            sc = tl.exp2(tl.ceil(tl.log2(sc)))
        r = v2 / sc[:, None]
        a = tl.abs(r)
        m = (a >= 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a >= 1.25).to(tl.int32) + \
            (a >= 1.75).to(tl.int32) + (a >= 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a >= 5.0).to(tl.int32)
        s = tl.where(r < 0, -m, m)
        return tl.reshape(s, (BLOCK,)) + O, sc
    else:
        QM: tl.constexpr = O - 1
        sc = tl.maximum(tl.max(tl.abs(v2), axis=1), 1e-30) / QM
        if E8:
            sc = tl.exp2(tl.ceil(tl.log2(sc)))
        r = v2 / sc[:, None]
        qv = tl.where(r >= 0, tl.floor(r + 0.5), tl.ceil(r - 0.5))
        return tl.reshape(qv, (BLOCK,)).to(tl.int32) + O, sc


@triton.jit
def _sst(p, g, sc, m, E8: tl.constexpr):
    """store per-group scales at byte p: fp32, or UE8M0 (one byte: exponent + 127)"""
    if E8:
        tl.store(p.to(tl.pointer_type(tl.uint8)) + g, (tl.log2(sc) + 127.0).to(tl.uint8), mask=m)
    else:
        tl.store(p.to(tl.pointer_type(tl.float32)) + g, sc, mask=m)


@triton.jit
def _sld(p, g, m, E8: tl.constexpr):
    if E8:
        e = tl.load(p.to(tl.pointer_type(tl.uint8)) + g, mask=m, other=127, cache_modifier=".cv")
        return tl.exp2(e.to(tl.float32) - 127.0)
    else:
        return tl.load(p.to(tl.pointer_type(tl.float32)) + g, mask=m, other=0.0, cache_modifier=".cv")


@triton.jit
def _dq(c, sc, G: tl.constexpr, BLOCK: tl.constexpr, FP4: tl.constexpr):
    """signed codes [BLOCK] -> fp32 values (in the quantized domain)"""
    return tl.reshape(tl.reshape(_lv(c, FP4), (BLOCK // G, G)) * sc[:, None], (BLOCK,))


@triton.jit
def _hmat(G: tl.constexpr):
    """+-1 Walsh-Hadamard matrix [G, G] (fp16, exact): H[i, j] = (-1)^popcount(i & j)"""
    i = tl.arange(0, G)[:, None]
    j = tl.arange(0, G)[None, :]
    p = i & j
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    return (1 - 2 * (p & 1)).to(tl.float16)


@triton.jit
def _had(v, Hm, G: tl.constexpr, BLOCK: tl.constexpr):
    """orthonormal per-group Walsh-Hadamard of fp32 [BLOCK]: group rows normalized by their absmax, fp16 dot, fp32 acc"""
    v2 = tl.reshape(v, (BLOCK // G, G))
    m = tl.maximum(tl.max(tl.abs(v2), axis=1), 1e-30)
    y = tl.dot((v2 / m[:, None]).to(tl.float16), Hm, out_dtype=tl.float32)
    return tl.reshape(y * (m * 0.08838834764831845)[:, None], (BLOCK,))


@triton.jit
def _h8(v, BLOCK: tl.constexpr):
    """unnormalized 8-point Walsh-Hadamard of each thread's 8 consecutive elements (in registers, exact on integers)"""
    a, b, c, d, e, f, g, h = _split8(v, BLOCK)
    t0, t1, t2, t3, t4, t5, t6, t7 = a + b, a - b, c + d, c - d, e + f, e - f, g + h, g - h
    u0, u1, u2, u3, u4, u5, u6, u7 = t0 + t2, t1 + t3, t0 - t2, t1 - t3, t4 + t6, t5 + t7, t4 - t6, t5 - t7
    return _join8(u0 + u4, u1 + u5, u2 + u6, u3 + u7, u0 - u4, u1 - u5, u2 - u6, u3 - u7, BLOCK)


@triton.jit
def _hstage(x, S: tl.constexpr, BLOCK: tl.constexpr):
    """one radix-2 Walsh-Hadamard stage over index bit log2(S): pairs (e, e + S) inside each 2S block -> (a + b, a - b).
    Pure tensor semantics (reshape / permute / split / join): correct for any register layout Triton picks."""
    x3 = tl.permute(tl.reshape(x, (BLOCK // (2 * S), 2, S)), (0, 2, 1))
    a, b = tl.split(x3)
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 2, 1)), (BLOCK,))


@triton.jit
def _hs(v, BLOCK: tl.constexpr):
    """unnormalized 128-point Walsh-Hadamard over each aligned 128-element group (7 radix-2 stages)"""
    y = _hstage(v, 1, BLOCK)
    y = _hstage(y, 2, BLOCK)
    y = _hstage(y, 4, BLOCK)
    y = _hstage(y, 8, BLOCK)
    y = _hstage(y, 16, BLOCK)
    y = _hstage(y, 32, BLOCK)
    return _hstage(y, 64, BLOCK)


@triton.jit
def _rot(v, Hm, G: tl.constexpr, BLOCK: tl.constexpr, HAD: tl.constexpr):
    """forward rotation of fp32 [BLOCK] (orthonormal): HAD 1 = 128-point dot, 2 = 8-point in registers,
    3 = 128-point radix-2 stages"""
    if HAD == 1:
        return _had(v, Hm, G, BLOCK)
    elif HAD == 3:
        return _hs(v, BLOCK) * 0.08838834764831845
    else:
        return _h8(v, BLOCK) * 0.35355339059327373


@triton.jit
def _dq_out(c, sc, Hm, G: tl.constexpr, BLOCK: tl.constexpr, FP4: tl.constexpr, HAD: tl.constexpr):
    """signed codes -> fp32 output values in the ORIGINAL domain (HAD: exact integer/level dot with H, then scale)"""
    if HAD == 3:
        y = _hs(_lv(c, FP4), BLOCK)   # exact: small integers / e2m1 levels
        return tl.reshape(tl.reshape(y, (BLOCK // G, G)) * (sc * 0.08838834764831845)[:, None], (BLOCK,))
    elif HAD == 2:
        y = _h8(_lv(c, FP4), BLOCK)   # exact: small integers / e2m1 levels
        return tl.reshape(tl.reshape(y, (BLOCK // G, G)) * (sc * 0.35355339059327373)[:, None], (BLOCK,))
    elif HAD == 1:
        L = tl.reshape(_lv(c, FP4), (BLOCK // G, G)).to(tl.float16)   # |levels| <= 127 or e2m1: exact in fp16
        y = tl.dot(L, Hm, out_dtype=tl.float32)
        return tl.reshape(y * (sc * 0.08838834764831845)[:, None], (BLOCK,))
    else:
        return _dq(c, sc, G, BLOCK, FP4)


@triton.jit
def _pk(p, w0, u, wm, m, PS, BLOCK: tl.constexpr, B: tl.constexpr):
    """host_ar._pack_store plus B = 4 (nibble plane only)"""
    a, b, c, d, e, f, g, h = _split8(u, BLOCK)
    lo = (a & 15) | ((b & 15) << 4) | ((c & 15) << 8) | ((d & 15) << 12) | ((e & 15) << 16) | ((f & 15) << 20) \
        | ((g & 15) << 24) | ((h & 15) << 28)
    wi = w0 + tl.arange(0, BLOCK // 8)
    tl.store(p.to(tl.pointer_type(tl.int32)) + wi, lo, mask=wm)
    if B > 4:
        HB: tl.constexpr = B - 4
        hi = (a >> 4) | ((b >> 4) << HB) | ((c >> 4) << (2 * HB)) | ((d >> 4) << (3 * HB)) | ((e >> 4) << (4 * HB)) \
            | ((f >> 4) << (5 * HB)) | ((g >> 4) << (6 * HB)) | ((h >> 4) << (7 * HB))
        if HB == 1:
            tl.store((p + PS) + wi, hi.to(tl.int8), mask=wm)
        else:
            tl.store((p + PS).to(tl.pointer_type(tl.int16)) + wi, hi.to(tl.int16), mask=wm)


@triton.jit
def _upk(p, w0, wm, m, PS, BLOCK: tl.constexpr, B: tl.constexpr):
    """inverse of _pk -> signed int32 codes [BLOCK]"""
    wi = w0 + tl.arange(0, BLOCK // 8)
    lo = tl.load(p.to(tl.pointer_type(tl.int32)) + wi, mask=wm, other=0, cache_modifier=".cv")
    O: tl.constexpr = 1 << (B - 1)
    if B == 4:
        a = (lo & 15) - O
        b = ((lo >> 4) & 15) - O
        c = ((lo >> 8) & 15) - O
        d = ((lo >> 12) & 15) - O
        e = ((lo >> 16) & 15) - O
        f = ((lo >> 20) & 15) - O
        g = ((lo >> 24) & 15) - O
        h = ((lo >> 28) & 15) - O
    else:
        HB: tl.constexpr = B - 4
        if HB == 1:
            hi = tl.load((p + PS) + wi, mask=wm, other=0, cache_modifier=".cv").to(tl.int32) & 255
        else:
            hi = tl.load((p + PS).to(tl.pointer_type(tl.int16)) + wi, mask=wm, other=0,
                         cache_modifier=".cv").to(tl.int32) & 65535
        HM: tl.constexpr = (1 << HB) - 1
        a = ((lo & 15) | ((hi & HM) << 4)) - O
        b = (((lo >> 4) & 15) | (((hi >> HB) & HM) << 4)) - O
        c = (((lo >> 8) & 15) | (((hi >> (2 * HB)) & HM) << 4)) - O
        d = (((lo >> 12) & 15) | (((hi >> (3 * HB)) & HM) << 4)) - O
        e = (((lo >> 16) & 15) | (((hi >> (4 * HB)) & HM) << 4)) - O
        f = (((lo >> 20) & 15) | (((hi >> (5 * HB)) & HM) << 4)) - O
        g = (((lo >> 24) & 15) | (((hi >> (6 * HB)) & HM) << 4)) - O
        h = (((lo >> 28) & 15) | (((hi >> (7 * HB)) & HM) << 4)) - O
    return _join8(a, b, c, d, e, f, g, h, BLOCK)


@triton.jit
def _ar_lb_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, hred16_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, cy_ptr,
                  q, per, maxn, emu_ptr, sink_ptr, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr,
                  FS: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr, B: tl.constexpr, FP4: tl.constexpr,
                  RB: tl.constexpr, RFP4: tl.constexpr, EF: tl.constexpr, LOADC: tl.constexpr, HAD: tl.constexpr, E8: tl.constexpr,
                  TMO: tl.constexpr = 0, LD: tl.constexpr = False, EMU: tl.constexpr = 1, EB: tl.constexpr = 4096):
    """host_ar._host_ar_qb_kernel (same phases / flags / epochs / slot layout / summation order) with 4-bit int / e2m1
    payloads, cross-layer error feedback through the bf16 carry cy (EF: store, LOADC: add the previous all-reduce's
    carry to this input) and an optional per-group Hadamard rotation (codes, sums and the carry live in the rotated
    domain; outputs are rotated back)."""
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    goffs = tl.arange(0, BLOCK // G)
    slot = 2 * maxn
    PS = maxn // 2
    w8 = tl.arange(0, BLOCK // 8) * 8
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * slot
    emp = emu_ptr + b * EMU_W
    OB: tl.constexpr = 1 << (B - 1)
    ORB: tl.constexpr = 1 << (RB - 1)
    if HAD == 1:
        Hm = _hmat(G)
    else:
        Hm = 0
    jk = 0
    for j in tl.static_range(W):
        if j != RANK:
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                v = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
                if HAD:
                    v = _rot(v, Hm, G, BLOCK, HAD)
                if LOADC:
                    v += tl.load(cy_ptr + base + o, mask=m, other=0.0).to(tl.float32)
                u, sc = _qz(v, G, BLOCK, B, FP4, E8)
                _pk(my_in, (base + s) // 8, u, (s + w8) < hi, m, PS, BLOCK, B)
                g = (base + s) // G + goffs
                _sst(my_in + maxn, g, sc, (s + goffs * G) < hi, E8)
                if EF:
                    tl.store(cy_ptr + base + o, (v - _dq(u - OB, sc, G, BLOCK, FP4)).to(tl.bfloat16), mask=m)
    if EMU > 1:
        jk += _emu_io(emp, (W - 1) * _emu_words(hi - lo, B, G), 0, EMU, EB)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem="release", scope="sys")
    for k in tl.static_range(W):
        if k != RANK:
            _wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 710 + k, TMO, LD)
    base = RANK * q
    red = hred_ptr + par.to(tl.int64) * slot
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        m = o < hi
        gm = (s + goffs * G) < hi
        g = (base + s) // G + goffs
        acc = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        if HAD:
            acc = _rot(acc, Hm, G, BLOCK, HAD)
        if LOADC:
            acc += tl.load(cy_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                c = _upk(pk, (base + s) // 8, (s + w8) < hi, m, PS, BLOCK, B)
                ps = _sld(pk + maxn, g, gm, E8)
                acc += _dq(c, ps, G, BLOCK, FP4)
        if RB == 16:
            r16 = acc.to(tl.bfloat16)
            tl.store(out_ptr + base + o, r16, mask=m)
            tl.store(hred16_ptr + par.to(tl.int64) * maxn + base + o, r16, mask=m)
            if EF:
                tl.store(cy_ptr + base + o, (acc - r16.to(tl.float32)).to(tl.bfloat16), mask=m)
        else:
            u, sc = _qz(acc, G, BLOCK, RB, RFP4, E8)
            tl.store(out_ptr + base + o, _dq_out(u - ORB, sc, Hm, G, BLOCK, RFP4, HAD).to(tl.bfloat16), mask=m)
            _pk(red, (base + s) // 8, u, (s + w8) < hi, m, PS, BLOCK, RB)
            _sst(red + maxn, g, sc, gm, E8)
            if EF:
                tl.store(cy_ptr + base + o, (acc - _dq(u - ORB, sc, G, BLOCK, RFP4)).to(tl.bfloat16), mask=m)
    if EMU > 1:
        jk += _emu_io(emp, _emu_words(hi - lo, RB, G), (W - 1) * _emu_words(hi - lo, B, G), EMU, EB)
    tl.debug_barrier()
    tl.atomic_xchg(f2_ptr + (RANK * NBLK + b) * FS, e, sem="release", scope="sys")
    for j in tl.static_range(W):
        if j != RANK:
            _wait(f2_ptr + (j * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 720 + j, TMO, LD)
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                if RB == 16:
                    tl.store(out_ptr + base + o, tl.load(hred16_ptr + par.to(tl.int64) * maxn + base + o, mask=m,
                                                         cache_modifier=".cv"), mask=m)
                else:
                    g = (base + s) // G + goffs
                    c = _upk(red, (base + s) // 8, (s + w8) < hi, m, PS, BLOCK, RB)
                    ps = _sld(red + maxn, g, (s + goffs * G) < hi, E8)
                    tl.store(out_ptr + base + o, _dq_out(c, ps, Hm, G, BLOCK, RFP4, HAD).to(tl.bfloat16), mask=m)
    if EMU > 1:
        jk += _emu_io(emp, 0, (W - 1) * _emu_words(hi - lo, RB, G), EMU, EB)
        tl.store(sink_ptr, jk, mask=jk == 0x5EED1234)
    tl.store(ep_ptr + b, e)



# ------------------------------------------------------------------------------------------------------------- launch
def _carry(har, x):
    c = _CY.get(x.device)
    if c is None:
        c = torch.zeros(har.maxn, dtype=torch.bfloat16, device=x.device)
        _CY[x.device] = c
    return c


def site(x):
    """(role, kind, loadc) of this all-reduce; advances the chain"""
    st = _ST
    n = x.numel()
    if st["role"] is None or x.dim() < 2 or x.shape[-1] != _HID:
        STATS["other"] += 1
        return "main", "mlp", False, False
    p = st["pos"]
    if p == 0:
        st["n"] = n
    if n != st["n"]:   # not part of this chain (should not happen): no carry in or out
        STATS["other"] += 1
        return st["role"], "mlp", False, False
    st["pos"] = p + 1
    STATS["chain"] += 1
    kind = "emb" if p == 0 else ("attn" if p % 2 == 1 else "mlp")
    return st["role"], kind, EF and p > 0, EF


def launch(har, x, out, q, cfg=None, had=None, e8=None):
    """called from HostAllReduce.__call__ (ARC_AR_LB): True if handled (cfg / had: test overrides)"""
    if not (har.q8 and H.TILE >= 8 * 32 * H.WARPS and H.TILE % H.Q8_G == 0 and H.TILE // H.Q8_G >= 16):
        return False
    role, kind, loadc, ef = site(x)
    b, f, rb, rf = cfg or CFG[(role, kind)]
    cy = _carry(har, x) if ef else har.sink
    per = triton.cdiv(triton.cdiv(q, H.NB), H.Q8_G) * H.Q8_G
    har._go(_ar_lb_kernel, (H.NB,))(x, out, har.hin8, har.hred8, har.hred, har.f1, har.f2, har.ep, har.err, har.stk,
                                    cy, q, per, har.maxn, har._emu_buf(), har.sink, RANK=har.rank, W=har.world,
                                    NBLK=H.NB, FS=H.FLAG_STRIDE, BLOCK=H.TILE, G=H.Q8_G, B=b, FP4=f, RB=rb, RFP4=rf,
                                    EF=ef, LOADC=loadc, HAD=HAD if had is None else int(had), E8=E8S if e8 is None else e8, TMO=H.TIMEOUT_NS, LD=H.POLL_LD, EMU=har.emu,
                                    num_warps=H.WARPS, **({'maxnreg': MAXNREG} if MAXNREG else {}))
    return True


# ---------------------------------------------------------------------------------------------------- forward hooks
def _wrap(cls, role):
    orig = cls.forward

    def fwd(self, *a, **kw):
        prev = dict(_ST)
        if CTL and role == "main":
            _reload()
        _ST.update(role=role, pos=0, n=0)
        try:
            return orig(self, *a, **kw)
        finally:
            if LOG and len(_LOGGED) < 12:
                _LOGGED.append((role, _ST["pos"]))
                if len(_LOGGED) in (4, 12):
                    from vllm.logger import init_logger
                    init_logger("vllm.arcstats.ar_lowbit").info("ar_lowbit chain lengths (role, ARs): %s; stats %s",
                                                                _LOGGED, STATS)
            _ST.update(prev)
    fwd.__wrapped__ = orig
    cls.forward = fwd


def install():
    """set host_ar's hook and wrap the main model / MTP predictor forwards (chain reset + role)"""
    from vllm.logger import init_logger
    log = init_logger("vllm.arcstats.ar_lowbit")
    H._LB = launch
    try:
        from vllm.models.dots3_note.nvidia import model as dm, mtp as dmtp
        _wrap(dm.Dots3NoteModel, "main")
        _wrap(dmtp.Dots3NoteMultiTokenPredictor, "draft")
    except Exception as ex:   # no hooks: every all-reduce uses the main widths and no carry
        log.warning("ar_lowbit: forward hooks not installed (%s); EF off", ex)
    log.info("ar_lowbit installed: cfg %s EF %s HAD %s", CFG, EF, HAD)
