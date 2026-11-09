import ctypes, mmap, os
import torch
import triton
import triton.language as tl
NB = int(os.environ.get('ARC_HOST_AR_BLOCKS', '32'))
MAX_BYTES = int(float(os.environ.get('ARC_HOST_AR_BUF_MB', '12')) * (1 << 20))
AR_MAX_BYTES = int(float(os.environ.get('ARC_HOST_AR_MAX_MB', '12')) * (1 << 20))
AR_MIN_BYTES = int(float(os.environ.get('ARC_HOST_AR_MIN_MB', '0')) * (1 << 20))
TILE = int(os.environ.get('ARC_HOST_AR_TILE', '2048'))
WARPS = int(os.environ.get('ARC_HOST_AR_WARPS', '8'))
FLAG_STRIDE = 16
Q8 = os.environ.get('ARC_HOST_AR_Q8') == '1'
Q8_G = int(os.environ.get('ARC_HOST_AR_Q8_G', '128'))
Q8_RED_BF16 = os.environ.get('ARC_HOST_AR_Q8_RED', 'q8') == 'bf16'
PIPE = os.environ.get('ARC_HOST_AR_PIPE') == '1'
PIPE_BLOCKS = int(os.environ.get('ARC_HOST_AR_PIPE_BLOCKS', '32'))
PIPE_CH = int(os.environ.get('ARC_HOST_AR_PIPE_CH', '2048'))
PIPE_WARPS = int(os.environ.get('ARC_HOST_AR_PIPE_WARPS', '4'))
PIPE_NS = int(os.environ.get('ARC_HOST_AR_PIPE_NS', '500'))
PIPE_ROUNDS = int(os.environ.get('ARC_HOST_AR_PIPE_ROUNDS', '2'))
TIMEOUT_NS = int(float(os.environ.get('ARC_HOST_AR_TIMEOUT_MS', '0')) * 1000000.0)
POLL_LD = os.environ.get('ARC_HOST_AR_POLL', 'atomic') == 'ld'
SKEW = int(os.environ.get('ARC_HOST_AR_SKEW', '0'))
PWAIT = os.environ.get('ARC_HOST_AR_PWAIT') == '1'
SKEW_KMAX = 8
BITS = int(os.environ.get('ARC_HOST_AR_BITS', '8'))
RED_BITS = int(os.environ.get('ARC_HOST_AR_RED_BITS', str(BITS)))
CPU = os.environ.get('ARC_HOST_AR_CPU') == '1'
CPU_THREADS = int(os.environ.get('ARC_HOST_AR_CPU_THREADS', '8'))
CPU_K = int(os.environ.get('ARC_HOST_AR_CPU_K', '4'))
CPU_TILE = int(os.environ.get('ARC_HOST_AR_CPU_TILE', '2048'))
CPU_WARPS = int(os.environ.get('ARC_HOST_AR_CPU_WARPS', '8'))
CPU_OWNQ = os.environ.get('ARC_HOST_AR_CPU_OWN', 'bf16') == 'q'
CPU_MIN_BYTES = int(float(os.environ.get('ARC_HOST_AR_CPU_MIN_MB', '0')) * (1 << 20))
CPU_MAX_BYTES = int(float(os.environ.get('ARC_HOST_AR_CPU_MAX_MB', '10.5')) * (1 << 20))
EMU = int(os.environ.get('ARC_LINK_EMU_X', '1'))
EMU_WORDS_N = 1 << 17
EMU_WORDS = tl.constexpr(EMU_WORDS_N)
_LB = None

@triton.jit
def _gtime(x):
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [x], dtype=tl.int64, is_pure=False, pack=1)

@triton.jit
def _poll(fp, LD: tl.constexpr):
    if LD:
        return tl.inline_asm_elementwise('ld.acquire.sys.global.b32 $0, [$1];', '=r,l', [fp], dtype=tl.int32, is_pure=False, pack=1)
    else:
        return tl.atomic_add(fp, 0, sem='acquire', scope='sys')

@triton.jit
def _wait(fp, e, errp, skp, code, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    v = _poll(fp, LD)
    if TMO > 0:
        t0 = _gtime(v)
        dt = t0 - t0
        if tl.load(skp, volatile=True) != 0:
            dt = dt + TMO
        while (v < e) & (dt < TMO):
            v = _poll(fp, LD)
            dt = _gtime(v) - t0
        if v < e:
            tl.atomic_cas(errp, 0, code, sem='relaxed', scope='sys')
            tl.atomic_xchg(skp, code, sem='relaxed', scope='gpu')
    else:
        while v < e:
            v = _poll(fp, LD)

@triton.jit
def _host_ar_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, n, q, per, maxn, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * maxn
    for j in tl.static_range(W):
        if j != RANK:
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = (o < hi) & (base + o < n)
                tl.store(my_in + base + o, tl.load(x_ptr + base + o, mask=m), mask=m)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 110 + k, TMO, LD)
    base = RANK * q
    red = hred_ptr + par.to(tl.int64) * maxn
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        m = (o < hi) & (base + o < n)
        acc = tl.load(x_ptr + base + o, mask=m).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin_ptr + (par * W + k).to(tl.int64) * maxn
                acc += tl.load(pk + base + o, mask=m, cache_modifier='.cv').to(tl.float32)
        r = acc.to(tl.bfloat16)
        tl.store(out_ptr + base + o, r, mask=m)
        tl.store(red + base + o, r, mask=m)
    tl.debug_barrier()
    tl.atomic_xchg(f2_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for j in tl.static_range(W):
        if j != RANK:
            _wait(f2_ptr + (j * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 120 + j, TMO, LD)
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = (o < hi) & (base + o < n)
                tl.store(out_ptr + base + o, tl.load(red + base + o, mask=m, cache_modifier='.cv'), mask=m)
    tl.store(ep_ptr + b, e)

@triton.jit
def _q8(v, G: tl.constexpr, BLOCK: tl.constexpr):
    v2 = tl.reshape(v.to(tl.float32), (BLOCK // G, G))
    sc = tl.maximum(tl.max(tl.abs(v2), axis=1), 1e-30) / 127.0
    r = v2 / sc[:, None]
    qv = tl.where(r >= 0, tl.floor(r + 0.5), tl.ceil(r - 0.5))
    return (tl.reshape(qv, (BLOCK,)).to(tl.int8), sc)

@triton.jit
def _dq8(qv, sc, G: tl.constexpr, BLOCK: tl.constexpr):
    return tl.reshape(tl.reshape(qv.to(tl.float32), (BLOCK // G, G)) * sc[:, None], (BLOCK,))

@triton.jit
def _host_ar_q8_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, hred16_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, n, q, per, maxn, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr, RED_BF16: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False, PW: tl.constexpr=False):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    goffs = tl.arange(0, BLOCK // G)
    slot = 2 * maxn
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * slot
    for j in tl.static_range(W):
        if j != RANK:
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                qv, sc = _q8(tl.load(x_ptr + base + o, mask=m, other=0.0), G, BLOCK)
                tl.store(my_in + base + o, qv, mask=m)
                g = (base + s) // G + goffs
                tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=s + goffs * G < hi)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    peers = tl.arange(0, W)
    if PW:
        _wait_peers(f1_ptr, (peers * NBLK + b) * FS, peers != RANK, e, 0, err_ptr + RANK * FS, stk_ptr, 210, TMO)
    else:
        for k in tl.static_range(W):
            if k != RANK:
                _wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 210 + k, TMO, LD)
    base = RANK * q
    red = hred_ptr + par.to(tl.int64) * slot
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        m = o < hi
        gm = s + goffs * G < hi
        g = (base + s) // G + goffs
        acc = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                pq = tl.load(pk + base + o, mask=m, other=0, cache_modifier='.cv')
                ps = tl.load((pk + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                acc += _dq8(pq, ps, G, BLOCK)
        if RED_BF16:
            r16 = acc.to(tl.bfloat16)
            tl.store(out_ptr + base + o, r16, mask=m)
            tl.store(hred16_ptr + par.to(tl.int64) * maxn + base + o, r16, mask=m)
        else:
            qv, sc = _q8(acc, G, BLOCK)
            tl.store(out_ptr + base + o, _dq8(qv, sc, G, BLOCK).to(tl.bfloat16), mask=m)
            tl.store(red + base + o, qv, mask=m)
            tl.store((red + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=gm)
    tl.debug_barrier()
    tl.atomic_xchg(f2_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    if PW:
        _wait_peers(f2_ptr, (peers * NBLK + b) * FS, peers != RANK, e, 0, err_ptr + RANK * FS, stk_ptr, 220, TMO)
    for j in tl.static_range(W):
        if j != RANK:
            if not PW:
                _wait(f2_ptr + (j * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 220 + j, TMO, LD)
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                if RED_BF16:
                    tl.store(out_ptr + base + o, tl.load(hred16_ptr + par.to(tl.int64) * maxn + base + o, mask=m, cache_modifier='.cv'), mask=m)
                else:
                    g = (base + s) // G + goffs
                    pq = tl.load(red + base + o, mask=m, other=0, cache_modifier='.cv')
                    ps = tl.load((red + maxn).to(tl.pointer_type(tl.float32)) + g, mask=s + goffs * G < hi, other=0.0, cache_modifier='.cv')
                    tl.store(out_ptr + base + o, _dq8(pq, ps, G, BLOCK).to(tl.bfloat16), mask=m)
    tl.store(ep_ptr + b, e)

@triton.jit
def _qb(v, G: tl.constexpr, BLOCK: tl.constexpr, B: tl.constexpr):
    QM: tl.constexpr = (1 << B - 1) - 1
    v2 = tl.reshape(v.to(tl.float32), (BLOCK // G, G))
    sc = tl.maximum(tl.max(tl.abs(v2), axis=1), 1e-30) / QM
    r = v2 / sc[:, None]
    qv = tl.where(r >= 0, tl.floor(r + 0.5), tl.ceil(r - 0.5))
    return (tl.reshape(qv, (BLOCK,)).to(tl.int32) + (QM + 1), sc)

@triton.jit
def _split8(u, BLOCK: tl.constexpr):
    e, o = tl.split(tl.reshape(u, (BLOCK // 8, 2, 2, 2)))
    ee, eo = tl.split(e)
    oe, oo = tl.split(o)
    a, e4 = tl.split(ee)
    c, g6 = tl.split(eo)
    b, f5 = tl.split(oe)
    d, h7 = tl.split(oo)
    return (a, b, c, d, e4, f5, g6, h7)

@triton.jit
def _join8(a, b, c, d, e, f, g, h, BLOCK: tl.constexpr):
    return tl.reshape(tl.join(tl.join(tl.join(a, e), tl.join(c, g)), tl.join(tl.join(b, f), tl.join(d, h))), (BLOCK,))

@triton.jit
def _pack_store(p, w0, u, wm, m, PS, BLOCK: tl.constexpr, B: tl.constexpr):
    if B == 8:
        tl.store(p + w0 * 8 + tl.arange(0, BLOCK), (u - 128).to(tl.int8), mask=m)
    else:
        a, b, c, d, e, f, g, h = _split8(u, BLOCK)
        lo = a & 15 | (b & 15) << 4 | (c & 15) << 8 | (d & 15) << 12 | (e & 15) << 16 | (f & 15) << 20 | (g & 15) << 24 | (h & 15) << 28
        wi = w0 + tl.arange(0, BLOCK // 8)
        tl.store(p.to(tl.pointer_type(tl.int32)) + wi, lo, mask=wm)
        HB: tl.constexpr = B - 4
        hi = a >> 4 | b >> 4 << HB | c >> 4 << 2 * HB | d >> 4 << 3 * HB | e >> 4 << 4 * HB | f >> 4 << 5 * HB | g >> 4 << 6 * HB | h >> 4 << 7 * HB
        if HB == 1:
            tl.store(p + PS + wi, hi.to(tl.int8), mask=wm)
        else:
            tl.store((p + PS).to(tl.pointer_type(tl.int16)) + wi, hi.to(tl.int16), mask=wm)

@triton.jit
def _unpack_load(p, w0, wm, m, PS, BLOCK: tl.constexpr, B: tl.constexpr):
    if B == 8:
        return tl.load(p + w0 * 8 + tl.arange(0, BLOCK), mask=m, other=0, cache_modifier='.cv').to(tl.int32)
    else:
        wi = w0 + tl.arange(0, BLOCK // 8)
        lo = tl.load(p.to(tl.pointer_type(tl.int32)) + wi, mask=wm, other=0, cache_modifier='.cv')
        HB: tl.constexpr = B - 4
        if HB == 1:
            hi = tl.load(p + PS + wi, mask=wm, other=0, cache_modifier='.cv').to(tl.int32) & 255
        else:
            hi = tl.load((p + PS).to(tl.pointer_type(tl.int16)) + wi, mask=wm, other=0, cache_modifier='.cv').to(tl.int32) & 65535
        HM: tl.constexpr = (1 << HB) - 1
        O: tl.constexpr = 1 << B - 1
        a = (lo & 15 | (hi & HM) << 4) - O
        b = (lo >> 4 & 15 | (hi >> HB & HM) << 4) - O
        c = (lo >> 8 & 15 | (hi >> 2 * HB & HM) << 4) - O
        d = (lo >> 12 & 15 | (hi >> 3 * HB & HM) << 4) - O
        e = (lo >> 16 & 15 | (hi >> 4 * HB & HM) << 4) - O
        f = (lo >> 20 & 15 | (hi >> 5 * HB & HM) << 4) - O
        g = (lo >> 24 & 15 | (hi >> 6 * HB & HM) << 4) - O
        h = (lo >> 28 & 15 | (hi >> 7 * HB & HM) << 4) - O
        return _join8(a, b, c, d, e, f, g, h, BLOCK)

@triton.jit
def _dqb(c, sc, G: tl.constexpr, BLOCK: tl.constexpr):
    return tl.reshape(tl.reshape(c.to(tl.float32), (BLOCK // G, G)) * sc[:, None], (BLOCK,))

@triton.jit
def _emu_io(ep, n_out, n_in, EMU: tl.constexpr, EB: tl.constexpr):
    offs = tl.arange(0, EB)
    junk = tl.zeros([EB], dtype=tl.int32)
    for r in tl.static_range(EMU - 1):
        for s in range(0, n_out, EB):
            tl.store(ep + s % EMU_WORDS + offs, offs, mask=s + offs < n_out)
        for s in range(0, n_in, EB):
            junk += tl.load(ep + s % EMU_WORDS + offs, mask=s + offs < n_in, other=0, cache_modifier='.cv')
    return tl.sum(junk)

@triton.jit
def _emu_words(n_el, B: tl.constexpr, G: tl.constexpr):
    if B == 16:
        return tl.maximum(n_el, 0) // 2
    else:
        return tl.maximum(n_el, 0) * (B * G // 8 + 4) // (4 * G)

@triton.jit
def _vpack_store(p, e0, u, m, gm, maxn, BLOCK: tl.constexpr, B: tl.constexpr):
    if B == 8:
        tl.store(p + e0 + tl.arange(0, BLOCK), (u - 128).to(tl.int8), mask=m)
    else:
        u3 = tl.reshape(u, (BLOCK // 128, 8, 16))
        k3 = tl.arange(0, 8)[None, :, None]
        HB: tl.constexpr = B - 4
        lo = tl.reshape(tl.sum((u3 & 15) << 4 * k3, axis=1), (BLOCK // 8,))
        hi = tl.reshape(tl.sum((u3 >> 4 & (1 << HB) - 1) << HB * k3, axis=1), (BLOCK // 8,))
        wi = e0 // 8 + tl.arange(0, BLOCK // 8)
        tl.store(p.to(tl.pointer_type(tl.int32)) + wi, lo, mask=gm)
        if HB == 1:
            tl.store(p + maxn // 2 + wi, hi.to(tl.int8), mask=gm)
        else:
            tl.store((p + maxn // 2).to(tl.pointer_type(tl.int16)) + wi, hi.to(tl.int16), mask=gm)

@triton.jit
def _vunpack_load(p, e0, m, gm, maxn, BLOCK: tl.constexpr, B: tl.constexpr):
    if B == 8:
        return tl.load(p + e0 + tl.arange(0, BLOCK), mask=m, other=0, cache_modifier='.cv').to(tl.int32)
    else:
        HB: tl.constexpr = B - 4
        wi = e0 // 8 + tl.arange(0, BLOCK // 8)
        lo = tl.load(p.to(tl.pointer_type(tl.int32)) + wi, mask=gm, other=0, cache_modifier='.cv')
        if HB == 1:
            hi = tl.load(p + maxn // 2 + wi, mask=gm, other=0, cache_modifier='.cv').to(tl.int32) & 255
        else:
            hi = tl.load((p + maxn // 2).to(tl.pointer_type(tl.int16)) + wi, mask=gm, other=0, cache_modifier='.cv').to(tl.int32) & 65535
        k3 = tl.arange(0, 8)[None, :, None]
        lo3 = tl.reshape(lo, (BLOCK // 128, 1, 16))
        hi3 = tl.reshape(hi, (BLOCK // 128, 1, 16))
        c3 = lo3 >> 4 * k3 & 15 | (hi3 >> HB * k3 & (1 << HB) - 1) << 4
        return tl.reshape(c3, (BLOCK,)) - (1 << B - 1)

@triton.jit
def _host_arc_kernel(x_ptr, out_ptr, hin_ptr, own_ptr, hred_ptr, gf_ptr, cf_ptr, ep_ptr, err_ptr, stk_ptr, emu_ptr, sink_ptr, q, per, sub, maxn, own_el, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, K: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr, B: tl.constexpr, RB: tl.constexpr, OWNQ: tl.constexpr, TMO: tl.constexpr=0, EMU: tl.constexpr=1, EB: tl.constexpr=4096):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    goffs = tl.arange(0, BLOCK // G)
    wg = tl.arange(0, BLOCK // 8) // 16 * 128
    slot = 2 * maxn
    my_in = hin_ptr + RANK * slot.to(tl.int64)
    my_own = own_ptr + RANK * own_el.to(tl.int64)
    errp = err_ptr + RANK * FS
    emp = emu_ptr + b * EMU_WORDS
    jk = 0
    for it in tl.static_range(K + 1):
        if it < K:
            c_lo = lo + it * sub
            c_hi = tl.minimum(c_lo + sub, hi)
            for j in tl.static_range(W):
                base = j * q
                if j != RANK or OWNQ:
                    for s in range(c_lo, c_hi, BLOCK):
                        o = s + offs0
                        m = o < c_hi
                        u, sc = _qb(tl.load(x_ptr + base + o, mask=m, other=0.0), G, BLOCK, B)
                        _vpack_store(my_in, base + s, u, m, s + wg < c_hi, maxn, BLOCK, B)
                        tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + (base + s) // G + goffs, sc, mask=s + goffs * G < c_hi)
                else:
                    for s in range(c_lo, c_hi, BLOCK):
                        o = s + offs0
                        m = o < c_hi
                        tl.store(my_own + o, tl.load(x_ptr + base + o, mask=m), mask=m)
            if EMU > 1:
                nb = c_hi - c_lo
                if OWNQ:
                    wo = W * _emu_words(nb, B, G)
                else:
                    wo = (W - 1) * _emu_words(nb, B, G) + _emu_words(nb, 16, G)
                jk += _emu_io(emp, wo, 0, EMU, EB)
        if it >= 1:
            c_lo = lo + (it - 1) * sub
            c_hi = tl.minimum(c_lo + sub, hi)
            _wait(cf_ptr + (b * K + it - 1) * FS, e, errp, stk_ptr, 710, TMO, False)
            for j in tl.static_range(W):
                base = j * q
                for s in range(c_lo, c_hi, BLOCK):
                    o = s + offs0
                    m = o < c_hi
                    c = _vunpack_load(hred_ptr, base + s, m, s + wg < c_hi, maxn, BLOCK, RB)
                    ps = tl.load((hred_ptr + maxn).to(tl.pointer_type(tl.float32)) + (base + s) // G + goffs, mask=s + goffs * G < c_hi, other=0.0, cache_modifier='.cv')
                    tl.store(out_ptr + base + o, _dqb(c, ps, G, BLOCK).to(tl.bfloat16), mask=m)
            if EMU > 1:
                jk += _emu_io(emp, 0, W * _emu_words(c_hi - c_lo, RB, G), EMU, EB)
        if it < K:
            fl = gf_ptr + ((RANK * NBLK + b) * K + it) * FS
            tl.store(fl + 1, q)
            tl.store(fl + 2, B | RB << 8 | OWNQ << 16)
            tl.debug_barrier()
            tl.atomic_xchg(fl, e, sem='release', scope='sys')
    if EMU > 1:
        tl.store(sink_ptr, jk, mask=jk == 1592594996)
    tl.store(ep_ptr + b, e)

@triton.jit
def _host_ar_qb_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, hred16_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, q, per, maxn, emu_ptr, sink_ptr, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr, B: tl.constexpr, RB: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False, EMU: tl.constexpr=1, EB: tl.constexpr=4096):
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
    emp = emu_ptr + b * EMU_WORDS
    jk = 0
    for j in tl.static_range(W):
        if j != RANK:
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                u, sc = _qb(tl.load(x_ptr + base + o, mask=m, other=0.0), G, BLOCK, B)
                _pack_store(my_in, (base + s) // 8, u, s + w8 < hi, m, PS, BLOCK, B)
                g = (base + s) // G + goffs
                tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=s + goffs * G < hi)
    if EMU > 1:
        jk += _emu_io(emp, (W - 1) * _emu_words(hi - lo, B, G), 0, EMU, EB)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 610 + k, TMO, LD)
    base = RANK * q
    red = hred_ptr + par.to(tl.int64) * slot
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        m = o < hi
        gm = s + goffs * G < hi
        g = (base + s) // G + goffs
        acc = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
        for k in tl.static_range(W):
            if k != RANK:
                pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                c = _unpack_load(pk, (base + s) // 8, s + w8 < hi, m, PS, BLOCK, B)
                ps = tl.load((pk + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                acc += _dqb(c, ps, G, BLOCK)
        if RB == 16:
            r16 = acc.to(tl.bfloat16)
            tl.store(out_ptr + base + o, r16, mask=m)
            tl.store(hred16_ptr + par.to(tl.int64) * maxn + base + o, r16, mask=m)
        else:
            u, sc = _qb(acc, G, BLOCK, RB)
            tl.store(out_ptr + base + o, _dqb(u - (1 << RB - 1), sc, G, BLOCK).to(tl.bfloat16), mask=m)
            _pack_store(red, (base + s) // 8, u, s + w8 < hi, m, PS, BLOCK, RB)
            tl.store((red + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=gm)
    if EMU > 1:
        jk += _emu_io(emp, _emu_words(hi - lo, RB, G), (W - 1) * _emu_words(hi - lo, B, G), EMU, EB)
    tl.debug_barrier()
    tl.atomic_xchg(f2_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for j in tl.static_range(W):
        if j != RANK:
            _wait(f2_ptr + (j * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 620 + j, TMO, LD)
            base = j * q
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                m = o < hi
                if RB == 16:
                    tl.store(out_ptr + base + o, tl.load(hred16_ptr + par.to(tl.int64) * maxn + base + o, mask=m, cache_modifier='.cv'), mask=m)
                else:
                    g = (base + s) // G + goffs
                    c = _unpack_load(red, (base + s) // 8, s + w8 < hi, m, PS, BLOCK, RB)
                    ps = tl.load((red + maxn).to(tl.pointer_type(tl.float32)) + g, mask=s + goffs * G < hi, other=0.0, cache_modifier='.cv')
                    tl.store(out_ptr + base + o, _dqb(c, ps, G, BLOCK).to(tl.bfloat16), mask=m)
    if EMU > 1:
        jk += _emu_io(emp, 0, (W - 1) * _emu_words(hi - lo, RB, G), EMU, EB)
        tl.store(sink_ptr, jk, mask=jk == 1592594996)
    tl.store(ep_ptr + b, e)

@triton.jit
def _host_ar_q8s_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, q, per, sub, maxn, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr, K: tl.constexpr, KMAX: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False, PW: tl.constexpr=False):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, q)
    offs0 = tl.arange(0, BLOCK)
    goffs = tl.arange(0, BLOCK // G)
    slot = 2 * maxn
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * slot
    red = hred_ptr + par.to(tl.int64) * slot
    errp = err_ptr + RANK * FS
    peers = tl.arange(0, W)
    for it in tl.static_range(K + 2):
        if it < K:
            c_lo = lo + it * sub
            c_hi = tl.minimum(c_lo + sub, hi)
            for j in tl.static_range(W):
                if j != RANK:
                    base = j * q
                    for s in range(c_lo, c_hi, BLOCK):
                        o = s + offs0
                        m = o < c_hi
                        qv, sc = _q8(tl.load(x_ptr + base + o, mask=m, other=0.0), G, BLOCK)
                        tl.store(my_in + base + o, qv, mask=m)
                        g = (base + s) // G + goffs
                        tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=s + goffs * G < c_hi)
        if it >= 1:
            if it <= K:
                c_lo = lo + (it - 1) * sub
                c_hi = tl.minimum(c_lo + sub, hi)
                if PW:
                    _wait_peers(f1_ptr, ((peers * NBLK + b) * KMAX + it - 1) * FS, peers != RANK, e, 0, errp, stk_ptr, 510, TMO)
                else:
                    for k in tl.static_range(W):
                        if k != RANK:
                            _wait(f1_ptr + ((k * NBLK + b) * KMAX + it - 1) * FS, e, errp, stk_ptr, 510 + k, TMO, LD)
                base = RANK * q
                for s in range(c_lo, c_hi, BLOCK):
                    o = s + offs0
                    m = o < c_hi
                    gm = s + goffs * G < c_hi
                    g = (base + s) // G + goffs
                    acc = tl.load(x_ptr + base + o, mask=m, other=0.0).to(tl.float32)
                    for k in tl.static_range(W):
                        if k != RANK:
                            pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                            pq = tl.load(pk + base + o, mask=m, other=0, cache_modifier='.cv')
                            ps = tl.load((pk + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                            acc += _dq8(pq, ps, G, BLOCK)
                    qv, sc = _q8(acc, G, BLOCK)
                    tl.store(out_ptr + base + o, _dq8(qv, sc, G, BLOCK).to(tl.bfloat16), mask=m)
                    tl.store(red + base + o, qv, mask=m)
                    tl.store((red + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=gm)
        if it <= K:
            tl.debug_barrier()
            fl = (RANK * NBLK + b) * KMAX
            if it >= 1:
                tl.atomic_xchg(f2_ptr + (fl + it - 1) * FS, e, sem='release', scope='sys')
                if it < K:
                    tl.atomic_xchg(f1_ptr + (fl + it) * FS, e, sem='relaxed', scope='sys')
            else:
                tl.atomic_xchg(f1_ptr + (fl + it) * FS, e, sem='release', scope='sys')
        if it >= 2:
            c_lo = lo + (it - 2) * sub
            c_hi = tl.minimum(c_lo + sub, hi)
            if PW:
                _wait_peers(f2_ptr, ((peers * NBLK + b) * KMAX + it - 2) * FS, peers != RANK, e, 0, errp, stk_ptr, 520, TMO)
            for j in tl.static_range(W):
                if j != RANK:
                    if not PW:
                        _wait(f2_ptr + ((j * NBLK + b) * KMAX + it - 2) * FS, e, errp, stk_ptr, 520 + j, TMO, LD)
                    base = j * q
                    for s in range(c_lo, c_hi, BLOCK):
                        o = s + offs0
                        m = o < c_hi
                        g = (base + s) // G + goffs
                        pq = tl.load(red + base + o, mask=m, other=0, cache_modifier='.cv')
                        ps = tl.load((red + maxn).to(tl.pointer_type(tl.float32)) + g, mask=s + goffs * G < c_hi, other=0.0, cache_modifier='.cv')
                        tl.store(out_ptr + base + o, _dq8(pq, ps, G, BLOCK).to(tl.bfloat16), mask=m)
    tl.store(ep_ptr + b, e)

@triton.jit
def _nap(x, NS: tl.constexpr):
    return tl.inline_asm_elementwise(f'nanosleep.u32 {NS}; mov.u32 $0, $1;', '=r,r', [x], dtype=tl.int32, is_pure=False, pack=1)

@triton.jit
def _wait_peers(fp, offs, msk, e, NS: tl.constexpr, errp, skp, code, TMO: tl.constexpr=0):
    lo = tl.min(tl.where(msk, tl.atomic_add(fp + offs, 0, mask=msk, sem='acquire', scope='sys'), e), 0)
    if TMO > 0:
        t0 = _gtime(lo)
        dt = t0 - t0
        if tl.load(skp, volatile=True) != 0:
            dt = dt + TMO
        while (lo < e) & (dt < TMO):
            if NS > 0:
                lo = _nap(lo, NS)
            lo = tl.min(tl.where(msk, tl.atomic_add(fp + offs, 0, mask=msk, sem='acquire', scope='sys'), e), 0)
            dt = _gtime(lo) - t0
        if lo < e:
            tl.atomic_cas(errp, 0, code, sem='relaxed', scope='sys')
            tl.atomic_xchg(skp, code, sem='relaxed', scope='gpu')
    else:
        while lo < e:
            if NS > 0:
                lo = _nap(lo, NS)
            lo = tl.min(tl.where(msk, tl.atomic_add(fp + offs, 0, mask=msk, sem='acquire', scope='sys'), e), 0)

@triton.jit
def _wait_each(fp, stride, off, e, RANK: tl.constexpr, W: tl.constexpr, NS: tl.constexpr, errp, skp, code, TMO: tl.constexpr=0):
    for k in tl.static_range(W):
        if k != RANK:
            if TMO > 0:
                _wait(fp + k * stride + off, e, errp, skp, code + k, TMO, False)
            else:
                v = tl.atomic_add(fp + k * stride + off, 0, sem='acquire', scope='sys')
                while v < e:
                    if NS > 0:
                        v = _nap(v, NS)
                    v = tl.atomic_add(fp + k * stride + off, 0, sem='acquire', scope='sys')

@triton.jit
def _host_ar_qbp_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, emu_ptr, sink_ptr, q, maxn, nc, L, ncmax, RANK: tl.constexpr, W: tl.constexpr, NBP: tl.constexpr, FS: tl.constexpr, CH: tl.constexpr, G: tl.constexpr, B: tl.constexpr, RB: tl.constexpr, NS: tl.constexpr, TMO: tl.constexpr=0, EMU: tl.constexpr=1, EB: tl.constexpr=4096):
    pid = tl.program_id(0)
    role = pid // NBP
    b = pid % NBP
    e = tl.load(ep_ptr + pid) + 1
    par = e % 2
    slot = 2 * maxn
    PS = maxn // 2
    offs = tl.arange(0, CH)
    goffs = tl.arange(0, CH // G)
    w8 = tl.arange(0, CH // 8) * 8
    emp = emu_ptr + pid * EMU_WORDS
    jk = 0
    if role == 0:
        my_in = hin_ptr + (par * W + RANK).to(tl.int64) * slot
        for c in range(b, nc, NBP):
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                for jj in tl.static_range(1, W):
                    base = (RANK + jj) % W * q
                    u, sc = _qb(tl.load(x_ptr + base + s + offs, mask=m, other=0.0), G, CH, B)
                    _pack_store(my_in, (base + s) // 8, u, s + w8 < q, m, PS, CH, B)
                    tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + (base + s) // G + goffs, sc, mask=gm)
            if EMU > 1:
                jk += _emu_io(emp, (W - 1) * _emu_words(tl.minimum(c * L + L, q) - c * L, B, G), 0, EMU, EB)
            tl.debug_barrier()
            tl.atomic_xchg(f1_ptr + (RANK * ncmax + c) * FS, e, sem='release', scope='sys')
    elif role == 1:
        base = RANK * q
        red = hred_ptr + par.to(tl.int64) * slot
        for c in range(b, nc, NBP):
            _wait_each(f1_ptr, ncmax * FS, c * FS, e, RANK, W, NS, err_ptr + RANK * FS, stk_ptr, 810, TMO)
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                g = (base + s) // G + goffs
                acc = tl.load(x_ptr + base + s + offs, mask=m, other=0.0).to(tl.float32)
                for k in tl.static_range(W):
                    if k != RANK:
                        pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                        cc = _unpack_load(pk, (base + s) // 8, s + w8 < q, m, PS, CH, B)
                        ps = tl.load((pk + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                        acc += _dqb(cc, ps, G, CH)
                u, sc = _qb(acc, G, CH, RB)
                tl.store(out_ptr + base + s + offs, _dqb(u - (1 << RB - 1), sc, G, CH).to(tl.bfloat16), mask=m)
                _pack_store(red, (base + s) // 8, u, s + w8 < q, m, PS, CH, RB)
                tl.store((red + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=gm)
            if EMU > 1:
                nl = tl.minimum(c * L + L, q) - c * L
                jk += _emu_io(emp, _emu_words(nl, RB, G), (W - 1) * _emu_words(nl, B, G), EMU, EB)
            tl.debug_barrier()
            tl.atomic_xchg(f2_ptr + (RANK * ncmax + c) * FS, e, sem='release', scope='sys')
    else:
        red = hred_ptr + par.to(tl.int64) * slot
        for c in range(b, nc, NBP):
            _wait_each(f2_ptr, ncmax * FS, c * FS, e, RANK, W, NS, err_ptr + RANK * FS, stk_ptr, 820, TMO)
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                for jj in tl.static_range(1, W):
                    base = (RANK + jj) % W * q
                    cc = _unpack_load(red, (base + s) // 8, s + w8 < q, m, PS, CH, RB)
                    ps = tl.load((red + maxn).to(tl.pointer_type(tl.float32)) + (base + s) // G + goffs, mask=gm, other=0.0, cache_modifier='.cv')
                    tl.store(out_ptr + base + s + offs, _dqb(cc, ps, G, CH).to(tl.bfloat16), mask=m)
            if EMU > 1:
                jk += _emu_io(emp, 0, (W - 1) * _emu_words(tl.minimum(c * L + L, q) - c * L, RB, G), EMU, EB)
    if EMU > 1:
        tl.store(sink_ptr, jk, mask=jk == 1592594996)
    tl.store(ep_ptr + pid, e)

@triton.jit
def _host_ar_q8p_kernel(x_ptr, out_ptr, hin_ptr, hred_ptr, f1_ptr, f2_ptr, ep_ptr, err_ptr, stk_ptr, q, maxn, nc, L, ncmax, RANK: tl.constexpr, W: tl.constexpr, NBP: tl.constexpr, FS: tl.constexpr, CH: tl.constexpr, G: tl.constexpr, NS: tl.constexpr, TMO: tl.constexpr=0):
    pid = tl.program_id(0)
    role = pid // NBP
    b = pid % NBP
    e = tl.load(ep_ptr + pid) + 1
    par = e % 2
    slot = 2 * maxn
    offs = tl.arange(0, CH)
    goffs = tl.arange(0, CH // G)
    peers = tl.arange(0, W)
    pm = peers != RANK
    if role == 0:
        my_in = hin_ptr + (par * W + RANK).to(tl.int64) * slot
        for c in range(b, nc, NBP):
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                for jj in tl.static_range(1, W):
                    base = (RANK + jj) % W * q
                    qv, sc = _q8(tl.load(x_ptr + base + s + offs, mask=m, other=0.0), G, CH)
                    tl.store(my_in + base + s + offs, qv, mask=m)
                    tl.store((my_in + maxn).to(tl.pointer_type(tl.float32)) + (base + s) // G + goffs, sc, mask=gm)
            tl.debug_barrier()
            tl.atomic_xchg(f1_ptr + (RANK * ncmax + c) * FS, e, sem='release', scope='sys')
    elif role == 1:
        base = RANK * q
        red = hred_ptr + par.to(tl.int64) * slot
        for c in range(b, nc, NBP):
            _wait_each(f1_ptr, ncmax * FS, c * FS, e, RANK, W, NS, err_ptr + RANK * FS, stk_ptr, 410, TMO)
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                g = (base + s) // G + goffs
                acc = tl.load(x_ptr + base + s + offs, mask=m, other=0.0).to(tl.float32)
                for k in tl.static_range(W):
                    if k != RANK:
                        pk = hin_ptr + (par * W + k).to(tl.int64) * slot
                        pq = tl.load(pk + base + s + offs, mask=m, other=0, cache_modifier='.cv')
                        ps = tl.load((pk + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                        acc += _dq8(pq, ps, G, CH)
                qv, sc = _q8(acc, G, CH)
                tl.store(out_ptr + base + s + offs, _dq8(qv, sc, G, CH).to(tl.bfloat16), mask=m)
                tl.store(red + base + s + offs, qv, mask=m)
                tl.store((red + maxn).to(tl.pointer_type(tl.float32)) + g, sc, mask=gm)
            tl.debug_barrier()
            tl.atomic_xchg(f2_ptr + (RANK * ncmax + c) * FS, e, sem='release', scope='sys')
    else:
        red = hred_ptr + par.to(tl.int64) * slot
        for c in range(b, nc, NBP):
            _wait_each(f2_ptr, ncmax * FS, c * FS, e, RANK, W, NS, err_ptr + RANK * FS, stk_ptr, 420, TMO)
            for s in range(c * L, tl.minimum(c * L + L, q), CH):
                m = s + offs < q
                gm = s + goffs * G < q
                for jj in tl.static_range(1, W):
                    base = (RANK + jj) % W * q
                    g = (base + s) // G + goffs
                    pq = tl.load(red + base + s + offs, mask=m, other=0, cache_modifier='.cv')
                    ps = tl.load((red + maxn).to(tl.pointer_type(tl.float32)) + g, mask=gm, other=0.0, cache_modifier='.cv')
                    tl.store(out_ptr + base + s + offs, _dq8(pq, ps, G, CH).to(tl.bfloat16), mask=m)
    tl.store(ep_ptr + pid, e)

@triton.jit
def _host_ag_kernel(x_ptr, out_ptr, hin_ptr, f1_ptr, ep_ptr, err_ptr, stk_ptr, m, per, maxn, RANK: tl.constexpr, W: tl.constexpr, NBLK: tl.constexpr, FS: tl.constexpr, BLOCK: tl.constexpr, TMO: tl.constexpr=0, LD: tl.constexpr=False):
    b = tl.program_id(0)
    e = tl.load(ep_ptr + b) + 1
    par = e % 2
    lo = b * per
    hi = tl.minimum(lo + per, m)
    offs0 = tl.arange(0, BLOCK)
    my_in = hin_ptr + (par * W + RANK).to(tl.int64) * maxn
    for s in range(lo, hi, BLOCK):
        o = s + offs0
        msk = o < hi
        v = tl.load(x_ptr + o, mask=msk)
        tl.store(my_in + o, v, mask=msk)
        tl.store(out_ptr + RANK * m + o, v, mask=msk)
    tl.debug_barrier()
    tl.atomic_xchg(f1_ptr + (RANK * NBLK + b) * FS, e, sem='release', scope='sys')
    for k in tl.static_range(W):
        if k != RANK:
            _wait(f1_ptr + (k * NBLK + b) * FS, e, err_ptr + RANK * FS, stk_ptr, 310 + k, TMO, LD)
            pk = hin_ptr + (par * W + k).to(tl.int64) * maxn
            for s in range(lo, hi, BLOCK):
                o = s + offs0
                msk = o < hi
                tl.store(out_ptr + k * m + o, tl.load(pk + o, mask=msk, cache_modifier='.cv'), mask=msk)
    tl.store(ep_ptr + b, e)
_INSTANCES = {}
_TRACE = os.environ.get('ARC_HOST_AR_TRACE')

def _trace(self, op, n):
    if _TRACE:
        if not hasattr(self, '_tf'):
            self._tf = open(f'{_TRACE}.{self.rank}', 'w', buffering=1)
            self._ops = 0
        cap = torch.cuda.is_current_stream_capturing()
        if not cap:
            self._ops += 1
        import traceback
        where = '|'.join((f'{f.name}' for f in traceback.extract_stack(limit=8)[:-2]))
        self._tf.write(f'{self._ops} {op} {n} cap={int(cap)} {where}\n')

def get(group='tp'):
    return _INSTANCES.get(group)

class _Ptr:

    def __init__(self, addr, dtype):
        self.addr, self.dtype = (addr, dtype)

    def data_ptr(self):
        return self.addr

def _cudart():
    torch.cuda.init()
    for so in ('libcudart.so.13', 'libcudart.so.12', 'libcudart.so'):
        try:
            return ctypes.CDLL(so)
        except OSError:
            pass
    raise OSError('libcudart not found')

def _gpu_node_cpus():
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        bdf = f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0'
        node = int(open(f'/sys/bus/pci/devices/{bdf}/numa_node').read())
        if node < 0:
            return set()
        cpus = set()
        for part in open(f'/sys/devices/system/node/node{node}/cpulist').read().strip().split(','):
            a, _, b = part.partition('-')
            cpus.update(range(int(a), int(b or a) + 1))
        return cpus & os.sched_getaffinity(0)
    except Exception:
        return set()

def _first_touch_local(buf, size):
    old = os.sched_getaffinity(0)
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        bdf = f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0'
        node = int(open(f'/sys/bus/pci/devices/{bdf}/numa_node').read())
        if node >= 0:
            cpus = set()
            for part in open(f'/sys/devices/system/node/node{node}/cpulist').read().strip().split(','):
                a, _, b = part.partition('-')
                cpus.update(range(int(a), int(b or a) + 1))
            os.sched_setaffinity(0, cpus & set(range(os.cpu_count())) or old)
    except Exception:
        pass
    ctypes.memset(buf, 0, size)
    os.sched_setaffinity(0, old)

class HostAllReduce:

    def __init__(self, rank, world, name, cpu_group=None, max_bytes=MAX_BYTES):
        self.rank, self.world = (rank, world)
        from .link_gate import word as _gw
        _gw(torch.cuda.current_device())
        self.maxn = max_bytes // 2
        n_flags = 2 * world * NB * FLAG_STRIDE
        self.size = 2 * (world + 1) * self.maxn * 2 + n_flags * 4
        self.ncmax = triton.cdiv(self.maxn // world, 512)
        base_size = self.size
        if PIPE or os.environ.get('ARC_HOST_AR_PIPE_ALLOC') == '1':
            self.size += 2 * (world + 1) * self.maxn * 2 + 2 * world * self.ncmax * FLAG_STRIDE * 4
        tail = self.size
        self.size += 4096
        skew_flags = 2 * world * NB * SKEW_KMAX * FLAG_STRIDE * 4
        if SKEW > 0 or os.environ.get('ARC_HOST_AR_SKEW_ALLOC') == '1':
            self.size += skew_flags
        path = f'/dev/shm/arc_host_ar_{name}'
        if rank == 0:
            with open(path, 'wb') as f:
                f.truncate(self.size)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
        fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
            if rank == 0 and os.environ.get('ARC_HOST_AR_KEEP') != '1':
                os.unlink(path)
        host = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if rank == 0:
            _first_touch_local(host, self.size)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
        rt = _cudart()
        assert rt.cudaHostRegister(ctypes.c_void_p(host), ctypes.c_size_t(self.size), ctypes.c_uint(1 | 2)) == 0
        dptr = ctypes.c_void_p()
        assert rt.cudaHostGetDevicePointer(ctypes.byref(dptr), ctypes.c_void_p(host), ctypes.c_uint(0)) == 0
        d = dptr.value
        self.hin = _Ptr(d, torch.bfloat16)
        self.hin32 = _Ptr(d, torch.int32)
        self.hred = _Ptr(d + 2 * world * self.maxn * 2, torch.bfloat16)
        f = d + 2 * (world + 1) * self.maxn * 2
        self.f1 = _Ptr(f, torch.int32)
        self.f2 = _Ptr(f + world * NB * FLAG_STRIDE * 4, torch.int32)
        self.ep = torch.zeros(NB, dtype=torch.int32, device='cuda')
        self._ev, self._ev_stream = (None, None)
        self.q8 = Q8
        self.bits, self.red_bits = (BITS, RED_BITS)
        self.hin8 = _Ptr(d, torch.int8)
        self.hred8 = _Ptr(d + 2 * world * self.maxn * 2, torch.int8)
        self.pipe = None
        if self.size > base_size:
            pd = d + base_size
            pf = pd + 2 * (world + 1) * self.maxn * 2
            self.pipe = dict(hin=_Ptr(pd, torch.int8), hred=_Ptr(pd + 2 * world * self.maxn * 2, torch.int8), f1=_Ptr(pf, torch.int32), f2=_Ptr(pf + world * self.ncmax * FLAG_STRIDE * 4, torch.int32), ep=torch.zeros(3 * PIPE_BLOCKS, dtype=torch.int32, device='cuda'))
        self.use_pipe = PIPE and self.pipe is not None and (world in (2, 4, 8)) and (PIPE_CH % Q8_G == 0) and (PIPE_CH >= 512)
        self.err = _Ptr(d + tail, torch.int32)
        self.stk = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.warm = False
        self._err_host = (ctypes.c_int32 * (4096 // 4)).from_address(host + tail)
        self.skew = None
        if self.size >= tail + 4096 + skew_flags:
            self.skew = dict(f1=_Ptr(d + tail + 4096, torch.int32), f2=_Ptr(d + tail + 4096 + skew_flags // 2, torch.int32))
        self.use_skew = SKEW if self.skew is not None and 2 <= SKEW <= SKEW_KMAX else 0
        self.emu, self._emu_t = (EMU, None)
        self.sink = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.cpu = None
        if CPU or os.environ.get('ARC_HOST_AR_CPU_ALLOC') == '1':
            self._cpu_init(name, cpu_group)
        self.cpu_on = CPU and self.cpu is not None
        self.cpu_ownq = CPU_OWNQ
        if TIMEOUT_NS > 0 and os.environ.get('ARC_HOST_AR_MONITOR', '1') == '1' and (cpu_group is not None):
            import threading
            threading.Thread(target=self._monitor, daemon=True, name='host_ar_monitor').start()

    def _emu_buf(self):
        if self.emu <= 1:
            return self.sink
        if self._emu_t is None:
            self._emu_t = torch.zeros(max(NB, 3 * PIPE_BLOCKS) * EMU_WORDS_N, dtype=torch.int32, pin_memory=True)
        return _Ptr(self._emu_t.data_ptr(), torch.int32)

    def _cpu_init(self, name, cpu_group):
        W, K, FS = (self.world, CPU_K, FLAG_STRIDE)
        maxn = triton.cdiv(CPU_MAX_BYTES // 2, W * Q8_G) * W * Q8_G
        gf_off, cf_off = (0, W * NB * K * FS * 4)
        in_off = triton.cdiv(cf_off + NB * K * FS * 4, 4096) * 4096
        slot = 2 * maxn
        own_slot = triton.cdiv(2 * (maxn // W), 4096) * 4096
        own_off = in_off + W * slot
        red_off = own_off + W * own_slot
        size = red_off + slot
        path = f'/dev/shm/arc_host_arc_{name}'
        if self.rank == 0:
            with open(path, 'wb') as f:
                f.truncate(size)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
        fd = os.open(path, os.O_RDWR)
        self.cmm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
            if self.rank == 0 and os.environ.get('ARC_HOST_AR_KEEP') != '1':
                os.unlink(path)
        host = ctypes.addressof(ctypes.c_char.from_buffer(self.cmm))
        if self.rank == 0:
            _first_touch_local(host, size)
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
        rt = _cudart()
        assert rt.cudaHostRegister(ctypes.c_void_p(host), ctypes.c_size_t(size), ctypes.c_uint(1 | 2)) == 0
        dptr = ctypes.c_void_p()
        assert rt.cudaHostGetDevicePointer(ctypes.byref(dptr), ctypes.c_void_p(host), ctypes.c_uint(0)) == 0
        d = dptr.value
        c = dict(maxn=maxn, K=K, gf=_Ptr(d + gf_off, torch.int32), cf=_Ptr(d + cf_off, torch.int32), hin=_Ptr(d + in_off, torch.int8), own=_Ptr(d + own_off, torch.bfloat16), own_el=own_slot // 2, red=_Ptr(d + red_off, torch.int8), ep=torch.zeros(NB, dtype=torch.int32, device='cuda'), lib=None)
        if self.rank == 0:
            so = os.environ.get('ARC_HOST_ARC_SO') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'host_arc.so')
            L = ctypes.CDLL(so)
            L.arc_create.restype = ctypes.c_void_p
            L.arc_create.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 6 + [ctypes.c_int64] * 8 + [ctypes.c_void_p, ctypes.c_int]
            for f in ('arc_start', 'arc_simd'):
                getattr(L, f).argtypes = [ctypes.c_void_p]
            L.arc_stop.argtypes = [ctypes.c_void_p]
            L.arc_stat.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.arc_stat.restype = ctypes.c_int64
            pin = os.environ.get('ARC_HOST_AR_CPU_PIN')
            cpus = None
            if pin:
                ids = []
                for part in pin.split(','):
                    a, _, b = part.partition('-')
                    ids += list(range(int(a), int(b or a) + 1))
                cpus = (ctypes.c_int * len(ids))(*ids)
                nt = len(ids)
            else:
                nt = CPU_THREADS
            h = L.arc_create(ctypes.c_void_p(host), W, NB, K, Q8_G, FS, nt, maxn, slot, own_slot, in_off, own_off, red_off, gf_off, cf_off, cpus, int(os.environ.get('ARC_HOST_AR_CPU_SIMD', '2')))
            assert h, 'arc_create failed'
            old = os.sched_getaffinity(0)
            if cpus is None:
                node = _gpu_node_cpus()
                if node:
                    os.sched_setaffinity(0, node)
            try:
                assert L.arc_start(h) == 0
            finally:
                os.sched_setaffinity(0, old)
            c.update(lib=L, h=h, nt=nt, simd=L.arc_simd(h))
            import atexit
            atexit.register(L.arc_stop, h)
        self.cpu = c

    def cpu_stats(self):
        c = self.cpu
        if not c or c['lib'] is None:
            return None
        return (c['lib'].arc_stat(c['h'], 0), c['lib'].arc_stat(c['h'], 1))

    def _go(self, k, grid):
        if not self.warm:
            return k[grid]

        def w(*a, **kw):
            ck = k.warmup(*a, grid=grid, **kw)
            if hasattr(ck, '_init_handles'):
                ck._init_handles()
        return w

    def error(self):
        return int(self._err_host[self.rank * FLAG_STRIDE])

    def clear_error(self):
        self._err_host[self.rank * FLAG_STRIDE] = 0
        self.stk.zero_()

    def _monitor(self):
        import sys, time
        while True:
            time.sleep(2.0)
            c = self.error()
            if c:
                print(f'[host_ar] rank {self.rank}: spin-wait timeout code {c} (peer never arrived); results of that all-reduce are garbage', file=sys.stderr, flush=True)
                if os.environ.get('ARC_HOST_AR_TIMEOUT_ACTION') == 'exit':
                    os._exit(70)
                self.clear_error()

    def _pre(self):
        if torch.cuda.is_current_stream_capturing():
            return None
        s = torch.cuda.current_stream()
        if self._ev is not None and self._ev_stream != s.cuda_stream:
            s.wait_event(self._ev)
        return s

    def _post(self, s):
        if s is not None:
            if self._ev is None or self._ev_stream != s.cuda_stream:
                self._ev, self._ev_stream = (torch.cuda.Event(), s.cuda_stream)
            self._ev.record(s)

    def supports(self, x):
        return x.dtype == torch.bfloat16 and x.is_contiguous() and (AR_MIN_BYTES // 2 <= x.numel() <= min(self.maxn, AR_MAX_BYTES // 2)) and (x.numel() % (self.world * (Q8_G if self.q8 else 1)) == 0)

    def __call__(self, x, out=None):
        n = x.numel()
        _trace(self, 'AR', n)
        out = torch.empty_like(x) if out is None else out
        q = n // self.world
        st = self._pre()
        from .link_gate import word as _gw
        gw = _gw(x.device)
        if gw is not None:
            gw.fill_(1)
        if _LB is not None and _LB(self, x, out, q):
            pass
        elif self.q8 and self.cpu_on and (self.bits in (5, 6, 8)) and (self.red_bits in (5, 6, 8)) and (CPU_MIN_BYTES <= 2 * n <= 2 * self.cpu['maxn']):
            c = self.cpu
            per = triton.cdiv(triton.cdiv(q, NB), Q8_G) * Q8_G
            sub = triton.cdiv(triton.cdiv(per, c['K']), Q8_G) * Q8_G
            self._go(_host_arc_kernel, (NB,))(x, out, c['hin'], c['own'], c['red'], c['gf'], c['cf'], c['ep'], self.err, self.stk, self._emu_buf(), self.sink, q, per, sub, c['maxn'], c['own_el'], RANK=self.rank, W=self.world, NBLK=NB, K=c['K'], FS=FLAG_STRIDE, BLOCK=CPU_TILE, G=Q8_G, B=self.bits, RB=self.red_bits, OWNQ=self.cpu_ownq, TMO=TIMEOUT_NS, EMU=self.emu, num_warps=CPU_WARPS)
        elif self.q8 and self.bits != 8 and self.use_pipe and (self.red_bits != 16):
            pp = self.pipe
            assert PIPE_CH % Q8_G == 0 and PIPE_CH >= 8 * 32 * PIPE_WARPS
            L = triton.cdiv(triton.cdiv(q, PIPE_BLOCKS * PIPE_ROUNDS), PIPE_CH) * PIPE_CH
            self._go(_host_ar_qbp_kernel, (3 * PIPE_BLOCKS,))(x, out, pp['hin'], pp['hred'], pp['f1'], pp['f2'], pp['ep'], self.err, self.stk, self._emu_buf(), self.sink, q, self.maxn, triton.cdiv(q, L), L, self.ncmax, RANK=self.rank, W=self.world, NBP=PIPE_BLOCKS, FS=FLAG_STRIDE, CH=PIPE_CH, G=Q8_G, B=self.bits, RB=self.red_bits, NS=PIPE_NS, TMO=TIMEOUT_NS, EMU=self.emu, num_warps=PIPE_WARPS)
        elif self.q8 and self.bits != 8:
            assert self.bits in (5, 6) and self.red_bits in (5, 6, 8, 16) and (TILE >= 8 * 32 * WARPS), (self.bits, self.red_bits)
            per = triton.cdiv(triton.cdiv(q, NB), Q8_G) * Q8_G
            self._go(_host_ar_qb_kernel, (NB,))(x, out, self.hin8, self.hred8, self.hred, self.f1, self.f2, self.ep, self.err, self.stk, q, per, self.maxn, self._emu_buf(), self.sink, RANK=self.rank, W=self.world, NBLK=NB, FS=FLAG_STRIDE, BLOCK=TILE, G=Q8_G, B=self.bits, RB=self.red_bits, TMO=TIMEOUT_NS, LD=POLL_LD, EMU=self.emu, num_warps=WARPS)
        elif self.q8 and self.use_pipe and (not Q8_RED_BF16):
            pp = self.pipe
            L = triton.cdiv(triton.cdiv(q, PIPE_BLOCKS * PIPE_ROUNDS), PIPE_CH) * PIPE_CH
            self._go(_host_ar_q8p_kernel, (3 * PIPE_BLOCKS,))(x, out, pp['hin'], pp['hred'], pp['f1'], pp['f2'], pp['ep'], self.err, self.stk, q, self.maxn, triton.cdiv(q, L), L, self.ncmax, RANK=self.rank, W=self.world, NBP=PIPE_BLOCKS, FS=FLAG_STRIDE, CH=PIPE_CH, G=Q8_G, NS=PIPE_NS, TMO=TIMEOUT_NS, num_warps=PIPE_WARPS)
        elif self.q8 and self.use_skew and (not Q8_RED_BF16):
            per = triton.cdiv(triton.cdiv(q, NB), Q8_G) * Q8_G
            sub = triton.cdiv(triton.cdiv(per, self.use_skew), Q8_G) * Q8_G
            self._go(_host_ar_q8s_kernel, (NB,))(x, out, self.hin8, self.hred8, self.skew['f1'], self.skew['f2'], self.ep, self.err, self.stk, q, per, sub, self.maxn, RANK=self.rank, W=self.world, NBLK=NB, FS=FLAG_STRIDE, BLOCK=TILE, G=Q8_G, K=self.use_skew, KMAX=SKEW_KMAX, TMO=TIMEOUT_NS, LD=POLL_LD, PW=PWAIT, num_warps=WARPS)
        elif self.q8:
            per = triton.cdiv(triton.cdiv(q, NB), Q8_G) * Q8_G
            self._go(_host_ar_q8_kernel, (NB,))(x, out, self.hin8, self.hred8, self.hred, self.f1, self.f2, self.ep, self.err, self.stk, n, q, per, self.maxn, RANK=self.rank, W=self.world, NBLK=NB, FS=FLAG_STRIDE, BLOCK=TILE, G=Q8_G, RED_BF16=Q8_RED_BF16, TMO=TIMEOUT_NS, LD=POLL_LD, PW=PWAIT, num_warps=WARPS)
        else:
            per = triton.cdiv(triton.cdiv(q, NB), 8) * 8
            self._go(_host_ar_kernel, (NB,))(x, out, self.hin, self.hred, self.f1, self.f2, self.ep, self.err, self.stk, n, q, per, self.maxn, RANK=self.rank, W=self.world, NBLK=NB, FS=FLAG_STRIDE, BLOCK=TILE, TMO=TIMEOUT_NS, LD=POLL_LD, num_warps=WARPS)
        if gw is not None:
            gw.fill_(0)
        self._post(st)
        return out

    def all_gather0(self, x):
        nbytes = x.numel() * x.element_size()
        if not x.is_contiguous() or nbytes % 4 or nbytes // 4 > self.maxn // 2:
            return None
        out = torch.empty((self.world * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        m = nbytes // 4
        _trace(self, 'AG', m)
        per = triton.cdiv(triton.cdiv(m, NB), 8) * 8
        st = self._pre()
        self._go(_host_ag_kernel, (NB,))(x.view(torch.int32), out.view(torch.int32), self.hin32, self.f1, self.ep, self.err, self.stk, m, per, self.maxn // 2, RANK=self.rank, W=self.world, NBLK=NB, FS=FLAG_STRIDE, BLOCK=TILE, TMO=TIMEOUT_NS, LD=POLL_LD, num_warps=WARPS)
        self._post(st)
        return out
