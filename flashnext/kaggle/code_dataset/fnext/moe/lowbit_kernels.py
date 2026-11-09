import torch
import triton
import triton.language as tl
G = 128
import os
SWAP = os.environ.get('FN_MOE_SWAP') == '1'
PAIRS = os.environ.get('FN_MOE_PAIRS') == '1'
_PF = int(os.environ.get('FN_MOE_PF', '0'))

@triton.jit
def _lowbit_moe_gemm(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr, topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr, N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS, TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr=0):
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
                q = (wd >> 4 * j & 15 | 25600).to(tl.int16).to(tl.float16, bitcast=True)
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
                q = (wd >> 2 * j & 3 | (hb >> j & 1) << 2 | 25600).to(tl.int16).to(tl.float16, bitcast=True)
                accg += tl.dot(a, tl.trans(q))
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[:, None] * (zz[None, :] + 1024.0)) * sc[None, :]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[:, None]
    if COMBINE > 0:
        tl.atomic_add(out_ptr + (tok // COMBINE)[:, None] * stride_om + offs_n[None, :], acc, mask=mmask[:, None] & nmask[None, :], sem='relaxed')
    else:
        tl.store(out_ptr + tok[:, None] * stride_om + offs_n[None, :], acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])

@triton.jit
def _lowbit_moe_gemm_t(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr, topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr, N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS, TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr=0):
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
    w32 = tl.arange(0, 32)
    if bits == 4:
        r4 = tl.load(off4_ptr + e).to(tl.int64) + offs_n
        for g in range(0, KG):
            wd = tl.load(p4_ptr + (g * R4 + r4[:, None]) * 32 + w32[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            accg = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
            for j in tl.static_range(4):
                at = tl.load(x_base + (g * 128 + j * 32 + w32)[:, None], mask=mmask[None, :], other=0.0)
                q = (wd >> 4 * j & 15 | 25600).to(tl.int16).to(tl.float16, bitcast=True)
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
                at = tl.load(x_base + (g * 128 + j * 16 + w16)[:, None], mask=mmask[None, :], other=0.0)
                q = (wd >> 2 * j & 3 | (hb >> j & 1) << 2 | 25600).to(tl.int16).to(tl.float16, bitcast=True)
                accg += tl.dot(q, at)
            sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0).to(tl.float32)
            zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0).to(tl.float32)
            acc += (accg - xs[None, :] * (zz[:, None] + 1024.0)) * sc[:, None]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[None, :]
    if COMBINE > 0:
        tl.atomic_add(out_ptr + (tok // COMBINE)[None, :] * stride_om + offs_n[:, None], acc, mask=mmask[None, :] & nmask[:, None], sem='relaxed')
    else:
        tl.store(out_ptr + tok[None, :] * stride_om + offs_n[:, None], acc.to(tl.bfloat16), mask=mmask[None, :] & nmask[:, None])
_DQ3 = tl.constexpr('{ .reg .b32 a, b, u; shr.b32 a, $1, $5; shf.r.wrap.b32 b, $3, $3, $7; lop3.b32 u, a, b, 0x00030003, 0xE4; lop3.b32 $0, u, 0x00070007, 0x64006400, 0xEA; }')

@triton.jit
def _lowbit_moe_gemm_tp(x_ptr, xs_ptr, out_ptr, p4_ptr, p2_ptr, p1_ptr, sc_ptr, zp_ptr, bits_ptr, off4_ptr, off2_ptr, off1_ptr, offs_ptr, topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr, N, K, num_valid, stride_xm, stride_om, R4, R2, R1, RS, TOPK: tl.constexpr, X_PER_TOKEN: tl.constexpr, MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, COMBINE: tl.constexpr=0, PF: tl.constexpr=0, EVICT: tl.constexpr=0):
    EV: tl.constexpr = 'evict_first' if EVICT > 0 else ''
    EVS: tl.constexpr = 'evict_last' if EVICT > 1 else ''
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
            hb = tl.reshape(tl.join(h8, h8), (BLOCK_N, 16))
        else:
            w8 = tl.load(p2_ptr + (g * R2 + r2[:, None]) * 8 + w8i[None, :], mask=nmask[:, None], other=0, eviction_policy=EV)
            wd = tl.reshape(tl.join(w8, w8), (BLOCK_N, 16))
            if bits == 3:
                h4 = tl.load(p1_ptr + (g * R1 + r1[:, None]) * 4 + w4i[None, :], mask=nmask[:, None], other=0, eviction_policy=EV)
                h8 = tl.reshape(tl.join(h4, h4), (BLOCK_N, 8))
                hb = tl.reshape(tl.join(h8, h8), (BLOCK_N, 16))
            else:
                hb = zero
        accg = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        xs = tl.load(xs_ptr + xrow * KG + g, mask=mmask, other=0.0)
        for j in tl.static_range(8):
            at = tl.load(x_base + (g * 128 + j * 16 + w16)[:, None], mask=mmask[None, :], other=0.0)
            sh = zero + 2 * j
            rot = zero + ((j + 30 + 8 * (w16 // 2 % 2)) % 32)[None, :]
            q = tl.inline_asm_elementwise(_DQ3, '=r,r,r,r,r,r,r,r,r', [wd, hb, sh, rot], dtype=tl.float16, is_pure=True, pack=2)
            accg += tl.dot(q, at)
        sc = tl.load(sc_ptr + g * RS + rs, mask=nmask, other=0.0, eviction_policy=EV).to(tl.float32)
        zz = tl.load(zp_ptr + g * RS + rs, mask=nmask, other=0, eviction_policy=EV).to(tl.float32)
        acc += (accg - xs[None, :] * (zz[:, None] + 1024.0)) * sc[:, None]
    if MUL_W:
        tw = tl.load(topk_w_ptr + tok, mask=mmask, other=0.0).to(tl.float32)
        acc = acc * tw[None, :]
    if COMBINE > 0:
        tl.atomic_add(out_ptr + (tok // COMBINE)[None, :] * stride_om + offs_n[:, None], acc, mask=mmask[None, :] & nmask[:, None], sem='relaxed')
    else:
        tl.store(out_ptr + tok[None, :] * stride_om + offs_n[:, None], acc.to(tl.bfloat16), mask=mmask[None, :] & nmask[:, None], eviction_policy=EVS)

def _wrap32(w):
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)

def _pairs_from_codes(q):
    q = q.to(torch.int64).view(*q.shape[:2], 8, 8, 2)
    sh = (torch.arange(8, device=q.device) * 2).view(1, 1, 8, 1)
    p2 = ((q[..., 0] & 3) << sh | (q[..., 1] & 3) << sh + 16).sum(2)
    jj = torch.arange(8, device=q.device).view(1, 1, 8, 1)
    hb = ((q[..., 0] >> 2 & 1) << jj | (q[..., 1] >> 2 & 1) << jj + 16).sum(2)
    p1 = hb[..., 0::2] | hb[..., 1::2] << 8
    return (_wrap32(p2), _wrap32(p1))

def _codes_from_planar(p2, p1, KG):
    R = p2.shape[0]
    w = p2.view(R, KG, 1, 16).to(torch.int32) & 65535
    j = torch.arange(8, device=p2.device).view(1, 1, 8, 1)
    q = w >> 2 * j & 3
    if p1 is not None:
        h = p1.view(R, KG, 1, 16).to(torch.int32) & 255
        q = q | (h >> j & 1) << 2
    return q

def _pack_planar(q, bits):
    R, K = q.shape
    q = q.to(torch.int32).view(R, K // 128, 128)
    out = {}
    if bits == 4:
        f = q.view(R, K // 128, 4, 32)
        w = sum(((f[:, :, j, :] & 15) << 4 * j for j in range(4)))
        out['p4'] = w.view(R, K // 4).to(torch.int32).to(torch.int16) if False else _to_i16(w).view(R, K // 4)
    else:
        f = q.view(R, K // 128, 8, 16)
        w = sum(((f[:, :, j, :] & 3) << 2 * j for j in range(8)))
        out['p2'] = _to_i16(w).view(R, K // 8)
        if bits == 3:
            h = sum(((f[:, :, j, :] >> 2 & 1) << j for j in range(8)))
            out['p1'] = h.to(torch.uint8).view(R, K // 8)
    return out

def _to_i16(w):
    w = w.to(torch.int32)
    return torch.where(w >= 32768, w - 65536, w).to(torch.int16)

def _to_pairs(obj, bits, KG, off2, off1, n_out):
    obj.pairs = False
    if not PAIRS or any((b == 4 for b in bits)) or obj.R2 <= 1:
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
    obj.p2, obj.p1, obj.pairs = (lo, hi, True)

class LowbitProj:

    def __init__(self, bits, codes, scales, zeros, device):
        E = len(bits)
        n_out = next((c.shape[0] for c in codes if c is not None))
        self.K = next((c.shape[1] for c in codes if c is not None))
        self.N = n_out
        p4, p2, p1, sc, zp = ([], [], [], [], [])
        off4, off2, off1, offs = ([0] * E, [0] * E, [0] * E, [0] * E)
        n4 = n2 = n1 = ns = 0
        for e in range(E):
            if not bits[e]:
                continue
            pk = _pack_planar(codes[e], bits[e])
            if bits[e] == 4:
                p4.append(pk['p4'])
                off4[e] = n4
                n4 += n_out
            else:
                p2.append(pk['p2'])
                off2[e] = n2
                n2 += n_out
                if bits[e] == 3:
                    p1.append(pk['p1'])
                    off1[e] = n1
                    n1 += n_out
            sc.append(scales[e].half())
            zp.append(zeros[e].to(torch.uint8))
            offs[e] = ns
            ns += n_out
        KG = self.K // G

        def gmajor(parts, words, dtype):
            if not parts:
                return (torch.zeros(1, 1, 1, dtype=dtype).to(device), 1)
            t = torch.cat(parts).view(-1, KG, words).transpose(0, 1).contiguous()
            return (t.to(device), t.shape[1])
        self.p4, self.R4 = gmajor(p4, 32, torch.int16)
        self.p2, self.R2 = gmajor(p2, 16, torch.int16)
        self.p1, self.R1 = gmajor(p1, 16, torch.uint8)
        _to_pairs(self, bits, KG, off2, off1, n_out)
        self.sc = torch.cat(sc).t().contiguous().to(device)
        self.zp = torch.cat(zp).t().contiguous().to(device)
        self.RS = self.sc.shape[1]
        t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
        self.bits, self.off4, self.off2, self.off1, self.offs = (t(bits), t(off4), t(off2), t(off1), t(offs))

    def nbytes(self):
        return sum((x.numel() * x.element_size() for x in (self.p4, self.p2, self.p1, self.sc, self.zp)))

    def __call__(self, x, out, topk_w, sorted_ids, expert_ids, num_post_pad, topk, x_per_token, mul_w, block_m, block_n=64, num_warps=4, num_stages=2, xs=None, combine=0, evict=0, max_rows=None):
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        if xs is None:
            xs = x.view(x.shape[0], -1, G).float().sum(-1)
        rows = sorted_ids.shape[0] if max_rows is None else min(sorted_ids.shape[0], max_rows)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(self.N, block_n))
        kern = _lowbit_moe_gemm_tp if getattr(self, 'pairs', False) else _lowbit_moe_gemm_t if SWAP else _lowbit_moe_gemm
        kern[grid](x, xs, out, self.p4, self.p2, self.p1, self.sc, self.zp, self.bits, self.off4, self.off2, self.off1, self.offs, topk_w, sorted_ids, expert_ids, num_post_pad, self.N, self.K, topk_w.numel(), x.stride(0), out.stride(0), self.R4, self.R2, self.R1, self.RS, TOPK=topk, X_PER_TOKEN=x_per_token, MUL_W=mul_w, BLOCK_M=block_m, BLOCK_N=block_n, COMBINE=combine, num_warps=num_warps, num_stages=num_stages, **{'PF': _PF, 'EVICT': evict} if kern is _lowbit_moe_gemm_tp else {})

def proj_from_packed(experts, which, device):
    obj = LowbitProj.__new__(LowbitProj)
    p4, p2, p1, sc, zp = ([], [], [], [], [])
    E = len(experts)
    off4, off2, off1, offs, bits = ([0] * E, [0] * E, [0] * E, [0] * E, [0] * E)
    n4 = n2 = n1 = ns = 0
    ref = next((ex for ex in experts if ex is not None))
    n_out, kg = ref[which + '_s'].shape
    for i, ex in enumerate(experts):
        if ex is None:
            continue
        b = int(ex['bits'])
        bits[i] = b
        pl = ex[which]
        if b == 4:
            p4.append(pl['p4'].to(device).view(n_out, kg * 32))
            off4[i] = n4
            n4 += n_out
        else:
            p2.append(pl['p2'].to(device).view(n_out, kg * 16))
            off2[i] = n2
            n2 += n_out
            if b == 3:
                p1.append(pl['p1'].to(device).view(n_out, kg * 16))
                off1[i] = n1
                n1 += n_out
        sc.append(ex[which + '_s'].to(device).half())
        zp.append(ex[which + '_z'].to(device).to(torch.uint8))
        offs[i] = ns
        ns += n_out
    obj.N, obj.K = (n_out, kg * G)

    def gmajor(parts, words, dtype):
        if not parts:
            return (torch.zeros(1, 1, 1, dtype=dtype, device=device), 1)
        t = torch.cat(parts).view(-1, kg, words).transpose(0, 1).contiguous()
        return (t, t.shape[1])
    obj.p4, obj.R4 = gmajor(p4, 32, torch.int16)
    obj.p2, obj.R2 = gmajor(p2, 16, torch.int16)
    obj.p1, obj.R1 = gmajor(p1, 16, torch.uint8)
    _to_pairs(obj, bits, kg, off2, off1, n_out)
    obj.sc = torch.cat(sc).t().contiguous()
    obj.zp = torch.cat(zp).t().contiguous()
    obj.RS = obj.sc.shape[1]
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    obj.bits, obj.off4, obj.off2, obj.off1, obj.offs = (t(bits), t(off4), t(off2), t(off1), t(offs))
    return obj
_TUNE = None

def _tile(proj, M, block_m):
    global _TUNE
    if _TUNE is None:
        import json
        f = os.environ.get('FN_MOE_TUNE')
        d = json.load(open(f)) if f and os.path.exists(f) else {}
        _TUNE = {k: sorted(((int(m), tuple(c)) for m, c in v.items())) for k, v in d.items() if k in ('gu', 'dn')}
        if 'lb2' in d:
            _TUNE['lb2'] = sorted(((int(m), c) for m, c in d['lb2'].items()))
    t = _TUNE.get(proj)
    if not t:
        return (block_m, 64, 4, 2)
    for m, c in t:
        if m >= M:
            return c
    return t[-1][1]

class LowbitMoE:

    def __init__(self, gate_up, down, num_global_experts, block_m=16):
        self.gu, self.dn, self.E, self.block_m = (gate_up, down, num_global_experts, block_m)

    def nbytes(self):
        return self.gu.nbytes() + self.dn.nbytes()

    def __call__(self, x, topk_w, topk_ids, expert_map=None, tiles=None, lb2=None):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        M, H = x.shape
        topk = topk_ids.shape[1]
        if lb2 is None and _LB2 and getattr(self.gu, 'pairs', False) and getattr(self.dn, 'pairs', False):
            lb2 = _lb2_cfg(M)
        if lb2:
            return self._call_lb2(x, topk_w, topk_ids, expert_map, lb2)
        if tiles is not None:
            cg, cd = tiles
        else:
            cg, cd = (_tile('gu', M, self.block_m), _tile('dn', M, self.block_m))
            if not _TUNE and getattr(self.gu, 'pairs', False):
                cg, cd = ((cg[0], 128, 4, 2), (cd[0], 128, 4, 2))
        al = {}
        for bm in {cg[0], cd[0]}:
            al[bm] = moe_align_block_size(topk_ids, bm, self.E, expert_map, ignore_invalid_experts=expert_map is not None)
        if _FUSE_SILU:
            xh, xs1 = prep_x(x)
        else:
            xh, xs1 = (x if x.dtype == torch.float16 else x.to(torch.float16), None)
        c1 = torch.empty(M * topk, self.gu.N, dtype=torch.bfloat16, device=x.device)
        tw = topk_w.reshape(-1).float()
        mr = (lambda bm: topk_ids.numel() + self.gu.bits.numel() * (bm - 1)) if expert_map is not None else lambda bm: None
        self.gu(xh, c1, tw, *al[cg[0]], topk, True, False, *cg, xs=xs1, max_rows=mr(cg[0]), evict=_EVICT)
        xs2 = None
        if _FUSE_SILU:
            c2, xs2 = silu_prep_local(c1, topk_ids, expert_map)
        else:
            c2 = torch.empty(M * topk, self.gu.N // 2, dtype=torch.bfloat16, device=x.device)
            torch.ops._C.silu_and_mul(c2, c1)
        c3 = torch.empty(M * topk, self.dn.N, dtype=torch.bfloat16, device=x.device)
        self.dn(c2, c3, tw, *al[cd[0]], 1, False, True, *cd, xs=xs2, max_rows=mr(cd[0]), evict=_EVICT)
        return combine_local(c3, topk_ids, expert_map, M, topk)

    def _call_lb2(self, x, topk_w, topk_ids, expert_map, cfg):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        try:
            from . import lowbit2
        except ImportError:
            import lowbit2
        M = x.shape[0]
        topk = topk_ids.shape[1]
        sid, eid, npp = moe_align_block_size(topk_ids, 16, self.E, expert_map, ignore_invalid_experts=expert_map is not None)
        its = {}
        items = lambda cm: its.setdefault(cm, lowbit2.make_items(eid, npp, cm))
        if _FUSE_SILU:
            xh, xs = prep_x(x)
        else:
            xh = x if x.dtype == torch.float16 else x.to(torch.float16)
            xs = xh.view(M, -1, G).float().sum(-1)
        tw = topk_w.reshape(-1).float()
        c1 = torch.empty(M * topk, self.gu.N, dtype=torch.bfloat16, device=x.device)
        bn, cm, nw, ns, ev = cfg['gu']
        lowbit2.gemm2(self.gu, xh, c1, tw, sid, eid, *items(cm), topk, True, False, (bn, cm, nw, ns), xs=xs, evict=ev)
        if _FUSE_SILU:
            c2h, xs2 = silu_prep_local(c1, topk_ids, expert_map)
        else:
            c2 = torch.empty(M * topk, self.gu.N // 2, dtype=torch.bfloat16, device=x.device)
            torch.ops._C.silu_and_mul(c2, c1)
            c2h = c2.to(torch.float16)
            xs2 = c2h.view(M * topk, -1, G).float().sum(-1)
        c3 = torch.empty(M * topk, self.dn.N, dtype=torch.bfloat16, device=x.device)
        bn, cm, nw, ns, ev = cfg['dn']
        lowbit2.gemm2(self.dn, c2h, c3, tw, sid, eid, *items(cm), 1, False, True, (bn, cm, nw, ns), xs=xs2, evict=ev)
        return combine_local(c3, topk_ids, expert_map, M, topk)

@triton.jit
def _combine_local_kernel(c3_ptr, ids_ptr, map_ptr, out_ptr, N, E, TOPK: tl.constexpr, HAS_MAP: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = cols < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k).to(tl.int64)
        inr = (e >= 0) & (e < E)
        if HAS_MAP:
            ok = inr & (tl.load(map_ptr + e, mask=inr, other=-1) >= 0)
        else:
            ok = inr
        v = tl.load(c3_ptr + (t * TOPK + k) * N + cols, mask=cm & ok, other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + t * N + cols, acc.to(tl.bfloat16), mask=cm)

def combine_local(c3, topk_ids, expert_map, M, topk, block=512):
    N = c3.shape[1]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=c3.device)
    if M == 0:
        return out
    has = expert_map is not None
    ids = topk_ids.contiguous()
    E = expert_map.numel() if has else 1 << 30
    _combine_local_kernel[M, triton.cdiv(N, block)](c3, ids, expert_map if has else ids, out, N, E, TOPK=topk, HAS_MAP=has, BLOCK=block, num_warps=4)
    return out

@triton.jit
def _silu_prep_local_kernel(c1_ptr, c2_ptr, xs_ptr, ids_ptr, map_ptr, R, I, E, HAS_MAP: tl.constexpr, ROWS: tl.constexpr, GB: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    IG = I // 128
    rv = r < R
    e = tl.load(ids_ptr + r, mask=rv, other=0).to(tl.int64)
    rv = rv & (e >= 0) & (e < E)
    if HAS_MAP:
        rv = rv & (tl.load(map_ptr + e, mask=rv, other=-1) >= 0)
    m = rv[:, None, None] & (g < IG)[None, :, None]
    col = g[None, :, None] * 128 + c[None, None, :]
    base = c1_ptr + r.to(tl.int64)[:, None, None] * (2 * I) + col
    gt = tl.load(base, mask=m, other=0.0).to(tl.float32)
    up = tl.load(base + I, mask=m, other=0.0).to(tl.float32)
    sl = (gt / (1.0 + tl.exp(-gt))).to(tl.bfloat16).to(tl.float32)
    c2 = (sl * up).to(tl.bfloat16).to(tl.float16)
    tl.store(c2_ptr + r.to(tl.int64)[:, None, None] * I + col, c2, mask=m)
    xs = tl.sum(c2.to(tl.float32), axis=2)
    tl.store(xs_ptr + r[:, None] * IG + g[None, :], xs, mask=rv[:, None] & (g < IG)[None, :])

def silu_prep_local(c1, topk_ids, expert_map, rows=2, gb=None):
    R, I2 = c1.shape
    I = I2 // 2
    gb = gb or triton.next_power_of_2(I // G)
    c2 = torch.empty(R, I, dtype=torch.float16, device=c1.device)
    xs = torch.empty(R, I // 128, dtype=torch.float32, device=c1.device)
    if R:
        has = expert_map is not None
        ids = topk_ids.contiguous()
        _silu_prep_local_kernel[triton.cdiv(R, rows), triton.cdiv(I // 128, gb)](c1, c2, xs, ids, expert_map if has else ids, R, I, expert_map.numel() if has else 1 << 30, HAS_MAP=has, ROWS=rows, GB=gb, num_warps=4)
    return (c2, xs)
_EVICT = int(os.environ.get('FN_MOE_EVICT', '0'))

@triton.jit
def _prep_x_kernel(x_ptr, xh_ptr, xs_ptr, R, K, stride_x, ROWS: tl.constexpr, GB: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    KG = K // 128
    m = (r < R)[:, None, None] & (g < KG)[None, :, None]
    off = r.to(tl.int64)[:, None, None] * stride_x + g[None, :, None] * 128 + c[None, None, :]
    xh = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float16)
    tl.store(xh_ptr + r.to(tl.int64)[:, None, None] * K + g[None, :, None] * 128 + c[None, None, :], xh, mask=m)
    xs = tl.sum(xh.to(tl.float32), axis=2)
    tl.store(xs_ptr + r[:, None] * KG + g[None, :], xs, mask=(r < R)[:, None] & (g < KG)[None, :])

def prep_x(x, rows=1, gb=None):
    R, K = x.shape
    gb = gb or triton.next_power_of_2(K // G)
    xh = torch.empty(R, K, dtype=torch.float16, device=x.device)
    xs = torch.empty(R, K // G, dtype=torch.float32, device=x.device)
    if R:
        _prep_x_kernel[triton.cdiv(R, rows), triton.cdiv(K // G, gb)](x, xh, xs, R, K, x.stride(0), ROWS=rows, GB=gb, num_warps=4)
    return (xh, xs)
_FUSE_SILU = os.environ.get('FN_MOE_SILU', 'fused') == 'fused'
_LB2 = os.environ.get('FN_MOE_LB2') == '1'
_LB2_DEFAULT = {'gu': (64, 4, 4, 2, 1), 'dn': (64, 4, 4, 2, 1)}

def _lb2_cfg(M):
    _tile('gu', M, 16)
    t = _TUNE.get('lb2') if isinstance(_TUNE, dict) else None
    if not t:
        return _LB2_DEFAULT
    for m, c in t:
        if m >= M:
            return c
    return t[-1][1]

def build_local(packed, local_global, num_global_experts, device, block_m=16, marlin4=None):
    kept = packed['experts']
    assert len(kept) == num_global_experts, (len(kept), num_global_experts)
    if marlin4 is None:
        marlin4 = os.environ.get('FN_MOE_MARLIN4') == '1'
    exps = [packed[kept[g]] for g in local_global]
    if marlin4 and any((int(e['bits']) == 4 for e in exps)):
        return HybridMoE(exps, local_global, num_global_experts, device, block_m)
    gu = proj_from_packed(exps, 'gu', device)
    dn = proj_from_packed(exps, 'd', device)
    return LowbitMoE(gu, dn, num_global_experts, block_m)

class HybridMoE:

    def __init__(self, exps, local_global, E, device, block_m=16):
        try:
            from .marlin_part import MarlinPart
        except ImportError:
            from marlin_part import MarlinPart
        self.E = E
        i4 = [i for i, e in enumerate(exps) if int(e['bits']) == 4]
        il = [i for i, e in enumerate(exps) if int(e['bits']) != 4]
        self.m4 = MarlinPart([exps[i] for i in i4], device)
        self.map4 = torch.full((E,), -1, dtype=torch.int32, device=device)
        self.map4[torch.tensor([local_global[i] for i in i4], device=device, dtype=torch.long)] = torch.arange(len(i4), dtype=torch.int32, device=device)
        self.lb = None
        if il:
            sub = [exps[i] for i in il]
            self.lb = LowbitMoE(proj_from_packed(sub, 'gu', device), proj_from_packed(sub, 'd', device), E, block_m)
            self.mapl = torch.full((E,), -1, dtype=torch.int32, device=device)
            self.mapl[torch.tensor([local_global[i] for i in il], device=device, dtype=torch.long)] = torch.arange(len(il), dtype=torch.int32, device=device)
        self.gu = self.lb.gu if self.lb is not None else None

    def nbytes(self):
        return self.m4.nbytes() + (self.lb.nbytes() if self.lb is not None else 0)

    def __call__(self, x, topk_w, topk_ids, expert_map=None, tiles=None, lb2=None):
        y = self.m4(x, topk_w, topk_ids, self.map4, self.E).to(torch.bfloat16)
        if self.lb is not None:
            y = y + self.lb(x, topk_w, topk_ids, self.mapl, tiles=tiles, lb2=lb2)
        return y

def dequant_expert(ex, which):
    s, z = (ex[which + '_s'].float(), ex[which + '_z'].float())
    n_out, kg = s.shape
    b = int(ex['bits'])
    pl = ex[which]
    if b == 4:
        w = pl['p4'].view(n_out, kg, 32).to(torch.int32) & 65535
        j = torch.arange(4, device=w.device).view(1, 1, 4, 1)
        q = (w.unsqueeze(2) >> 4 * j & 15).reshape(n_out, kg, 128)
    else:
        q = _codes_from_planar(pl['p2'].view(n_out, kg * 16), pl.get('p1').view(n_out, kg * 16) if b == 3 else None, kg).reshape(n_out, kg, 128)
    return ((q.float() - z[..., None]) * s[..., None]).reshape(n_out, kg * 128)
