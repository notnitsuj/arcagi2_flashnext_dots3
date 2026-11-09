"""ARC_SHARD_PK=1: request-packed sharded indexer logits for decode (G = MTP query tokens per request, consecutive rows).

The per-row kernel (shard_indexer._shard_logits_kernel) loads every key block once per query token and re-loads the
query for every key block, converting both FP8 operands to fp16. Here one program owns one request's G tokens x H heads
as a single FP8 MMA operand, kept resident while it walks KB key blocks: keys are read once per request, the query once
per program, and the dot runs on FP8 tensor cores (sm89+). Output layout is unchanged ([rows, Lc], col j = pos j*P+rank).
"""
import os

import triton
import triton.language as tl

BN = int(os.environ.get("ARC_SHARD_PK_BN", "64"))
KB = int(os.environ.get("ARC_SHARD_PK_KB", "8"))          # key blocks per program
WARPS = int(os.environ.get("ARC_SHARD_PK_WARPS", "8"))


@triton.jit
def _shard_logits_pk_kernel(q_ptr, w_ptr, base_ptr, bt_ptr, req_ptr, ctx_ptr, out_ptr,
                            rank, P, BS, R_local, RPB, STRIDE_BLK, stride_q_row, stride_q_h, stride_w_row, stride_bt,
                            stride_out, Lc,
                            G: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    grp = tl.program_id(0)
    jb = tl.program_id(1) * (BN * KB)
    r0 = grp * G
    gh = tl.arange(0, G * H)                     # packed (token, head) index
    g = gh // H
    h = gh % H
    d = tl.arange(0, D)
    ctx_g = tl.load(ctx_ptr + r0 + tl.arange(0, G))          # [G] causal context per token
    cmax = tl.max(ctx_g, 0)
    if jb * P + rank >= cmax:
        for kb in range(KB):
            j = jb + kb * BN + tl.arange(0, BN)
            for t in range(G):
                tl.store(out_ptr + (r0 + t) * stride_out + j, float("-inf"), mask=j < Lc)
        return
    req = tl.load(req_ptr + r0).to(tl.int64)
    q = tl.load(q_ptr + (r0 + g)[:, None] * stride_q_row + h[:, None] * stride_q_h + d[None, :])   # [G*H, D] fp8
    w = tl.load(w_ptr + (r0 + g) * stride_w_row + h)                                             # [G*H]
    for kb in range(KB):
        j = jb + kb * BN + tl.arange(0, BN)
        pos = j * P + rank
        anyv = pos < cmax
        blk = tl.load(bt_ptr + req * stride_bt + pos // BS, mask=anyv, other=0).to(tl.int64)
        lrow = (blk * BS + pos % BS) // P
        okr = anyv & (blk >= 0) & (lrow < R_local)
        lrow = tl.where(okr, lrow, 0)
        a = base_ptr + (lrow // RPB) * STRIDE_BLK + (lrow % RPB) * 132
        k = tl.load(a.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :], mask=okr[:, None], other=0.0)
        sc = tl.load((a + 128).to(tl.pointer_type(tl.float32)), mask=okr, other=0.0)
        s = tl.maximum(tl.dot(k, tl.trans(q)), 0.0) * w[None, :]      # [BN, G*H]
        s3 = tl.reshape(s, (BN, G, H))
        lg = tl.sum(s3, axis=2) * sc[:, None]                          # [BN, G]
        for t in range(G):
            sel = (tl.arange(0, G) == t)[None, :]
            col = tl.sum(tl.where(sel, lg, 0.0), axis=1)               # [BN]
            ct = tl.load(ctx_ptr + r0 + t)
            v = okr & (pos < ct)
            tl.store(out_ptr + (r0 + t) * stride_out + j, tl.where(v, col, float("-inf")), mask=j < Lc)


def logits(store, q, w, block_table, req, ctx, out, Lc, G):
    rows, H, D = q.shape
    _shard_logits_pk_kernel[(rows // G, triton.cdiv(Lc, BN * KB))](
        q, w, store.base, block_table, req, ctx, out, store.rank, store.P, store.BS, store.R_local, store.RPB,
        store.STRIDE_BLK, q.stride(0), q.stride(1), w.stride(0), block_table.stride(0), out.stride(0), Lc,
        G=G, H=H, D=D, BN=BN, KB=KB, num_warps=WARPS)
    return out
