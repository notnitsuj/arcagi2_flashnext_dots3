"""Sharded indexer decode logits, variants of shard_indexer._shard_logits_kernel for the L4 (benchmarked by
dots3/bench_idx_l4.py; chosen by a startup gate).

  row8: per-row like the original, but the QK dot runs on FP8 tensor cores (sm89+): no fp8->fp16 conversion of the key
        tile or of the [H, D] query that every program re-loads.
  grp:  one program owns G consecutive rows of the same request (the MTP query tokens) and a run of KB key tiles: each
        key tile is loaded once and dotted with the G queries in turn (FP8 or fp16 MMA), instead of once per row.
Output layout is unchanged ([rows, Lc] fp32, col j = pos j*P+rank, -inf where invalid).
"""
import triton
import triton.language as tl


@triton.jit
def _row8_kernel(q_ptr, w_ptr, base_ptr, bt_ptr, req_ptr, ctx_ptr, out_ptr,
                 rank, P, BS, R_local, RPB, STRIDE_BLK, stride_q_row, stride_q_h, stride_w_row, stride_bt,
                 stride_out, Lc,
                 H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, F8: tl.constexpr, NOINF: tl.constexpr = False):
    row = tl.program_id(0)
    j0 = tl.program_id(1) * BN
    ctx = tl.load(ctx_ptr + row)
    j = j0 + tl.arange(0, BN)
    if j0 * P + rank >= ctx:
        if not NOINF:
            tl.store(out_ptr + row * stride_out + j, float("-inf"), mask=j < Lc)
        return
    pos = j * P + rank
    valid = pos < ctx
    req = tl.load(req_ptr + row).to(tl.int64)
    blk = tl.load(bt_ptr + req * stride_bt + pos // BS, mask=valid, other=0).to(tl.int64)
    lrow = (blk * BS + pos % BS) // P
    valid = valid & (blk >= 0) & (lrow < R_local)
    lrow = tl.where(valid, lrow, 0)
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    a = base_ptr + (lrow // RPB) * STRIDE_BLK + (lrow % RPB) * 132
    k = tl.load(a.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :], mask=valid[:, None], other=0.0)
    sc = tl.load((a + 128).to(tl.pointer_type(tl.float32)), mask=valid, other=0.0)
    q = tl.load(q_ptr + row * stride_q_row + h[:, None] * stride_q_h + d[None, :])
    if F8:
        s = tl.dot(k, tl.trans(q))
    else:
        s = tl.dot(k.to(tl.float16), tl.trans(q.to(tl.float16)))
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + row * stride_w_row + h)
    logit = tl.minimum(tl.sum(s * w[None, :], axis=1) * sc, 60000.0)   # fits an fp16 output buffer
    tl.store(out_ptr + row * stride_out + j, tl.where(valid, logit, float("-inf")), mask=j < Lc)


@triton.jit
def _grp_kernel(q_ptr, w_ptr, base_ptr, bt_ptr, req_ptr, ctx_ptr, out_ptr,
                rank, P, BS, R_local, RPB, STRIDE_BLK, stride_q_row, stride_q_h, stride_w_row, stride_bt,
                stride_out, Lc, rows,
                G: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr,
                F8: tl.constexpr, NOINF: tl.constexpr = False):
    """G consecutive rows x KB key tiles. Rows of one request (MTP verify tokens, prefill chunk runs) share each key
    tile; a group spanning requests loads keys per row."""
    r0 = tl.program_id(0) * G
    jb = tl.program_id(1) * (BN * KB)
    gi = r0 + tl.arange(0, G)
    gv = gi < rows
    ctx_g = tl.load(ctx_ptr + gi, mask=gv, other=0)
    req_g = tl.load(req_ptr + gi, mask=gv, other=-1)
    req0 = tl.load(req_ptr + r0)
    uni = tl.sum(((req_g == req0) | (~gv)).to(tl.int32), 0) == G
    cmax = tl.max(ctx_g, 0)
    d = tl.arange(0, D)
    h = tl.arange(0, H)
    for kb in range(KB):
        j = jb + kb * BN + tl.arange(0, BN)
        pos = j * P + rank
        if jb + kb * BN < Lc:
            if (jb + kb * BN) * P + rank >= cmax:
                if not NOINF:   # NOINF: the consumer (fast_topk.local) never reads columns at or beyond ctx
                    for t in tl.static_range(G):
                        tl.store(out_ptr + (r0 + t) * stride_out + j, float("-inf"), mask=(j < Lc) & (r0 + t < rows))
            elif uni:
                anyv = pos < cmax
                blk = tl.load(bt_ptr + req0.to(tl.int64) * stride_bt + pos // BS, mask=anyv, other=0).to(tl.int64)
                lrow = (blk * BS + pos % BS) // P
                okr = anyv & (blk >= 0) & (lrow < R_local)
                lrow = tl.where(okr, lrow, 0)
                a = base_ptr + (lrow // RPB) * STRIDE_BLK + (lrow % RPB) * 132
                k = tl.load(a.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :], mask=okr[:, None], other=0.0)
                sc = tl.load((a + 128).to(tl.pointer_type(tl.float32)), mask=okr, other=0.0)
                if not F8:
                    k = k.to(tl.float16)
                for t in tl.static_range(G):
                    if r0 + t < rows:
                        q = tl.load(q_ptr + (r0 + t) * stride_q_row + h[:, None] * stride_q_h + d[None, :])
                        if F8:
                            s = tl.dot(k, tl.trans(q))
                        else:
                            s = tl.dot(k, tl.trans(q.to(tl.float16)))
                        s = tl.maximum(s, 0.0)
                        w = tl.load(w_ptr + (r0 + t) * stride_w_row + h)
                        lg = tl.minimum(tl.sum(s * w[None, :], axis=1) * sc, 60000.0)   # fits an fp16 output buffer
                        ct = tl.load(ctx_ptr + r0 + t)
                        tl.store(out_ptr + (r0 + t) * stride_out + j,
                                 tl.where(okr & (pos < ct), lg, float("-inf")), mask=j < Lc)
            else:
                for t in tl.static_range(G):
                    if r0 + t < rows:
                        ct = tl.load(ctx_ptr + r0 + t)
                        rq = tl.load(req_ptr + r0 + t).to(tl.int64)
                        v = pos < ct
                        blk = tl.load(bt_ptr + rq * stride_bt + pos // BS, mask=v, other=0).to(tl.int64)
                        lrow = (blk * BS + pos % BS) // P
                        okr = v & (blk >= 0) & (lrow < R_local)
                        lrow = tl.where(okr, lrow, 0)
                        a = base_ptr + (lrow // RPB) * STRIDE_BLK + (lrow % RPB) * 132
                        k = tl.load(a.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :], mask=okr[:, None],
                                    other=0.0)
                        sc = tl.load((a + 128).to(tl.pointer_type(tl.float32)), mask=okr, other=0.0)
                        q = tl.load(q_ptr + (r0 + t) * stride_q_row + h[:, None] * stride_q_h + d[None, :])
                        if F8:
                            s = tl.dot(k, tl.trans(q))
                        else:
                            s = tl.dot(k.to(tl.float16), tl.trans(q.to(tl.float16)))
                        s = tl.maximum(s, 0.0)
                        w = tl.load(w_ptr + (r0 + t) * stride_w_row + h)
                        lg = tl.minimum(tl.sum(s * w[None, :], axis=1) * sc, 60000.0)
                        tl.store(out_ptr + (r0 + t) * stride_out + j, tl.where(okr, lg, float("-inf")), mask=j < Lc)


def logits(kind, store, q, w, block_table, req, ctx, out, Lc, G=4, BN=64, KB=1, warps=4, stages=2, f8=True, noinf=False):
    """noinf: skip the -inf stores of tiles wholly beyond the rows' contexts (those columns stay uninitialized)."""
    rows, H, D = q.shape
    common = (q, w, store.base, block_table, req, ctx, out, store.rank, store.P, store.BS, store.R_local, store.RPB,
              store.STRIDE_BLK, q.stride(0), q.stride(1), w.stride(0), block_table.stride(0), out.stride(0), Lc)
    if kind == "row8":
        _row8_kernel[(rows, triton.cdiv(Lc, BN))](*common, H=H, D=D, BN=BN, F8=f8, NOINF=noinf,
                                                  num_warps=warps, num_stages=stages)
    else:
        _grp_kernel[(triton.cdiv(rows, G), triton.cdiv(Lc, BN * KB))](*common, rows, G=G, H=H, D=D, BN=BN, KB=KB,
                                                                      F8=f8, NOINF=noinf, num_warps=warps,
                                                                      num_stages=stages)
    return out
