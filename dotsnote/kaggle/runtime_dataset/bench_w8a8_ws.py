"""Block-FP8 W8A8 dense linears at decode sizes, timed the way serving sees them: weights cold in L2 (each call in the
CUDA graph reads a different copy of the weight, > 4x L2 in total; between two calls of one layer type the whole model
streams through L2), activations hot. Compares vLLM's Triton kernel (+ the separate per-token-group quant launch) with
arcstats.w8a8_ws (weight-stationary M sub-tiles, split-K with in-kernel reduction, quant fused into the prologue).

  python dots3/bench_w8a8_ws.py check                      correctness (bit-exactness vs vLLM kernel, split-K stress)
  python dots3/bench_w8a8_ws.py gate M[,M...]              per shape: base (quant + vLLM, ARC_W8A8_CONFIG_DIR tiles)
                                                           vs ws (ARC_W8A8_WS_CONFIG table); prints the chosen shapes
  python dots3/bench_w8a8_ws.py tune OUT M[,M...]          cold-L2 sweep of both kernels -> OUT/w8a8_configs.json
                                                           ({vLLM config filename: {M: cfg}}, merged into the current
                                                           tiles) and OUT/w8a8_ws.json (WS table, null = keep vLLM)
Env: ARC_W8A8_SHAPES "N:K:count,..." (count = calls per decode step), ARC_TRITON_SM89=1 (+ ARC_EMULATE_L4_OCC=1) for
L4 codegen on an H100, BENCH_HOT=1 to time with L2-hot weights (old method), BENCH_WS_QUICK=1 smaller WS sweep."""
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "arcstats_pkg"))
if os.environ.get("ARC_TRITON_SM89") == "1":
    sys.path.insert(0, HERE)
    import triton_sm89  # noqa: F401
import torch
import triton
from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
from arcstats import w8a8_ws as WS

dev = torch.device("cuda")
torch.manual_seed(0)
SHAPES = [(2112, 5120, 33), (6144, 1024, 13), (5120, 4096, 13), (768, 5120, 45), (5120, 384, 45), (4096, 1024, 33)]
if os.environ.get("ARC_W8A8_SHAPES"):
    SHAPES = [tuple(int(v) for v in e.split(":")) + (1,) * (3 - len(e.split(":")))
              for e in os.environ["ARC_W8A8_SHAPES"].split(",") if e]
SMS = torch.cuda.get_device_properties(dev).multi_processor_count
L2 = getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 50 * 2**20)
HOT = os.environ.get("BENCH_HOT") == "1"
BASE_TABLE = {}   # (N, K) -> {M: cfg} from ARC_W8A8_CONFIG_DIR (the tiles serving uses now)
_cd = os.environ.get("ARC_W8A8_CONFIG_DIR", "")   # a directory of vLLM config files, or a bundled {filename: cfg} json
if _cd:
    import glob
    _files = (json.load(open(_cd)).items() if _cd.endswith(".json") else
              ((f, json.load(open(f))) for f in glob.glob(os.path.join(_cd, "N=*,K=*,*.json"))))
    for f, c in _files:
        b = os.path.basename(f).split(",")
        BASE_TABLE[(int(b[0][2:]), int(b[1][2:]))] = {int(k): v for k, v in c.items()}
_cur_cfg = {}
fu.get_w8a8_block_fp8_configs = lambda N, K, bn, bk: _cur_cfg.get((N, K))   # set per call below


def quant(x):
    return fu.per_token_group_quant_fp8(x, 128, use_ue8m0=False)   # serving: QuantFP8(use_ue8m0=False)


def base_mm(qx, xs, w, s, cfg):
    _cur_cfg[(w.shape[0], w.shape[1])] = {1: cfg} if cfg else None
    return fu.w8a8_triton_block_scaled_mm(qx, w, xs, s, [128, 128], torch.bfloat16)


def nearest(tab, M):
    return tab[min(tab, key=lambda m: abs(m - M))] if tab else None


class Shape:
    def __init__(self, N, K):
        self.N, self.K = N, K
        nb = N * K
        self.R = 1 if HOT else max(1, min(1024, -(-max(4 * L2, 160 * 2**20) // nb)))
        self.wall = (torch.randn(self.R * N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        self.w = [self.wall[i * N:(i + 1) * N] for i in range(self.R)]
        self.s = torch.rand((N + 127) // 128, (K + 127) // 128, device=dev) * 0.02 + 0.005
        self.ncall = max(self.R, 24)

    def x(self, M):
        g = torch.Generator(device=dev).manual_seed(M)
        return (torch.randn(M, self.K, device=dev, dtype=torch.bfloat16, generator=g)
                * (0.5 + 2 * torch.rand(M, 1, device=dev, generator=g))).to(torch.bfloat16)

    def ref(self, x):
        wf = self.w[0].float() * self.s.repeat_interleave(128, 0)[:self.N].repeat_interleave(128, 1)[:, :self.K]
        return x.float() @ wf.T

    def time(self, f, reps=None):
        """us per call of f(i) (i = weight copy), CUDA-graph replayed; None if it fails to compile/run"""
        try:
            for i in range(min(3, self.R)):
                f(i)
            torch.cuda.synchronize()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                f(0)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(self.ncall):
                    f(i % self.R)
            g.replay()
            torch.cuda.synchronize()
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n = 0
            while n < 3 or (time.perf_counter() - t0 < 0.03 and n < 50):   # >= 3 replays and ~30 ms
                g.replay()
                n += 1
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / (n * self.ncall) * 1e6
            del g
            return dt
        except Exception as ex:   # OutOfResources (smem on L4), compile errors
            if os.environ.get("BENCH_VERBOSE"):
                print("  fail", repr(ex)[:160], flush=True)
            torch.cuda.synchronize()
            return None


def relerr(o, r):
    return float((o.float() - r).abs().max() / r.abs().max())


def ws_fn(sh, x, qx, xs, cfg):
    if cfg.get("FUSE_Q"):
        return lambda i: WS.ws_mm(x, None, sh.w[i], sh.s, cfg)
    return lambda i: WS.ws_mm(qx, xs, sh.w[i], sh.s, cfg)


def ws_candidates(M, N, K, quick=False):
    """WS tile space for one (M, N, K): M covered by <= 4 groups of NSUB x BM rows without fully padded sub-tiles; split-K
    only while the grid is under ~2 waves, to about 1-2.5 waves; 8 warps only for big tiles; <= 99 KB smem (L4)"""
    nkb = -(-K // 128)
    out = []
    for BM, NSUB in itertools.product((16, 32, 64), (1, 2, 3, 4)):
        cover = BM * NSUB
        gm = -(-M // cover)
        if gm > 4 or gm * cover - M >= BM:
            continue
        for BN in (32, 64, 128):
            if BN > max(32, triton.next_power_of_2(N)):
                continue
            gn = -(-N // BN)
            sks = [1] + [sk for sk in (2, 3, 4, 6, 8, 12, 16, 20, 32) if gn * gm < 2 * SMS and nkb // sk >= 2
                         and 0.75 * SMS <= gn * gm * sk <= (2.5 if quick else 4) * SMS]
            for SK in sks:
                for nw, ns in itertools.product((4, 8), (2, 3, 4)):
                    if nw == 8 and cover * BN < (128 * 64 if quick else 64 * 64):
                        continue
                    if (cover + BN) * 128 * ns > 99 * 1024 or (ns == 4 and (quick or nkb // SK < 8)):
                        continue
                    out.append(dict(BM=BM, NSUB=NSUB, BN=BN, SPLIT_K=SK, FUSE_Q=0, num_warps=nw, num_stages=ns))
    return out


BASE_GRID = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128, GROUP_SIZE_M=g, num_warps=w, num_stages=st)
             for bm, bn, g, w, st in itertools.product((16, 32, 64, 128), (32, 64, 128), (1, 16), (4, 8), (2, 3, 4))
             if (bm + bn) * 128 * st <= 99 * 1024 and not (w == 8 and bm * bn < 64 * 64)]


def cmd_check():
    """bit-exactness of the non-split WS paths vs the vLLM kernel (same 1x128 x 128x128 block math), split-K closeness and
    determinism under repeated graph replays (arrival counter self-reset)"""
    bad = 0
    for N, K, _ in SHAPES[:6]:
        sh = Shape(N, K)
        for M in (1, 48, 192, 200, 256):
            x = sh.x(M)
            qx, xs = quant(x)
            ref = base_mm(qx, xs, sh.w[0], sh.s, dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
                                                       GROUP_SIZE_M=1, num_warps=4, num_stages=2))
            for cfg in (dict(BM=64, NSUB=3, BN=64), dict(BM=16, NSUB=4, BN=32, num_warps=8),
                        dict(BM=32, NSUB=2, BN=128, FUSE_Q=1), dict(BM=64, NSUB=3, BN=64, SPLIT_K=3),
                        dict(BM=16, NSUB=3, BN=32, SPLIT_K=8, FUSE_Q=1, num_stages=2)):
                if cfg.get("SPLIT_K", 1) > K // 256:
                    continue
                o = ws_fn(sh, x, qx, xs, cfg)(0)
                nmis = int((o != ref).sum())
                e = relerr(o, ref.float())
                ok = (nmis == 0) if cfg.get("SPLIT_K", 1) == 1 else e < 5e-3
                if cfg.get("SPLIT_K", 1) > 1:   # replay 50x in a graph: identical output each time
                    outs = []
                    g = torch.cuda.CUDAGraph()
                    f = ws_fn(sh, x, qx, xs, cfg)
                    f(0)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g):
                        oo = f(0)
                    for _ in range(50):
                        g.replay()
                        outs.append(oo.clone())
                    ok = ok and all(bool((t == o).all()) for t in outs)
                bad += not ok
                print(f"check N={N} K={K} M={M} {cfg}: mismatches {nmis} rel {e:.1e} {'ok' if ok else 'FAIL'}",
                      flush=True)
    print("check", "PASS" if not bad else f"FAIL {bad}")


def cmd_gate(Ms):
    """per shape and M: base = quant + vLLM kernel with the serving tiles (BASE_TABLE), ws = table cfg; err vs fp32"""
    table = WS.load_table(os.environ.get("ARC_W8A8_WS_CONFIG", ""))
    tot_b = tot_w = 0.0
    keep = []
    for N, K, cnt in SHAPES:
        tab = table.get((N, K))
        sh = Shape(N, K)
        tb_s = tw_s = 0.0
        errs = []
        for M in Ms:
            x = sh.x(M)
            qx, xs = quant(x)
            r = sh.ref(x)
            bcfg = nearest(BASE_TABLE.get((N, K)), M)
            tq = sh.time(lambda i: quant(x))
            tb = tq + sh.time(lambda i: base_mm(qx, xs, sh.w[i], sh.s, bcfg))
            eb = relerr(base_mm(qx, xs, sh.w[0], sh.s, bcfg), r)
            cfg = nearest(tab, M) if tab else None
            if cfg:
                tw = sh.time(ws_fn(sh, x, qx, xs, cfg))
                tw = None if tw is None else tw + (0 if cfg.get("FUSE_Q") else tq)
                ew = relerr(ws_fn(sh, x, qx, xs, cfg)(0), r)
            else:
                tw, ew = tb, eb
            if tw is None:
                tw, ew = float("inf"), float("inf")
            errs.append((eb, ew))
            tb_s += tb
            tw_s += tw
            print(f"N={N:5d} K={K:5d} M={M:4d}  base {tb:7.1f} us (quant {tq:5.1f}, err {eb:.1e})  ws {tw:7.1f} us "
                  f"(err {ew:.1e}) {cfg if cfg else '-'}", flush=True)
        use = tab is not None and tw_s < 0.97 * tb_s and max(e[1] for e in errs) <= max(0.03, 1.05 * max(e[0] for e in errs))
        if use:
            keep.append(f"{N}:{K}")
        tot_b += cnt * tb_s / len(Ms)
        tot_w += cnt * (tw_s if use else tb_s) / len(Ms)
    print(f"WS_GATE base {tot_b:.1f} us ws {tot_w:.1f} us per step-weighted layer set (M={','.join(map(str, Ms))}) "
          f"shapes {','.join(keep) or '-'}")


def cmd_tune(out, Ms):
    quick = os.environ.get("BENCH_WS_QUICK") == "1"
    os.makedirs(out, exist_ok=True)
    wsj = os.path.join(out, "w8a8_ws.json")
    bj = os.path.join(out, "w8a8_configs.json")
    ws_table = json.load(open(wsj)) if os.path.exists(wsj) else {}
    base_out = json.load(open(bj)) if os.path.exists(bj) else {}
    tot = {}
    for N, K, cnt in SHAPES:
        sh = Shape(N, K)
        fn = f"N={N},K={K},device_name=ANY,dtype=fp8_w8a8,block_shape=[128,128].json"
        bt = dict(base_out.get(fn) or {str(m): c for m, c in (BASE_TABLE.get((N, K)) or {}).items()})
        wt = ws_table.setdefault(f"{N}:{K}", {})
        wt.setdefault("1024", None)   # prefill chunks: vLLM kernel
        for M in Ms:
            t0 = time.time()
            x = sh.x(M)
            qx, xs = quant(x)
            tq = sh.time(lambda i: quant(x))
            cur = nearest(BASE_TABLE.get((N, K)), M)
            t_cur = sh.time(lambda i: base_mm(qx, xs, sh.w[i], sh.s, cur)) if cur else None
            ref = base_mm(qx, xs, sh.w[0], sh.s, cur) if cur else None
            res_b = []
            for cfg in BASE_GRID:
                if cfg["BLOCK_SIZE_M"] > max(16, triton.next_power_of_2(M)) or cfg["BLOCK_SIZE_N"] > max(32, triton.next_power_of_2(N)):
                    continue
                t = sh.time(lambda i, c=cfg: base_mm(qx, xs, sh.w[i], sh.s, c))
                if t is not None:
                    res_b.append((t, cfg))
            res_b.sort(key=lambda r: r[0])
            tb, cb = res_b[0]
            if ref is None:
                ref = base_mm(qx, xs, sh.w[0], sh.s, cb)
            res_w = []
            for cfg in ws_candidates(M, N, K, quick):
                t = sh.time(ws_fn(sh, x, qx, xs, cfg))
                if t is not None:
                    res_w.append((t + tq, cfg))
            res_w.sort(key=lambda r: r[0])
            for t, cfg in list(res_w[:6]):   # quant fused into the prologue for the best few
                c2 = dict(cfg, FUSE_Q=1)
                t2 = sh.time(ws_fn(sh, x, qx, xs, c2))
                if t2 is not None:
                    res_w.append((t2, c2))
            res_w.sort(key=lambda r: r[0])
            tw, cw = res_w[0] if res_w else (float("inf"), None)
            if cw is not None:   # numerics guard: within the split-K summation-order noise of the vLLM result
                e = relerr(ws_fn(sh, x, qx, xs, cw)(0), ref.float())
                if e > 5e-3:
                    print(f"  WS best cfg rejected (rel {e:.1e} vs vLLM kernel) {cw}", flush=True)
                    tw, cw = float("inf"), None
            bt[str(M)] = cb
            use_ws = cw is not None and tw < 0.97 * (tq + tb)
            wt[str(M)] = cw if use_ws else None
            tc = (tq + t_cur) if t_cur else float("nan")
            print(f"tune N={N:5d} K={K:5d} M={M:4d}  cur {tc:7.1f} us  base-cold {tq + tb:7.1f} us  ws {tw:7.1f} us "
                  f"(quant {tq:.1f}) -> {'WS ' + json.dumps(cw) if use_ws else 'base ' + json.dumps(cb)} "
                  f"[{len(res_b)}+{len(res_w)} cfgs, {time.time() - t0:.0f} s]", flush=True)
            a = tot.setdefault(M, [0.0, 0.0, 0.0])
            a[0] += cnt * tc
            a[1] += cnt * (tq + tb)
            a[2] += cnt * min(tq + tb, tw)
        base_out[fn] = dict(sorted(bt.items(), key=lambda kv: int(kv[0])))
        ws_table[f"{N}:{K}"] = dict(sorted(wt.items(), key=lambda kv: int(kv[0])))
        json.dump(base_out, open(bj, "w"), indent=0)
        json.dump(ws_table, open(wsj, "w"), indent=0)
    for M, (a, b, c) in tot.items():
        print(f"tune total M={M}: cur {a / 1e3:.2f} ms  base-cold-tuned {b / 1e3:.2f} ms  best(base|ws) {c / 1e3:.2f} ms "
              f"per step-weighted layer set (incl. quant)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    print(f"[bench_w8a8_ws] {torch.cuda.get_device_name()} SMs {SMS} L2 {L2 / 2**20:.0f} MB "
          f"{'HOT' if HOT else 'cold'} weights, sm89={os.environ.get('ARC_TRITON_SM89') == '1'}", flush=True)
    if cmd == "check":
        cmd_check()
    elif cmd == "gate":
        cmd_gate([int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "192").split(",")])
    elif cmd == "tune":
        cmd_tune(sys.argv[2], [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "192").split(",")])
