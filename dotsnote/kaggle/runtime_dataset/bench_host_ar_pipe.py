"""Host all-reduce (arcstats.host_ar, int8 payloads) on NG GPUs: the current Q8 kernel (tile x warps), the candidates
ARC_HOST_AR_POLL=ld (flag polling by ld.acquire instead of atomics), ARC_HOST_AR_SKEW=K (K-stage software pipeline
per block) and ARC_HOST_AR_PIPE (writer/reducer/gatherer CTA roles); bit-exactness vs the Q8 kernel (eager, all ranks,
CUDA-graph replay), timings, and PCIe/host-memory direction probes. Self-contained, no server.
  BENCH_GPUS=0,1,2,3 python dots3/bench_host_ar_pipe.py [NG=4] [tokens=1,64,192,256,1024]
env: ARC_HOST_AR_BLOCKS (Q8/skew CTAs), ARC_HOST_AR_PIPE_BLOCKS (CTAs per pipe role), BENCH_PROBE=1 (probes, run first), BENCH_PROBE_ONLY=1,
     BENCH_PIPE=0 (skip the pipe sweep), BENCH_DRILL=1 (only a watchdog drill: one rank skips an all-reduce, the
     peers' 3 s spin-wait timeout must fire -> 'har drill OK'), BENCH_KINDS=pipe,skew,ld,pw,skewpw (candidates to
     check/time), BENCH_ISOLATE=1 (default: sync + all-rank watchdog check after every candidate call; a candidate that
     times out is reported as 'har kindfail <kind> T <t> codes [...]' and dropped, the rest continue), BENCH_DEADLINE_S (default 600: whole-run deadline, workers dump their stacks
     and exit 15 s before it), ARC_HOST_AR_TIMEOUT_MS (default 20000: device-side spin-wait watchdog), BENCH_PORT.
Hang safety: fresh c10d port per run, workers die with the parent (PR_SET_PDEATHSIG), progress lines on stderr
('har-progress'), stack dump (faulthandler) at the deadline, kernel watchdog errors reported as 'har TIMEOUT', no
NCCL teardown (os._exit after a final gloo barrier).
Lines (stdout, rank 0, 'har ' prefix):
  har check T <t> pipe_vs_q8_equal <b> ranks_equal <b> graph_equal <b> rel_err <e> skew_equal <b> ld_pw_equal <b>
      kinds pipe=<0|1>,skew=..,ld=..,pw=..,skewpw=.. failed <kind:code,...|none>   (per-kind exactness incl. failures)
  har kindfail <kind> T <t> codes [<per rank>] after <s> s
  har q8tile T <t> q8 t<TILE>w<WARPS> <us graph> ... blocks <NB>
  har time T <t> nccl <us> q8 <us> q8_graph <us> | pipe r<R>ch<CH>w<W>ns<NS> <us graph> ... | best <cfg> <us> ...
  har cand T <t> ld t4096w8 <us> pw t<T>w<W> <us> ... skew k<K>t<T>w<W> <us> ... skewpw k2t<T>w<W> <us> ... (graph)
  har probe <what> T <t> MB <m> <us> <GB/s per GPU>   (all NG GPUs at once: D2H writes / H2D reads / both)
  har done | har TIMEOUT ... | har FAILED ...
T=1 isolates the fixed cost (launch + flag round trips through host memory)."""
import ctypes, faulthandler, os, signal, socket, sys, time
if os.environ.get("BENCH_GPUS"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["BENCH_GPUS"]
os.environ["ARC_HOST_AR_PIPE_ALLOC"] = "1"
os.environ["ARC_HOST_AR_SKEW_ALLOC"] = "1"
os.environ["ARC_HOST_AR_MONITOR"] = "0"          # the bench reads the error words itself
os.environ.setdefault("ARC_HOST_AR_Q8", "1")
os.environ.setdefault("ARC_HOST_AR_TIMEOUT_MS", "20000")
import torch
import torch.multiprocessing as mp
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOKS = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "1,64,192,256,1024").split(",")]
H = 5120
DEADLINE = float(os.environ.get("BENCH_DEADLINE_S", "600"))
CFGS = [(r, 2048, w, ns) for r in (1, 2, 4) for w in (4, 8) for ns in (0, 300, 1000)]
TILES = ((1024, 4), (1024, 8), (2048, 4), (2048, 8), (4096, 8), (4096, 16))
SKEWS = tuple(int(k) for k in os.environ.get("BENCH_SKEWS", "2,3,4").split(","))
SKEW_TILES = ((2048, 8), (4096, 8))


@triton.jit
def _probe_kernel(x_ptr, host_ptr, nbytes, NW, MODE: tl.constexpr, BLOCK: tl.constexpr):
    """MODE 0: programs write x -> host; 1: read host -> x; 2: first NW programs write, the rest read"""
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    if MODE == 2:
        wr = pid < NW
        idx = tl.where(wr, pid, pid - NW)
        cnt = tl.where(wr, NW, npg - NW)
    else:
        wr = MODE == 0
        idx = pid
        cnt = npg
    for s in range(idx * BLOCK, nbytes, cnt * BLOCK):
        o = s + tl.arange(0, BLOCK)
        m = o < nbytes
        if wr:
            tl.store(host_ptr + o, tl.load(x_ptr + o, mask=m), mask=m)
        else:
            tl.store(x_ptr + o, tl.load(host_ptr + o, mask=m, cache_modifier=".cv"), mask=m)


def _die_with_parent(parent):
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() != parent:
        os._exit(1)


def worker(rank, q, parent):
    _die_with_parent(parent)
    faulthandler.dump_traceback_later(max(DEADLINE - 15, 10), exit=True)   # a hang dumps every thread's stack
    t_start = time.time()

    def prog(*a):
        print(f"har-progress r{rank} {time.time() - t_start:.0f}s", *a, file=sys.stderr, flush=True)

    import torch.distributed as dist
    from arcstats import host_ar as ha
    from arcstats.host_ar import HostAllReduce
    torch.cuda.set_device(rank % torch.cuda.device_count())
    nccl = torch.cuda.device_count() >= NG
    prog("init_process_group port", os.environ["MASTER_PORT"])
    dist.init_process_group("nccl" if nccl else "gloo", rank=rank, world_size=NG)
    cpu = dist.new_group(backend="gloo")
    har = HostAllReduce(rank, NG, f"pipebench{os.getppid()}", cpu)
    prog("HostAllReduce ok")
    tile0, warps0 = ha.TILE, ha.WARPS

    def say(*a):
        if rank == 0:
            q.put(" ".join(str(x) for x in a))

    def finish(code):
        if rank == 0:
            q.put(None)
        q.close(); q.join_thread()
        os._exit(code)

    def check_err(where):
        """kernel watchdog: any rank's timed-out spin-wait (error word) stops all ranks with a report"""
        torch.cuda.synchronize()
        c = har.error()
        if c:
            prog("TIMEOUT", where, "code", c)
        bad = torch.tensor([c]); dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=cpu)
        if int(bad):
            say(f"har TIMEOUT at {where}: device spin-wait watchdog, max code {int(bad)} (see har-progress on stderr)")
            finish(3)

    def allreduce_ref(x):
        if nccl:
            r = x.float().clone(); dist.all_reduce(r); return r
        r = x.float().cpu(); dist.all_reduce(r, group=cpu); return r.cuda()

    def same_all(y):
        z = y.view(torch.int32).cpu().clone(); dist.broadcast(z, 0, group=cpu)
        ok = torch.tensor([int(torch.equal(z, y.view(torch.int32).cpu()))]); dist.all_reduce(ok, group=cpu)
        return int(ok) == NG

    def call(x, kind="q8", cfg=None):
        """kind: q8 | ld | pw | skew | skewpw | pipe; cfg: (tile, warps) for q8/ld/pw (+ K for skew*),
        (rounds, ch, warps, ns) for pipe"""
        har.use_pipe = kind == "pipe"
        har.use_skew = cfg[2] if kind.startswith("skew") else 0
        ha.POLL_LD = kind == "ld"
        ha.PWAIT = kind in ("pw", "skewpw")
        if kind == "pipe":
            ha.PIPE_ROUNDS, ha.PIPE_CH, ha.PIPE_WARPS, ha.PIPE_NS = cfg
        else:
            ha.TILE, ha.WARPS = cfg[:2] if cfg else (tile0, warps0)
        y = har(x)
        ha.TILE, ha.WARPS, ha.POLL_LD, ha.PWAIT = tile0, warps0, False, False
        har.use_pipe, har.use_skew = False, 0
        return y

    def tm(f, it):
        for _ in range(3):
            f()
        torch.cuda.synchronize(); dist.barrier(group=cpu); t0 = time.time()
        for _ in range(it):
            f()
        torch.cuda.synchronize()
        return (time.time() - t0) / it * 1e6

    def graph_time(fn, reps=10, it=10):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier(group=cpu)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(reps):
                y = fn()
        gr.replay(); torch.cuda.synchronize()
        return tm(lambda: gr.replay(), it) / reps, y

    if os.environ.get("BENCH_PROBE") == "1":
        prog("probe (first, no all-reduce kernels involved)")
        hp = har.pipe["hin"].addr + rank * (har.maxn * 2)     # this rank's own slot region (no AR in flight now)
        nb = 3 * ha.PIPE_BLOCKS
        for T in (64, 192, 1024):
            S = T * H                                           # int8 payload bytes of a T-token all-reduce
            for what, MODE, nbytes in (("d2h_1.0S", 0, S), ("h2d_1.5S", 1, S * 3 // 2), ("both_1.5S_each_way", 2, S * 3 // 2)):
                xb = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                hptr = ha._Ptr(hp, torch.uint8)

                def f():
                    _probe_kernel[(nb,)](xb, hptr, nbytes, nb * 2 // 5 if MODE == 2 else 0, MODE=MODE, BLOCK=4096,
                                         num_warps=4)
                us = tm(f, 30)
                say("har probe", what, "T", T, f"MB {nbytes / 1e6:.2f}", f"{us:.0f} us", f"{nbytes / us / 1e3:.1f} GB/s")
        if os.environ.get("BENCH_PROBE_ONLY") == "1":
            dist.barrier(group=cpu)
            say("har done")
            finish(0)

    def variants():
        v = [("q8", None), ("q8", (4096, 8)), ("ld", None), ("ld", (4096, 8)), ("pw", None)]
        v += [("q8", t) for t in TILES] + [("pw", t) for t in SKEW_TILES]
        for k in SKEWS:
            v += [("skew", (tile0, warps0, k)), ("skewpw", (tile0, warps0, k))]
            v += [("skew", t + (k,)) for t in SKEW_TILES]
        v += [("skewpw", t + (2,)) for t in SKEW_TILES]
        if do_pipe:
            v += [("pipe", c) for c in CFGS] + [("pipe", (2, 2048, 4, 300))]
        return v

    def precompile(T, tensors):
        """compile + load every variant on every rank before any of them runs: a rank compiling a new variant for tens
        of seconds while its peers spin in it desynchronized the 4xL4 bench (run 48: every spin-wait of the peers then
        timed out one after another, > 7 min); after this, all launches only wait on peers, never on a compiler"""
        t0 = time.time()
        har.warm = True
        n = 0
        for x in tensors:
            for kind, cfg in variants():
                call(x, kind, cfg)
                n += 1
        har.warm = False
        dt = time.time() - t0
        torch.cuda.synchronize(); dist.barrier(group=cpu)
        prog("T", T, f"precompiled {n} launches in {dt:.0f} s")
        say("har compile T", T, f"{n} variant launches {dt:.0f} s rank0, {time.time() - t0:.0f} s incl. barrier")

    it = 30 if nccl else 3
    do_pipe = os.environ.get("BENCH_PIPE", "1") == "1"
    if os.environ.get("BENCH_DRILL") == "1":   # watchdog drill: the last rank skips one all-reduce -> peers must time out
        ha.TIMEOUT_NS = int(3e9)
        x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16)
        for i in range(3):
            if not (rank == NG - 1 and i == 1):
                call(x)
        t0 = time.time()
        torch.cuda.synchronize()
        c = har.error()
        bad = torch.tensor([c]); dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=cpu)
        say(f"har drill {'OK' if int(bad) else 'FAILED (no timeout reported)'}: max code {int(bad)}, rank0 waited "
            f"{time.time() - t0:.1f} s")
        dist.barrier(group=cpu)
        finish(0 if int(bad) else 4)
    KINDS = [k for k in os.environ.get("BENCH_KINDS", "pipe,skew,ld,pw,skewpw").split(",") if k]
    if not do_pipe and "pipe" in KINDS:
        KINDS.remove("pipe")
    ISO = os.environ.get("BENCH_ISOLATE", "1") == "1"
    bad = {}                                   # kind -> first error code (same on every rank)
    FAULT = [os.environ.get("BENCH_FAULT")]    # isolation test only (use a kind with its own epochs: pipe)

    def guard(kind, T, fn):
        """run one candidate call; BENCH_ISOLATE=1 (default): sync + all-rank watchdog check right after it, so a
        candidate that times out on this platform is reported by name and dropped instead of poisoning every later
        check (sticky watchdog); the other candidates and the q8 kernel keep being measured"""
        if kind in bad or (kind != "q8" and kind not in KINDS):
            return None
        t0 = time.time()
        if kind == FAULT[0] and rank == NG - 1:   # BENCH_FAULT=<kind> test: the last rank skips one call of it
            FAULT[0] = None
            y = None
        else:
            y = fn()
        if not ISO:
            return y
        torch.cuda.synchronize()
        c = har.error()
        t = torch.tensor([c]); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu)
        if int(t):
            codes = [torch.tensor([0]) for _ in range(NG)]
            dist.all_gather(codes, torch.tensor([c]), group=cpu)
            prog("kindfail", kind, "T", T, "code", c, f"{time.time() - t0:.1f}s")
            say(f"har kindfail {kind} T {T} codes {[int(v) for v in codes]} after {time.time() - t0:.1f} s "
                f"(first timed-out wait per rank; code = 100*kernel + 10*phase + peer)")
            bad[kind] = int(t)
            dist.barrier(group=cpu); har.clear_error(); dist.barrier(group=cpu)
            if kind == "q8":
                say(f"har TIMEOUT at T{T} q8: the default int8 kernel itself timed out")
                finish(3)
            return None
        return y

    for T in TOKS:
        prog("T", T, "check")
        g = torch.Generator(device="cuda").manual_seed(1000 + rank)
        xs = [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3)]
        precompile(T, [xs[0]] + ([xs[0][:1].contiguous()] if T > 1 else []))
        eq = {k: True for k in ("pipe", "skew", "ld", "pw", "skewpw")}   # bit-identical to q8 (this rank)
        oks = teq = True
        err = 0.0

        def same(kind, a, y):
            if y is not None:
                eq[kind] = eq[kind] and torch.equal(a, y)
            return y
        for i in range(8):
            x = xs[i % 3]
            a = guard("q8", T, lambda: call(x))
            oks = oks and same_all(a)
            k = SKEWS[i % len(SKEWS)]
            b = same("pipe", a, guard("pipe", T, lambda: call(x, "pipe", CFGS[i % len(CFGS)])))
            c = same("skew", a, guard("skew", T, lambda: call(x, "skew", SKEW_TILES[i % len(SKEW_TILES)] + (k,))))
            same("ld", a, guard("ld", T, lambda: call(x, "ld")))
            same("pw", a, guard("pw", T, lambda: call(x, "pw")))
            same("skewpw", a, guard("skewpw", T, lambda: call(x, "skewpw", (tile0, warps0, k))))
            teq = teq and torch.equal(a, call(x, "q8", TILES[i % len(TILES)]))   # q8 tile/warps variants: same math
            if T > 1:   # interleave another size
                guard("skew", T, lambda: call(xs[0][:1].contiguous(), "skew", (tile0, warps0, k)))
                guard("pipe", T, lambda: call(xs[0][:1].contiguous(), "pipe", CFGS[0]))
            for y in (b, c):
                if y is not None:
                    oks = oks and same_all(y)
            ref = allreduce_ref(x)
            err = max(err, ((a.float() - ref).abs().max() / ref.abs().max()).item())
        check_err(f"T{T} check")
        # failures are agreed across ranks (bad), equality is per rank -> AND over ranks
        for kk in eq:
            v = torch.tensor([int(eq[kk] and kk not in bad)]); dist.all_reduce(v, op=dist.ReduceOp.MIN, group=cpu)
            eq[kk] = bool(int(v))
        prog("T", T, "graph check")
        y0 = call(xs[1])
        geq = True
        for kind, cfg in (("skew", (tile0, warps0, SKEWS[0])), ("ld", None), ("pw", None), ("skewpw", (tile0, warps0, 2)), ("pipe", (2, 2048, 4, 300))):
            r_ = guard(kind, T, lambda: graph_time(lambda: call(xs[1], kind, cfg), reps=3, it=2))
            if r_ is not None:
                geq = geq and torch.equal(r_[1], y0)
            else:
                eq[kind] = False
        check_err(f"T{T} graph check")
        say("har check T", T, "pipe_vs_q8_equal", eq["pipe"] if do_pipe else "skipped", "ranks_equal", oks, "graph_equal", geq,
            f"rel_err {err:.4f}", "skew_equal", eq["skew"], "ld_pw_equal", eq["ld"] and eq["pw"] and eq["skewpw"],
            "kinds", ",".join(f"{kk}={int(eq[kk])}" for kk in eq), "tiles_equal", teq, "failed", ",".join(f"{kk}:{v}" for kk, v in bad.items()) or "none")
        x = xs[0]
        xn = x.clone()
        t_n = tm(lambda: dist.all_reduce(xn), it) if nccl else float("nan")
        t_q = tm(lambda: call(x), it)
        t_qg, _ = graph_time(lambda: call(x), it=it // 3 + 1)
        prog("T", T, "q8 tiles")
        q8v = {}
        for tile, wq in TILES:
            q8v[f"t{tile}w{wq}"], _ = graph_time(lambda: call(x, "q8", (tile, wq)), it=it // 3 + 1)
        check_err(f"T{T} q8tile")
        say("har q8tile T", T, " ".join(f"q8 {k} {v:.0f}" for k, v in q8v.items()), f"blocks {ha.NB}")
        prog("T", T, "candidates")
        cand = {}

        def gt(name, kind, cfg):
            r_ = guard(kind, T, lambda: graph_time(lambda: call(x, kind, cfg), it=it // 3 + 1))
            if r_ is not None:
                cand[name] = r_[0]
        gt("ld t4096w8", "ld", (4096, 8))
        for tile, wq in SKEW_TILES:
            gt(f"pw t{tile}w{wq}", "pw", (tile, wq))
        for k in SKEWS:
            for tile, wq in SKEW_TILES:
                gt(f"skew k{k}t{tile}w{wq}", "skew", (tile, wq, k))
        for tile, wq in SKEW_TILES:
            gt(f"skewpw k2t{tile}w{wq}", "skewpw", (tile, wq, 2))
        check_err(f"T{T} candidates")
        say("har cand T", T, " ".join(f"{k} {v:.0f}" for k, v in cand.items()))
        res = {}
        if do_pipe and "pipe" in KINDS:
            prog("T", T, "pipe sweep")
            for cfg in CFGS:
                r_ = guard("pipe", T, lambda: graph_time(lambda: call(x, "pipe", cfg), it=it // 3 + 1))
                if r_ is not None:
                    res["r%dch%dw%dns%d" % cfg] = r_[0]
            check_err(f"T{T} pipe")
        best = min(res.items(), key=lambda kv: kv[1]) if res else ("none", float("nan"))
        say("har time T", T, f"nccl {t_n:.0f} q8 {t_q:.0f} q8_graph {t_qg:.0f} |",
            " ".join(f"pipe {k} {v:.0f}" for k, v in res.items()), "| best", best[0], f"{best[1]:.0f}",
            f"blocks/role {ha.PIPE_BLOCKS}")
    prog("done")
    dist.barrier(group=cpu)
    say("har done")
    finish(0)    # no destroy_process_group: an NCCL teardown must never be able to hang the caller


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("BENCH_PORT") or str(_free_port())   # fresh port: no stale listeners
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, q, os.getpid()), daemon=True) for i in range(NG)]
    [p.start() for p in ps]
    import queue
    t0 = time.time()
    rc, done = 2, 0
    try:
        while time.time() - t0 < DEADLINE:
            try:
                m = q.get(timeout=2)
            except queue.Empty:
                if not all(p.is_alive() for p in ps):
                    time.sleep(1)
                    if not all(p.is_alive() for p in ps):
                        print("har FAILED: a worker exited", [p.exitcode for p in ps], flush=True)
                        rc = 1
                        break
                continue
            if m is None:
                done += 1
                if done >= 1:
                    rc = 0 if rc == 2 else rc
                    break
                continue
            print(m, flush=True)
            if m.startswith("har TIMEOUT"):
                rc = 3
        else:
            print(f"har TIMEOUT: bench deadline {DEADLINE:.0f} s", flush=True)
        for p in ps:
            p.join(timeout=20 if rc == 0 else 2)
    finally:
        for p in ps:
            if p.is_alive():
                p.kill()
    sys.exit(rc)
