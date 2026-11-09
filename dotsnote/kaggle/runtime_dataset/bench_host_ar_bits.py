"""Packed low-bit host all-reduce (arcstats.host_ar, ARC_HOST_AR_BITS) vs the int8 (q8) kernel: error vs the fp32
reference sum, rank agreement, eager vs CUDA-graph agreement, PCIe bytes and CUDA-graph time per all-reduce.
  BENCH_GPUS=5,6 python dots3/bench_host_ar_bits.py [NG=2] [tokens=64,256,1024] [variants=8,6,6r8,5,5r8]
variant 'B' = B-bit payloads, 'BrR' = B-bit peer contributions + R-bit reduced quarter (r16 = bf16);
'cB' / 'cBrR' = the CPU-reduced path (ARC_HOST_AR_CPU, own quarter bf16), 'cBq' = own quarter as B-bit codes too;
'pB' = role-split pipelined kernel (ARC_HOST_AR_PIPE; p8 = the int8 one, p6 = int6, bit-identical to '6').
cX / pX rows also report the fraction of output elements that differ from the GPU kernel's X variant (vs_gpu_diff).
ARC_LINK_EMU_X=k inflates the PCIe bytes of the packed and CPU-path kernels k-fold (Kaggle link emulation).
Data: 'gauss' randn rows and 'heavy' rows (randn * exp(randn), per-row scale spread like residual-stream deltas).
Lines (stdout, rank 0): json per (T, variant). Hang safety as bench_host_ar.py (pdeathsig, deadline, os._exit).
"""
import ctypes, faulthandler, json, os, signal, socket, sys, time
if os.environ.get("BENCH_GPUS"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["BENCH_GPUS"]
os.environ.setdefault("ARC_HOST_AR_TIMEOUT_MS", "20000")
os.environ["ARC_HOST_AR_MONITOR"] = "0"
if any(v.startswith("c") for v in (sys.argv[3] if len(sys.argv) > 3 else "").split(",")):
    os.environ["ARC_HOST_AR_CPU_ALLOC"] = "1"
if any(v.startswith("p") for v in (sys.argv[3] if len(sys.argv) > 3 else "").split(",")):
    os.environ["ARC_HOST_AR_PIPE_ALLOC"] = "1"
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcstats_pkg"))
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOKS = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "64,256,1024").split(",")]
VARS = (sys.argv[3] if len(sys.argv) > 3 else "8,6,6r8,5,5r8").split(",")
H = 5120
DEADLINE = float(os.environ.get("BENCH_DEADLINE_S", "600"))


def parse(v):
    """-> (bits, red_bits, cpu path, own quarter quantized)"""
    cpu, ownq = v.startswith("c"), v.endswith("q")
    b, _, r = v.lstrip("cp").rstrip("q").partition("r")
    return int(b), int(r or b), cpu, ownq


def setv(har, v):
    har.bits, har.red_bits, cpu, ownq = parse(v)
    har.cpu_on, har.cpu_ownq = cpu and har.cpu is not None, ownq
    har.use_pipe = v.startswith("p") and har.pipe is not None


def pcie_bytes(n, w, b, r, g, cpu=False, ownq=False):
    """per-rank bytes (out, in) of one all-reduce of n elements: (W-1)/W n contributions out + n/W reduced out;
    (W-1)/W n peer copies in + (W-1)/W n reduced in; payload b/8 per element + 4/g scale bytes.
    CPU path: out (W-1)/W n contributions + n/W own (bf16, or b bits), in n reduced"""
    pb = lambda bits: (2.0 if bits == 16 else bits / 8 + 4 / g)
    q = n / w
    if cpu:
        return (w - 1) * q * pb(b) + q * pb(b if ownq else 16), n * pb(r)
    return (w - 1) * q * pb(b) + q * pb(r), (w - 1) * q * pb(b) + (w - 1) * q * pb(r)


def worker(rank, qu, parent):
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:
        pass
    if os.getppid() != parent:
        os._exit(1)
    faulthandler.dump_traceback_later(max(DEADLINE - 15, 10), exit=True)
    import torch.distributed as dist
    from arcstats import host_ar
    torch.cuda.set_device(int(os.environ.get("BENCH_RANK_DEV", ",".join(map(str, range(NG)))).split(",")[rank]))
    dist.init_process_group("gloo" if os.environ.get("BENCH_RANK_DEV") else "nccl", rank=rank, world_size=NG)
    cpu = dist.new_group(backend="gloo")
    har = host_ar.HostAllReduce(rank, NG, "benchbits", cpu)
    har.q8 = True
    G = host_ar.Q8_G
    # compile every variant before the ranks interact
    har.warm = True
    for T in TOKS:
        x = torch.zeros(T, H, device="cuda", dtype=torch.bfloat16)
        for v in VARS:
            setv(har, v)
            har(x)
    har.warm = False
    dist.barrier(group=cpu)
    res = []

    def tm(f, it=30):
        for _ in range(3):
            f()
        torch.cuda.synchronize(); dist.barrier(group=cpu); t0 = time.time()
        for _ in range(it):
            f()
        torch.cuda.synchronize()
        return (time.time() - t0) / it * 1e6

    for T in TOKS:
        g = torch.Generator(device="cuda").manual_seed(1000 + rank)
        data = {"gauss": torch.randn(T, H, device="cuda", generator=g).bfloat16(),
                "heavy": (torch.randn(T, H, device="cuda", generator=g) *
                          torch.exp(torch.randn(T, 1, device="cuda", generator=g))).bfloat16()}
        refs = {}
        for k, x in data.items():
            r = x.float().clone(); dist.all_reduce(r); refs[k] = r
        outs_by = {}
        for v in VARS:
            setv(har, v)
            row = dict(T=T, var=v, bits=har.bits, red_bits=har.red_bits, cpu=int(har.cpu_on), emu=har.emu)
            ok_rank = True
            for k, x in data.items():
                ys = [har(x) for _ in range(3)]
                torch.cuda.synchronize()
                same_rep = all(torch.equal(ys[0], y) for y in ys[1:])
                yr = ys[0].clone(); dist.broadcast(yr, 0)
                ok_rank = ok_rank and torch.equal(yr, ys[0]) and same_rep
                ref = refs[k]
                outs_by[(v, k)] = ys[0]
                twin = v.lstrip("cp")
                if v[0] in "cp" and (twin, k) in outs_by:
                    row[f"{k}_vs_gpu_diff"] = round((ys[0] != outs_by[(twin, k)]).float().mean().item(), 7)
                d = ys[0].float() - ref
                row[f"{k}_maxrel"] = round((d.abs().max() / ref.abs().max()).item(), 5)
                row[f"{k}_rmsrel"] = round((d.norm() / ref.norm()).item(), 5)
                # per-row relative error (what a layer's residual add sees)
                row[f"{k}_rowrel_p99"] = round(torch.quantile((d.norm(dim=1) / ref.norm(dim=1)), 0.99).item(), 5)
            x = data["gauss"]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    har(x)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier(group=cpu)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                outs = [har(x) for _ in range(10)]
            gr.replay(); torch.cuda.synchronize()
            eager = har(x); torch.cuda.synchronize()
            row["graph_equal"] = all(torch.equal(o, eager) for o in outs)
            row["ranks_equal"] = ok_rank
            row["graph_us"] = round(tm(lambda: gr.replay()) / 10, 1)
            row["err_code"] = har.error()
            o_, i_ = pcie_bytes(T * H, NG, har.bits, har.red_bits, G, har.cpu_on, har.cpu_ownq)
            o4, i4 = pcie_bytes(T * H, 4, har.bits, har.red_bits, G, har.cpu_on, har.cpu_ownq)
            if rank == 0 and har.cpu_stats() is not None:
                row["cpu_units"], row["cpu_spins"] = har.cpu_stats()
                row["cpu_simd"] = har.cpu["simd"]
            row.update(out_kb=round(o_ / 1024), in_kb=round(i_ / 1024), w4_out_kb=round(o4 / 1024), w4_in_kb=round(i4 / 1024))
            res.append(row)
            del gr, outs
    if rank == 0:
        info = dict(platform=1, nproc=os.cpu_count(), affinity=len(os.sched_getaffinity(0)),
                    cpu_threads=os.environ.get("ARC_HOST_AR_CPU_THREADS", "8"), cpu_k=os.environ.get("ARC_HOST_AR_CPU_K", "4"),
                    emu=os.environ.get("ARC_LINK_EMU_X", "1"), tile=os.environ.get("ARC_HOST_AR_TILE", "2048"))
        try:
            fl = open("/proc/cpuinfo").read()
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in fl.splitlines() if l.startswith("model name"))
            info["avx512f"] = " avx512f" in fl
        except Exception:
            pass
        res.insert(0, info)
        qu.put(res)
    dist.barrier(group=cpu)
    qu.close(); qu.join_thread()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    sk = socket.socket(); sk.bind(("127.0.0.1", 0)); os.environ["MASTER_PORT"] = str(sk.getsockname()[1]); sk.close()
    ctx = mp.get_context("spawn"); qu = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, qu, os.getpid()), daemon=True) for i in range(NG)]
    [p.start() for p in ps]
    import queue
    t0, r = time.time(), None
    try:
        while time.time() - t0 < DEADLINE:
            try:
                r = qu.get(timeout=2)
                break
            except queue.Empty:
                if not all(p.is_alive() for p in ps):
                    print("bench_host_ar_bits FAILED: a worker exited", [p.exitcode for p in ps], file=sys.stderr)
                    break
        for p in ps:
            p.join(timeout=20 if r is not None else 1)
    finally:
        for p in ps:
            if p.is_alive():
                p.kill()
    if r is None:
        sys.exit(1)
    for row in r:
        print(json.dumps(row), flush=True)
