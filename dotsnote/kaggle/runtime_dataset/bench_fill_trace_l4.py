"""Offload row-cache fill under a serving-like synthetic access stream, GPU kernel time only (CUDA-graph replay).
Each decode call reads N entries drawn from a working set of U rows that drifts by NEW random rows per call (no real
traces). L layers each own an R-row registered host buffer (shared by the processes, like the TP ranks), so the host
footprint matches serving (13 x 7.9M rows x 576 B = 59 GB) or not (L=1).
  python dots3/bench_fill_trace_l4.py NPROC L C [R=7895040] [N=344064] [U=116000] [NEW=16000]
Prints 'filltrace procs P L C <us/layer-call> misses <m> spill <s>' (slowest process)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)


def child(gpu, L, C, R, N, U, NEW, start_at):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from arcstats import sparse_offload as so
    so.CACHE_ROWS = C
    dev = torch.device("cuda")
    los = [so.LayerOffload(R, dev, key=f"fbench{i}") for i in range(L)]
    g = torch.Generator(device=dev).manual_seed(gpu)
    ws = torch.randperm(R, device=dev, generator=g)[:U]
    calls = []
    for t in range(96):   # drifting working set; every layer sees the same stream
        k = torch.randperm(U, device=dev, generator=g)[:NEW]
        ws = ws.clone()
        ws[k] = torch.randint(0, R, (NEW,), device=dev, generator=g)
        calls.append(ws[torch.randint(0, U, (N,), device=dev, generator=g)].contiguous())
    for x in calls[:56]:   # warm the caches into steady state
        for lo in los:
            lo._sa_prefill(x)
    mcs, scs = [], []
    for x in calls[56:64]:
        for j, lo in enumerate(los):
            rb = lo._sa_prefill(x)
            if j == 0:
                mcs.append(int(lo.mcnt)); scs.append(int(rb["scnt"]) if rb else 0)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):   # capture warm-up: repeats call 63 (all hits, state unchanged)
        for lo in los:
            lo._sa_prefill(calls[63])
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graphs = []
    for x in calls[64:]:   # captured, not run: replay continues the stream from call 63
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for lo in los:
                lo._sa_prefill(x)
        graphs.append(gr)
    torch.cuda.synchronize()
    while time.time() < start_at:
        time.sleep(0.01)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for gr in graphs:
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / (len(graphs) * L)
    print(f"R {us:.1f} {sum(mcs) / len(mcs):.0f} {sum(scs) / len(scs):.0f}", flush=True)


def main():
    if sys.argv[1] == "--child":
        a = sys.argv[2:]
        child(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), float(a[7]))
        return
    P, L, C = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 7895040
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 344064
    U = int(sys.argv[6]) if len(sys.argv) > 6 else 116000
    NEW = int(sys.argv[7]) if len(sys.argv) > 7 else 16000
    gpus = os.environ.get("BENCH_GPUS", "0,1,2,3").split(",")[:P]
    start_at = time.time() + float(os.environ.get("BENCH_FILL_WARM_S", "150"))
    ps = [subprocess.Popen([sys.executable, __file__, "--child", g, str(L), str(C), str(R), str(N), str(U), str(NEW),
                            str(start_at)], stdout=subprocess.PIPE, text=True) for g in gpus]
    res = []
    for p in ps:
        o, _ = p.communicate()
        res += [l.split()[1:] for l in o.splitlines() if l.startswith("R ")]
    if res:
        w = max(res, key=lambda r: float(r[0]))
        print(f"filltrace procs {P} L {L} C {C} R {R} {w[0]} us/layer-call misses {w[1]} spill {w[2]} "
              f"(all: {' '.join(r[0] for r in res)})", flush=True)
    else:
        print(f"filltrace procs {P} L {L} C {C} FAILED", flush=True)


if __name__ == "__main__":
    main()
