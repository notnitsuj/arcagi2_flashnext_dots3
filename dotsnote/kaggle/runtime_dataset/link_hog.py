"""PCIe link hog: emulate a narrower / shared GPU<->host link (Kaggle 4xL4: Gen3 x16, GPU pairs behind one uplink) on a
machine with a wider link, so host-link mitigations (host all-reduce, row-cache fills, all-gathers) can be A/B'd offline.
the design notes.

One process per GPU keeps rate-paced copy-engine DMA traffic between pinned host memory and that GPU in flight
(H2D and D2H on their own streams/threads). The victim (our server / benches) then sees link capacity minus the hog's
rate. Plain ctypes + libcudart, no torch: the CUDA context is the only GPU memory it takes (measure: --report shows it).

  python dots3/link_hog.py --gpus 0,1,2,3 --h2d 18 --d2h 18 [--chunk-mb 2] [--seconds 0] [--report 10]
    --h2d/--d2h  target GB/s per GPU per direction (0 = off; 'max' = unpaced, back-to-back copies)
    --chunk-mb   copy size; smaller = finer interleaving with the victim's traffic (more launch overhead)
    --streams    copies in flight per direction (default 2: the link never idles between paced copies)
    --seconds    0 = until killed (dies with its parent: PR_SET_PDEATHSIG)
Prints 'hog gpu <g> h2d <GB/s> d2h <GB/s>' achieved rates every --report s on stderr, and 'hog ready' on stdout once
every GPU's copy loops run. Spawns one child process per GPU (separate contexts, as on separate GPUs).
"""
import argparse, ctypes, os, signal, sys, threading, time


def _cudart():
    for so in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(so)
        except OSError:
            pass
    import glob
    for p in sys.path:   # pip wheel (nvidia/cu13/lib or nvidia/cuda_runtime/lib)
        for f in glob.glob(os.path.join(p, "nvidia", "*", "lib", "libcudart.so.1*")):
            try:
                return ctypes.CDLL(f)
            except OSError:
                pass
    raise OSError("libcudart not found")


def _die_with_parent():
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: cuda error {rc}")


def run_gpu(dev, h2d, d2h, chunk, streams, seconds, report):
    _die_with_parent()
    rt = _cudart()
    _ck(rt.cudaSetDevice(ctypes.c_int(dev)), "cudaSetDevice")
    rt.cudaFree(ctypes.c_void_p(0))   # create the context
    stats = {"h2d": 0, "d2h": 0}
    stop = time.time() + seconds if seconds > 0 else float("inf")

    def loop(kind, rate):
        # one thread per direction; `streams` copies in flight, each re-issued when its pacing slot comes up
        # (the runtime's current device is per thread: without this, new threads default to device 0)
        _ck(rt.cudaSetDevice(ctypes.c_int(dev)), "cudaSetDevice")
        hbuf, dbuf = ctypes.c_void_p(), ctypes.c_void_p()
        _ck(rt.cudaMallocHost(ctypes.byref(hbuf), ctypes.c_size_t(chunk * streams)), "cudaMallocHost")
        _ck(rt.cudaMalloc(ctypes.byref(dbuf), ctypes.c_size_t(chunk * streams)), "cudaMalloc")
        ss = []
        for _ in range(streams):
            s = ctypes.c_void_p()
            _ck(rt.cudaStreamCreateWithFlags(ctypes.byref(s), ctypes.c_uint(1)), "cudaStreamCreate")   # non-blocking
            ss.append(s)
        k = 1 if kind == "h2d" else 2     # cudaMemcpyHostToDevice / DeviceToHost
        t0 = time.time()
        n = 0
        while time.time() < stop:
            i = n % streams
            rt.cudaStreamSynchronize(ss[i])      # ctypes drops the GIL: the other direction keeps going
            if rate > 0:                          # paced: copy n may start at t0 + n * chunk / rate
                dt = t0 + n * chunk / rate - time.time()
                if dt > 0:
                    time.sleep(dt)
                elif dt < -0.05:                  # fell behind (link saturated): do not burst to catch up
                    t0 -= dt + 0.05
            off = i * chunk
            dst, src = (dbuf.value + off, hbuf.value + off) if k == 1 else (hbuf.value + off, dbuf.value + off)
            _ck(rt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(chunk), ctypes.c_int(k),
                                   ss[i]), "cudaMemcpyAsync")
            stats[kind] += chunk
            n += 1

    ths = []
    for kind, r in (("h2d", h2d), ("d2h", d2h)):
        if r is not None:
            th = threading.Thread(target=loop, args=(kind, r), daemon=True)
            th.start()
            ths.append(th)
    print(f"hog ready gpu {dev}", flush=True)
    last, lt = dict(stats), time.time()
    rt.cudaSetDevice(ctypes.c_int(dev))
    while time.time() < stop and any(t.is_alive() for t in ths):
        time.sleep(min(report, max(0.1, stop - time.time())) if report > 0 else 1.0)
        if report > 0:
            now = time.time()
            free, tot = ctypes.c_size_t(), ctypes.c_size_t()
            rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(tot))
            print(f"hog gpu {dev} " + " ".join(f"{k} {(stats[k] - last[k]) / (now - lt) / 1e9:.1f}" for k in stats) +
                  f" GB/s  (gpu free {free.value >> 20} MiB)", file=sys.stderr, flush=True)
            last, lt = dict(stats), now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--h2d", default="0")
    ap.add_argument("--d2h", default="0")
    ap.add_argument("--chunk-mb", type=float, default=2.0)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--report", type=float, default=0)
    a = ap.parse_args()
    rate = lambda s: None if s in ("0", "0.0", "") else (0.0 if s == "max" else float(s) * 1e9)
    chunk = int(a.chunk_mb * (1 << 20))
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    if len(gpus) == 1:
        run_gpu(gpus[0], rate(a.h2d), rate(a.d2h), chunk, a.streams, a.seconds, a.report)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=run_gpu, args=(g, rate(a.h2d), rate(a.d2h), chunk, a.streams, a.seconds, a.report),
                      daemon=True) for g in gpus]
    [p.start() for p in ps]
    _die_with_parent()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        [p.join() for p in ps]
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
