"""Host-side throughput of the CPU-reduced all-reduce reducer (arcstats csrc/host_arc.c, ARC_HOST_AR_CPU) without GPUs:
one thread reduces every unit of one TP4 all-reduce of T tokens x 5120 (random codes, real layout), per SIMD level, and
checks that the AVX-512 / AVX2 / scalar paths write identical reduced codes.
  python dots3/bench_host_arc_cpu.py [tokens=256,1024] [bits=8,6]
Prints one json line per (simd, T, bits): us per all-reduce on one thread and Gelem/s."""
import ctypes, json, os, sys, time
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
so = os.environ.get("ARC_HOST_ARC_SO")
if not so:
    try:
        import arcstats
        so = os.path.join(os.path.dirname(arcstats.__file__), "host_arc.so")
    except Exception:
        so = os.path.join(here, "arcstats_pkg", "arcstats", "host_arc.so")
L = ctypes.CDLL(so)
L.arc_create.restype = ctypes.c_void_p
L.arc_create.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 6 + [ctypes.c_int64] * 8 + [ctypes.c_void_p, ctypes.c_int]
L.arc_reduce_all.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
L.arc_simd.argtypes = [ctypes.c_void_p]
TOKS = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "256,1024").split(",")]
BITS = [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "8,6").split(",")]
W, NB, K, G, FS, H = 4, 32, 4, 128, 16, 5120
maxn = max(TOKS) * H
slot, own_slot = 2 * maxn, 2 * maxn // W
gf_off, cf_off = 0, W * NB * K * FS * 4
in_off = 1 << 20
own_off = in_off + W * slot
red_off = own_off + W * own_slot
size = red_off + slot
rng = np.random.default_rng(0)
base = rng.integers(0, 255, size, dtype=np.uint8)
for r in range(W):
    o = in_off + r * slot
    base[o + maxn: o + maxn + 4 * (maxn // G)].view(np.float32)[:] = rng.random(maxn // G, dtype=np.float32) * 0.01
base[own_off:own_off + W * own_slot].view(np.uint16)[:] = (
    rng.standard_normal(W * own_slot // 2).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
reds = {}
info = {"cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "nproc": os.cpu_count()}
print(json.dumps(info), flush=True)
for simd in (2, 1, 0):
    buf = base.copy()
    c = L.arc_create(buf.ctypes.data, W, NB, K, G, FS, 1, maxn, slot, own_slot, in_off, own_off, red_off, gf_off,
                     cf_off, None, simd)
    got = L.arc_simd(c)
    if got != simd:
        print(json.dumps(dict(simd=simd, available=False)), flush=True)
        continue
    for T in TOKS:
        q = T * H // W
        for B in BITS:
            n = 1 if simd == 0 else 5
            L.arc_reduce_all(c, q, B, B, 0)
            if T == TOKS[0]:
                reds[(simd, B)] = buf[red_off:red_off + slot].copy()
            t0 = time.perf_counter()
            for _ in range(n):
                L.arc_reduce_all(c, q, B, B, 0)
            dt = (time.perf_counter() - t0) / n
            print(json.dumps(dict(simd=simd, T=T, bits=B, us=round(dt * 1e6), gelem_s=round(T * H / dt / 1e9, 3))),
                  flush=True)
for B in BITS:
    ok = [s for s in (2, 1) if (s, B) in reds and (0, B) in reds]
    print(json.dumps(dict(bits=B, equal_to_scalar={s: bool(np.array_equal(reds[(s, B)], reds[(0, B)])) for s in ok})),
          flush=True)
