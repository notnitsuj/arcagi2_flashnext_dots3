"""Kaggle 4xL4 runner for the Dots3 L4 build (offline, save runs). Driven by mode.json next to this file.

Phases: unpack bundled CPython 3.12 env -> install our plugin / patched vLLM files / CUDA 13.0 toolkit ->
hardware microbench -> vLLM server (exact L4 config from mode.json) -> workload -> logs in /kaggle/working.
Workloads:
  "sweep": synthetic decode-throughput sweep over (batch, prompt length): random prompt token ids, fixed
           ignore_eos generations; metrics polled every 5 s (works with ARC_LOWBIT_RANDOM=1 expert weights)
  "arc":   ARC direct-output attempts on holdout tasks (needs real weights)
"""
import asyncio, glob, json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path


def _prefetch_main(spec_path):
    """`run_l4.py --prefetch SPEC.json` (MODE["prefetch"], default off): read files into the page cache with many
    parallel pread streams, in the given order, so the env tar / checkpoint reads that follow hit RAM.
    /kaggle/input is NFS: one sequential reader (tar, torch.load) gets ~65-90 MB/s there, vLLM's 8-thread safetensors
    prefetch ~500 MB/s. Stdlib only (runs under the image's python before the env exists).
    SPEC: {"files": [...], "threads": 16, "block_mb": 16, "max_gib": 90, "log": path, "nice": 5, "timeout_s": 3600}"""
    import threading
    spec = json.load(open(spec_path))
    logf = open(spec["log"], "a")
    t0 = time.time()

    plock = threading.Lock()   # one writer at a time: concurrent TextIOWrapper writes from 32 copier threads lost lines

    def plog(*a):
        with plock:
            logf.write(f"[{time.time() - t0:7.1f}s] " + " ".join(str(x) for x in a) + "\n"); logf.flush()
    try:
        os.nice(int(spec.get("nice", 5)))
    except OSError:
        pass
    blk = int(spec.get("block_mb", 16)) << 20
    budget = float(spec.get("max_gib", 90)) * 2**30
    files, total = [], 0
    for f in spec["files"]:
        try:
            n = os.path.getsize(f)
        except OSError as e:
            plog("skip (stat)", f, repr(e)[:80]); continue
        if total + n > budget:
            plog("skip (over max_gib)", f, f"{n / 2**30:.2f} GiB"); continue
        files.append((f, n)); total += n
    plog(f"prefetch {len(files)} files {total / 2**30:.2f} GiB, {spec.get('threads', 16)} threads x {blk >> 20} MB")
    chunks = [(i, off) for i, (f, n) in enumerate(files) for off in range(0, n, blk)]
    left = [sum(1 for _ in range(0, n, blk)) for _, n in files]
    lock, pos, done_b = threading.Lock(), [0], [0]
    deadline = t0 + float(spec.get("timeout_s", 3600))
    if spec.get("deadline_epoch"):   # MODE["startup_guard"]: copy for as long as the session budget allows
        deadline = float(spec["deadline_epoch"])

    def worker():
        buf = bytearray(blk); mv = memoryview(buf); fds = {}
        while time.time() < deadline:
            with lock:
                if pos[0] >= len(chunks):
                    break
                i, off = chunks[pos[0]]; pos[0] += 1
            try:
                if i not in fds:
                    fds[i] = os.open(files[i][0], os.O_RDONLY)
                got = os.preadv(fds[i], [mv], off)
            except OSError as e:
                plog("read error", files[i][0], off, repr(e)[:80]); got = 0
            with lock:
                done_b[0] += got; left[i] -= 1
                if left[i] == 0:
                    plog(f"done {os.path.basename(files[i][0])} {files[i][1] / 2**30:.2f} GiB, "
                         f"cum {done_b[0] / 2**30:.1f} GiB at {done_b[0] / 2**20 / max(time.time() - t0, 1e-3):.0f} MB/s")
        for fd in fds.values():
            os.close(fd)
    th = [threading.Thread(target=worker, daemon=True) for _ in range(int(spec.get("threads", 16)))]
    for t in th:
        t.start()
    if spec.get("copy"):   # MODE["prefetch"]["stage"]: local checkpoint copies, concurrently
        try:
            (_stage_copy_robust if spec.get("robust") else _stage_copy)(spec, plog, deadline)
        except Exception as e:
            plog("stage copy error", repr(e)[:200])
        finally:
            try:
                os.unlink(spec["stage_marker"])
            except OSError:
                pass
    for t in th:
        t.join()
    plog(f"prefetch finished: {done_b[0] / 2**30:.2f} GiB in {time.time() - t0:.0f} s"
         + (" (TIMEOUT)" if pos[0] < len(chunks) else ""))


def _stage_copy(spec, plog, deadline):
    """Copy each [src, dst] of spec["copy"] to local disk once (dst.part, then an atomic rename over the dst symlink that
    points at src), in list order, with spec["stage_threads"] workers on spec["seg_mb"] segments that are each read
    sequentially through one fd (a FUSE/GCSFuse mount drops a file's page cache on every open() and treats strided
    reads as random, so the page-cache prefetch above does not help there and every TP rank re-read every expert file
    from the mount: v24n save run, 1080 s to ready). Source pages are dropped after use (no double page cache).
    Same bytes: the size is checked; any failure leaves the symlink, i.e. the old read path. Stdlib only."""
    import threading
    t0 = time.time()
    blk, seg = int(spec.get("block_mb", 16)) << 20, int(spec.get("seg_mb", 256)) << 20
    rate = float(spec.get("throttle_mbps", 0)) * 2**20      # emulation only (test rig): cap the aggregate read rate
    files = []
    for src, dst in spec["copy"]:
        try:
            n = os.path.getsize(src)
            with open(dst + ".part", "wb") as f:
                f.truncate(n)
            files.append([src, dst, n, 0, False])          # src, dst, size, bytes done, failed
        except OSError as e:
            plog("stage skip", src, repr(e)[:80])
    segs = [(i, off) for i, (_, _, n, _, _) in enumerate(files) for off in range(0, n, seg)]
    left = [sum(1 for _ in range(0, f[2], seg)) for f in files]
    lock, pos, tot, tb = threading.Lock(), [0], [0], [time.time(), 0.0]
    plog(f"stage {len(files)} files {sum(f[2] for f in files) / 2**30:.2f} GiB -> {os.path.dirname(spec['copy'][0][1])}, "
         f"{spec.get('stage_threads', 8)} threads x {seg >> 20} MB segments" + (f", throttle {rate / 2**20:.0f} MB/s" if rate else ""))

    def worker():
        buf = bytearray(blk); mv = memoryview(buf)
        while time.time() < deadline:
            with lock:
                if pos[0] >= len(segs):
                    return
                i, off = segs[pos[0]]; pos[0] += 1
            src, dst, n, _, bad = files[i]
            got = 0
            if not bad:
                try:
                    fi = os.open(src, os.O_RDONLY); fo = os.open(dst + ".part", os.O_WRONLY)
                    try:
                        o, end = off, min(n, off + seg)
                        while o < end:
                            k = os.preadv(fi, [mv[:min(blk, end - o)]], o)
                            if k <= 0:
                                raise OSError(f"short read at {o}")
                            os.pwrite(fo, mv[:k], o); o += k; got += k
                            if rate:
                                with lock:
                                    tb[1] += k
                                    wait = tb[0] + tb[1] / rate - time.time()
                                if wait > 0:
                                    time.sleep(wait)
                        try:
                            os.posix_fadvise(fi, off, end - off, os.POSIX_FADV_DONTNEED)
                        except (OSError, AttributeError):
                            pass
                    finally:
                        os.close(fi); os.close(fo)
                except OSError as e:
                    plog("stage read error", src, off, repr(e)[:80]); files[i][4] = True
            with lock:
                tot[0] += got; files[i][3] += got; left[i] -= 1
                last = left[i] == 0
            if last:
                ok = not files[i][4] and files[i][3] == n
                try:
                    if ok:
                        os.replace(dst + ".part", dst)   # the symlink to the mount becomes the local file
                    else:
                        os.unlink(dst + ".part")
                except OSError as e:
                    ok = False; plog("stage rename error", dst, repr(e)[:80])
                plog(f"staged {os.path.basename(dst)} {n / 2**30:.2f} GiB" if ok else f"stage FAILED {os.path.basename(dst)} (mount read)",
                     f"cum {tot[0] / 2**30:.1f} GiB at {tot[0] / 2**20 / max(time.time() - t0, 1e-3):.0f} MB/s")
    th = [threading.Thread(target=worker, daemon=True) for _ in range(int(spec.get("stage_threads", 8)))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    plog(f"stage finished: {tot[0] / 2**30:.2f} GiB in {time.time() - t0:.0f} s"
         + (" (TIMEOUT)" if pos[0] < len(segs) else ""))


def _stage_copy_robust(spec, plog, deadline):
    """MODE["prefetch"]["robust"] (default off): _stage_copy for a degraded input mount.
    Same result (dst.part filled with the source bytes, atomic rename over the symlink, a failed file keeps its symlink),
    but: (1) segments are taken strictly lowest-first, so files complete in list order (env tar first when staged);
    (2) a failed segment read is retried up to max_attempts times with backoff instead of failing the file at once;
    (3) a segment with no byte progress for stall_s is re-read by a second worker (hedged read; both write the same
    bytes at the same offsets, the first to finish counts), so one read that hangs for minutes does not hold its file;
    (4) the coordinator never joins a worker (a read stuck in the kernel cannot block the stage end / marker removal);
    (5) progress (bytes, time) is written to spec["progress"] every second for run_l4's startup watchdog.
    Emulation only: throttle_mbps (aggregate rate cap), stall_emul {"seg": k, "s": S} (first read of segment k sleeps S)."""
    import threading
    t0 = time.time()
    blk, seg = int(spec.get("block_mb", 8)) << 20, int(spec.get("seg_mb", 64)) << 20
    rate = float(spec.get("throttle_mbps", 0)) * 2**20
    stall_s, max_att = float(spec.get("stall_s", 120)), int(spec.get("max_attempts", 8))
    emul = spec.get("stall_emul") or {}
    files = []
    for src, dst in spec["copy"]:
        try:
            n = os.path.getsize(src)
            with open(dst + ".part", "wb") as f:
                f.truncate(n)
            files.append({"src": src, "dst": dst, "n": n, "left": 0, "bad": False, "t": None})
        except OSError as e:
            plog("stage skip", src, repr(e)[:80])
    # barrier files (spec["barrier"]: the env / CUDA / Triton tars with stage_env): everything after a barrier file waits
    # until all its segments are read, and they use small segments (barrier_seg_mb) so the tail is short. Without it the
    # 32 x 64 MB in-flight window let lowbit segments share the bandwidth with the CUDA tar: at 10 MB/s the CUDA tar was
    # local 164 s after the env tar instead of ~62 s (10 MB/s emulation)
    bset = set(spec.get("barrier") or [])
    bseg = int(spec.get("barrier_seg_mb", 16)) << 20
    for f in files:
        f["bar"] = f["src"] in bset
    segs = [(i, off, min(f["n"], off + (bseg if f["bar"] else seg))) for i, f in enumerate(files)
            for off in range(0, f["n"], bseg if f["bar"] else seg)]
    bars = [i for i, f in enumerate(files) if f["bar"]]
    for i, _, _ in segs:
        files[i]["left"] += 1
    S = len(segs)
    st = [0] * S            # 0 pending, 1 in flight, 2 done
    att = [0] * S           # attempts started
    act = [0] * S           # attempts running
    prog = [0.0] * S        # last byte progress
    nb = [0.0] * S          # not before (retry backoff)
    lock, tot, tb, nd, ncl = threading.Lock(), [0], [time.time(), 0.0], [0], [0]
    plog(f"stage(robust) {len(files)} files {sum(f['n'] for f in files) / 2**30:.2f} GiB, {spec.get('stage_threads', 32)} threads"
         f" x {seg >> 20} MB segments, {blk >> 20} MB reads, hedge after {stall_s:.0f} s, {max_att} attempts"
         + (f", throttle {rate / 2**20:.0f} MB/s" if rate else "") + (f", stall_emul {emul}" if emul else ""))

    def finish(j, ok):   # under lock: segment j done (ok) or given up; returns the file index if the file is complete
        i = segs[j][0]
        st[j] = 2; nd[0] += 1
        if not ok:
            files[i]["bad"] = True
        files[i]["left"] -= 1
        return i if files[i]["left"] == 0 else None

    def close_file(i):
        f = files[i]
        ok = not f["bad"]
        try:
            if ok:
                os.replace(f["dst"] + ".part", f["dst"])
            else:
                os.unlink(f["dst"] + ".part")
        except OSError as e:
            ok = False; plog("stage rename error", f["dst"], repr(e)[:80])
        plog(f"staged {os.path.basename(f['dst'])} {f['n'] / 2**30:.2f} GiB" if ok else
             f"stage FAILED {os.path.basename(f['dst'])} (mount read)",
             f"cum {tot[0] / 2**30:.1f} GiB at {tot[0] / 2**20 / max(time.time() - t0, 1e-3):.0f} MB/s")
        with lock:
            ncl[0] += 1

    bstall = float(spec.get("barrier_stall_s", 30))

    def hedge(j, now, why):
        att[j] += 1; act[j] += 1; prog[j] = now
        plog(f"stage hedge {os.path.basename(files[segs[j][0]]['dst'])} @{segs[j][1] >> 20} MB "
             f"(no progress for {why:.0f} s, attempt {att[j]})")
        return ("read", j, att[j])

    def pick():
        now = time.time()
        gate = next((k for k in bars if files[k]["left"] > 0), None)   # first barrier file not fully read
        hedgeable = lambda j, lim: st[j] == 1 and act[j] < 3 and now - prog[j] > lim and att[j] < max_att
        for j in range(S):   # in order: a stalled earlier segment is re-read before new work is started
            if gate is not None and segs[j][0] > gate:
                break
            if not files[segs[j][0]]["bar"] and hedgeable(j, stall_s):
                return hedge(j, now, stall_s)
            if st[j] == 0 and now >= nb[j]:
                if files[segs[j][0]]["bad"]:          # file already failed: skip its remaining segments
                    k = finish(j, False)
                    if k is not None:
                        return ("close", k)
                    continue
                st[j] = 1; att[j] += 1; act[j] += 1; prog[j] = now
                return ("read", j, att[j])
        for j in range(S):   # nothing new to start (end of the list, or a barrier file's tail): hedge sooner
            lim = bstall if files[segs[j][0]]["bar"] else stall_s
            if hedgeable(j, lim):
                return hedge(j, now, lim)
        return None

    def worker():
        buf = bytearray(blk); mv = memoryview(buf)
        while time.time() < deadline:
            with lock:
                if nd[0] >= S:
                    return
                job = pick()
            if job is None:
                time.sleep(0.5); continue
            if job[0] == "close":
                close_file(job[1]); continue
            j = job[1]
            i, off, end = segs[j]
            f = files[i]
            first = job[2] == 1
            ok, got = True, 0
            try:
                fi = os.open(f["src"], os.O_RDONLY); fo = os.open(f["dst"] + ".part", os.O_WRONLY)
                try:
                    if first and emul and int(emul.get("seg", -1)) == j:
                        time.sleep(float(emul.get("s", 300)))   # emulated hung read (first attempt only)
                    o = off
                    while o < end and st[j] != 2 and time.time() < deadline:
                        k = os.preadv(fi, [mv[:min(blk, end - o)]], o)
                        if k <= 0:
                            raise OSError(f"short read at {o}")
                        if st[j] == 2:
                            break
                        os.pwrite(fo, mv[:k], o); o += k; got += k
                        with lock:
                            prog[j] = time.time(); tot[0] += k
                            if rate:
                                tb[1] += k
                                wait = tb[0] + tb[1] / rate - time.time()
                                if wait < -1.0:   # no burst credit beyond 1 s after idle time (a real mount has none)
                                    tb[0] -= wait + 1.0; wait = -1.0
                            else:
                                wait = 0
                        if wait > 0:
                            time.sleep(wait)
                    ok = o >= end
                    try:
                        os.posix_fadvise(fi, off, end - off, os.POSIX_FADV_DONTNEED)
                    except (OSError, AttributeError):
                        pass
                finally:
                    os.close(fi); os.close(fo)
            except OSError as e:
                ok = False
                plog("stage read error", os.path.basename(f["src"]), off, f"attempt {att[j]}", repr(e)[:80])
            done_file = None
            with lock:
                act[j] -= 1
                if st[j] != 2:
                    if ok:
                        done_file = finish(j, True)
                    elif act[j] == 0:
                        if att[j] >= max_att:
                            done_file = finish(j, False)
                        else:   # retry later (backoff 2, 4, 8 .. 60 s)
                            st[j] = 0; nb[j] = time.time() + min(60.0, 2.0 ** att[j])
            if done_file is not None:
                close_file(done_file)

    th = [threading.Thread(target=worker, daemon=True) for _ in range(int(spec.get("stage_threads", 32)))]
    for t in th:
        t.start()
    pf = spec.get("progress")
    # until every file is renamed / dropped (not just every segment read: the last rename ran after the process exit)
    while ncl[0] < len(files) and time.time() < deadline and any(t.is_alive() for t in th):
        if pf:
            try:
                with open(pf + ".tmp", "w") as fp:
                    fp.write(f"{tot[0]} {time.time():.1f} {nd[0]} {S}\n")
                os.replace(pf + ".tmp", pf)
            except OSError:
                pass
        time.sleep(1.0)
    if pf:
        try:
            with open(pf, "w") as fp:
                fp.write(f"{tot[0]} {time.time():.1f} {nd[0]} {S} end\n")
        except OSError:
            pass
    for f in files:   # timeout: incomplete files keep their symlink; drop the partial local copy
        if f["left"] > 0:
            try:
                os.unlink(f["dst"] + ".part")
            except OSError:
                pass
    plog(f"stage finished: {tot[0] / 2**30:.2f} GiB in {time.time() - t0:.0f} s"
         + ("" if ncl[0] >= len(files) else f" (TIMEOUT: {S - nd[0]} segments left, their files keep the mount symlink)"))


if __name__ == "__main__" and sys.argv[1:2] == ["--prefetch"]:
    _prefetch_main(sys.argv[2])
    sys.exit(0)

T0 = time.time()
os.environ.setdefault("ARC_T0", str(T0))   # run start for time-windowed server A/B (ARC_OFFLOAD_AB)
W = Path(os.environ.get("ARC_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
IN = os.environ.get("ARC_INPUT", "/kaggle/input")
LOG = open(W / "run_log.txt", "a")


def log(*a):
    s = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


_FIND = {"on": False, "idx": None}   # MODE["startup_guard"]["find_index"]: one input walk instead of one per find()


def _input_index():
    """all files under IN (one os.walk, symlinked dirs followed as glob ** does; hidden names skipped as glob does),
    cached in W/input_index.json for the re-exec'd process. On the GCSFuse mount every recursive glob cost 6-12 s
    (v26 save run: 83 s of globs before the prefetcher started)."""
    if _FIND["idx"] is None:
        cf = W / "input_index.json"
        try:
            c = json.load(open(cf))
            if c.get("in") == IN:
                _FIND["idx"] = c["files"]
        except Exception:
            pass
    if _FIND["idx"] is None:
        out = []
        for root, dns, fns in os.walk(IN, followlinks=True):
            dns[:] = sorted(d for d in dns if not d.startswith("."))
            out += [os.path.join(root, f) for f in fns if not f.startswith(".")]
        _FIND["idx"] = sorted(out)
        try:
            json.dump({"in": IN, "files": _FIND["idx"]}, open(cf, "w"))
        except OSError:
            pass
    return _FIND["idx"]


def find(pat):
    if _FIND["on"] and pat.startswith(IN + "/**/"):   # same answer as the glob below: first sorted match
        import fnmatch
        rest = pat[len(IN) + 4:]
        for p in _input_index():
            rel = p[len(IN) + 1:]
            if (fnmatch.fnmatchcase(rel, rest) or fnmatch.fnmatchcase(rel, "*/" + rest)) and os.path.exists(p):
                return p
    r = sorted(glob.glob(pat, recursive=True))
    if not r:
        raise FileNotFoundError(pat)
    return r[0]


def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def run_bench(cmd, env, timeout, cwd=None):
    """subprocess.run(cmd, capture_output=True, text=True) for (multi-process) benches, hang-safe: the bench runs in its
    own session and on timeout the WHOLE process group is killed (subprocess.run kills only the parent, so spawned
    workers lived on holding the GPUs, the c10d port and the shm: run 45). Partial output is logged, not discarded.
    Returns a CompletedProcess (returncode -9 after a timeout)."""
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass
        out, err = p.communicate()
        rc = -9
        log("bench TIMEOUT", os.path.basename(str(cmd[1] if len(cmd) > 1 else cmd[0])), f"after {timeout} s, stdout tail:",
            out[-1500:], "| stderr tail:", err[-2500:])
    finally:
        try:   # leftovers of a parent that exited (e.g. workers stuck in teardown)
            os.killpg(p.pid, 9)
        except OSError:
            pass
    return subprocess.CompletedProcess(cmd, rc, out, err)


# this file's own directory is what the glob below returns in every launcher (run_submit / the test harness run exactly
# sorted(glob(IN/**/run_l4.py))[0]); skipping the recursive walk saves 6-12 s on a FUSE input mount
_HERE = Path(os.path.abspath(globals().get("__file__", "") or ".")).parent
code = _HERE if (globals().get("__file__") and str(_HERE).startswith(IN + "/") and (_HERE / "run_l4.py").exists()) \
    else Path(find(IN + "/**/run_l4.py")).parent
MODE = json.load(open(code / os.environ.get("ARC_MODE_FILE", "mode.json")))   # env survives the re-exec
# Competition submissions: the scored rerun (hidden test set) sets KAGGLE_IS_COMPETITION_RERUN; the interactive save run
# applies MODE["save_overrides"] instead (a short real-weights smoke test that proves the stack fits and solves).
RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
if not RERUN and MODE.get("save_overrides"):
    MODE.update(MODE["save_overrides"])

# ---------------------------------------------------------------- storage-degradation startup guard (default off)
# MODE["startup_guard"]: every startup time counts from the session anchor S0 (container
# init / first run_l4 start, exported as ARC_S0 so it survives the re-exec), not from this process's T0 (reset by the
# re-exec, so a slow env-tar unpack was invisible to the solver deadline). Keys:
#   anchor "session" (S0 = min(now, own start, pid 1 start) within anchor_max_age_s) | "t0" (first run_l4 start; tests)
#   deadline_abs_s   solver deadline S0 + this (and never later than T0 + deadline_total_s, so short save runs stay short)
#   kill_by_s        hard stop at S0 + this: kill every child, keep/repair submission.json, exit 0 (12 h net)
#   retries          server restarts allowed before it is first ready (death or startup hang), if >= retry_min_left_s left
#   hang_s           startup hang = no server-log growth and no staging progress for this long
#   solve_min_s      give up on the server (fallback submission, exit 0) when less than this would be left to solve
#   find_index       one walk of the inputs for every find()
#   emul_kill_s      tests only: [s after 1st server start, s after 2nd, ...] SIGKILL that server tree once
SG = MODE.get("startup_guard") or {}
_FIND["on"] = bool(SG.get("find_index"))


def _proc_start(pid):
    try:
        st = open(f"/proc/{pid}/stat").read().rpartition(")")[2].split()
        btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        return btime + int(st[19]) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


if SG and not os.environ.get("ARC_S0"):   # first (pre-re-exec) process only
    _now = time.time()
    _s0 = min(T0, _now)
    if SG.get("anchor", "session") == "session":
        for _pid in (os.getpid(), 1):
            _t = _proc_start(_pid)
            if _t is not None and _now - float(SG.get("anchor_max_age_s", 3600)) <= _t <= _now + 2:
                _s0 = min(_s0, _t)
    os.environ["ARC_S0"] = repr(_s0)
S0 = float(os.environ["ARC_S0"]) if SG and os.environ.get("ARC_S0") else None


def deadline_epoch():
    """solver deadline as an epoch: T0 + deadline_total_s, and with the guard also <= S0 + deadline_abs_s"""
    d = T0 + float(MODE.get("deadline_total_s", 41400))
    if S0 is not None:
        d = min(d, S0 + float(SG.get("deadline_abs_s", 41700)))
    return d


def _challenges_file():
    return find(IN + "/**/arc-agi_test_challenges.json")


def fallback_submission():
    """Valid submission.json (test input echoed as both attempts) so a crash anywhere still scores 0, not an error."""
    if MODE.get("workload") != "submit" or (W / "submission.json").exists():
        return
    try:
        f = _challenges_file()
        ch = json.load(open(f))
        json.dump({t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in v["test"]] for t, v in ch.items()},
                  open(W / "submission.json.tmp", "w"))
        os.replace(W / "submission.json.tmp", W / "submission.json")   # atomic: a kill never leaves a partial file
    except Exception as ex:
        print("fallback submission failed", repr(ex)[:200], flush=True)


def ensure_submission():
    """(guard) submission.json must parse and hold every test input of the challenges file with two grids; missing or
    malformed entries become the input echo (valid entries are kept). Atomic rewrite only if something changed."""
    if MODE.get("workload") != "submit":
        return "n/a"
    try:
        ch = json.load(open(_challenges_file()))
    except Exception as ex:
        return f"no challenges ({repr(ex)[:80]})"
    try:
        sub = json.load(open(W / "submission.json"))
        sub = sub if isinstance(sub, dict) else {}
    except Exception:
        sub = {}
    isg = lambda g: isinstance(g, list) and g and all(isinstance(r, list) and r and all(isinstance(v, int) for v in r)
                                                      for r in g)
    out, fixed = {}, 0
    for t, v in ch.items():
        cur = sub.get(t) if isinstance(sub.get(t), list) else []
        outs = []
        for k, x in enumerate(v["test"]):
            e = cur[k] if k < len(cur) and isinstance(cur[k], dict) else {}
            a1 = e.get("attempt_1") if isg(e.get("attempt_1")) else None
            a2 = e.get("attempt_2") if isg(e.get("attempt_2")) else None
            fixed += (a1 is None) + (a2 is None)
            outs.append({"attempt_1": a1 if a1 is not None else x["input"], "attempt_2": a2 if a2 is not None else x["input"]})
        out[t] = outs
    if fixed or set(sub) != set(out):
        json.dump(out, open(W / "submission.json.tmp", "w"))
        os.replace(W / "submission.json.tmp", W / "submission.json")
    return f"ok ({len(out)} tasks, {fixed} attempts filled with the input echo)"


fallback_submission()
PY = "/tmp/arc_py312/bin/python3.12"


def _descendants(root):
    """pids of every live descendant of root (from /proc ppid links)"""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                pp = int(open(f"/proc/{d}/stat").read().rpartition(")")[2].split()[1])
                kids.setdefault(pp, []).append(int(d))
            except Exception:
                pass
    out, todo = [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c); todo.append(c)
    return out


def _kill_all_children():
    n = 0
    for pid in _descendants(os.getpid()):
        try:
            os.kill(pid, 9); n += 1
        except OSError:
            pass
    return n


def _start_killby():
    """(guard) hard 12 h net: at S0 + kill_by_s SIGKILL every descendant (solver, server tree, prefetcher, tar), make
    submission.json valid and os._exit(0), whatever the main thread is blocked in. One daemon thread per process image
    (the pre-re-exec one dies with the exec; the re-exec'd process starts its own)."""
    import threading
    at = S0 + float(SG.get("kill_by_s", 42600))

    def fire():
        while time.time() < at:
            time.sleep(min(30.0, max(0.5, at - time.time())))
        try:
            log("kill-by reached (S0 + %.0f s): SIGKILLed %d processes" % (float(SG.get("kill_by_s", 42600)),
                                                                           _kill_all_children()),
                "| submission:", ensure_submission())
        finally:
            os._exit(0)
    threading.Thread(target=fire, daemon=True).start()


if S0 is not None:
    _start_killby()


def _ckpt_dir():
    """checkpoint directory: ARC_CKPT_DIR when set (the notebook launcher sets it after checking every file: the Kaggle
    Models mount, a local directory of links merging the checkpoint's Model parts, or the checkpoint dataset), else the
    first sorted IN/**/ckpt_marker match (v27). A set but unusable ARC_CKPT_DIR is an error, never a fallback to a
    marker search (which could find one part of a split checkpoint)."""
    marker = MODE.get("ckpt_marker", "model.safetensors.index.json")
    d = os.environ.get("ARC_CKPT_DIR")
    if d:
        if not (Path(d) / marker).is_file():
            raise FileNotFoundError(f"ARC_CKPT_DIR={d} has no {marker}")
        return Path(d)
    return Path(find(IN + "/**/" + marker)).parent


def _mount_fstype(p):
    """fstype of the mount holding path p (longest mount-point prefix in /proc/mounts), '' if unknown"""
    p, best = os.path.realpath(p), ("", "")
    try:
        for l in open("/proc/mounts"):
            x = l.split()
            mp = x[1].replace("\\040", " ")
            if (p == mp or p.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best[0]):
                best = (mp, x[2])
    except OSError:
        pass
    return best[1]


def _stage_plan(cfg, files):
    """MODE["prefetch"]["stage"] (default off): "fuse" = only when the checkpoint is on a FUSE
    mount (Kaggle's GCSFuse inputs, first seen in the v24n save run), "always" = any mount (tests). Builds stage_dir
    (default /tmp/arc_ckpt) as a mirror of the checkpoint dir made of symlinks; the prefetcher then replaces every file
    >= stage_min_mb (delta overlay, safetensors, lowbit-L*.pt, in load order) by a local copy (atomic rename). The server
    is pointed at stage_dir (ARC_STAGE_DIR); arcstats.stage_wait makes a TP rank wait for a file still being copied.
    Returns {} (no staging: the page-cache prefetch as before) or {"files", "copy", "stage_marker"}."""
    how = cfg.get("stage")
    if how not in ("fuse", "always"):
        return {}
    ck = _ckpt_dir()
    fs = _mount_fstype(str(ck))
    if how == "fuse" and not fs.lower().startswith("fuse"):
        log("stage: off, checkpoint mount is", fs or "?")
        return {}
    sd = Path(os.path.expandvars(cfg.get("stage_dir", "/tmp/arc_ckpt")))
    big = int(cfg.get("stage_min_mb", 64)) << 20
    allf = []
    for root, _, fns in os.walk(ck, followlinks=True):
        rel = Path(root).relative_to(ck)
        (sd / rel).mkdir(parents=True, exist_ok=True)
        for fn in sorted(fns):
            s, d = Path(root) / fn, sd / rel / fn
            if not (d.is_symlink() or d.exists()):
                os.symlink(s, d)
            if d.is_symlink() and s.stat().st_size >= big:
                allf.append((s, d))
    rank = lambda sd_: (0 if sd_[0].suffix == ".safetensors" else 1 if sd_[0].name.startswith("lowbit-L") else 2, str(sd_[0]))
    copy = [[str(s), str(d)] for s, d in sorted(allf, key=rank)]
    if MODE.get("delta_file"):   # read by every rank at weight load: first in line
        try:
            df = Path(find(IN + "/**/" + MODE["delta_file"]))
            dd = sd / "_delta" / df.name
            dd.parent.mkdir(exist_ok=True)
            if not (dd.is_symlink() or dd.exists()):
                os.symlink(df, dd)
            if dd.is_symlink():
                copy.insert(0, [str(df), str(dd)])
            os.environ["ARC_STAGE_DELTA"] = str(dd)
        except Exception as ex:
            log("stage: delta not staged", repr(ex)[:120])
    if cfg.get("stage_env"):   # (guard, default off) env tar, CUDA tar, Triton tar first: everything waits for the env
        tars = [(f, "ARC_STAGE_ENV_TAR") for f in files if re.search(r"/arc_py312\.tar[^/]*$", f)][:1] + \
            [(f, "ARC_STAGE_CUDA_TAR") for f in files if f.endswith("/cuda_toolkit.tarbin")][:1]
        if MODE.get("triton_cache_tar") and (code / MODE["triton_cache_tar"]).exists():
            tars.append((str(code / MODE["triton_cache_tar"]), "ARC_STAGE_TRITON_TAR"))
        (sd / "_env").mkdir(parents=True, exist_ok=True)
        for k, (f, var) in enumerate(tars):
            d = sd / "_env" / Path(f).name
            if not (d.is_symlink() or d.exists()):
                os.symlink(f, d)
            if d.is_symlink():
                copy.insert(k, [f, str(d)])
            os.environ[var] = str(d)
    marker = sd / ".staging"
    marker.write_text(str(time.time()))
    os.environ["ARC_STAGE_DIR"] = str(sd)                       # survives the re-exec; inherited by the server
    srcs = {s for s, _ in copy}
    log(f"stage: {how} (mount {fs or '?'}), {len(copy)} files "
        f"{sum(os.path.getsize(s) for s in srcs) / 2**30:.1f} GiB -> {sd}")
    # the env / CUDA tars stay on the page-cache list: tar and the prefetcher open them at t ~ 1 s, so FUSE's
    # invalidate-on-open does not drop the prefetched pages before tar reads them
    return {"files": [f for f in files if f not in srcs],
            "copy": copy, "stage_marker": str(marker),
            **({"barrier": [c[0] for c in copy if c[1].startswith(str(sd / "_env") + "/")]} if cfg.get("stage_env") else {})}


def _glob_in(rest):
    """sorted matches of IN/**/rest (the input index when find_index is on)"""
    if _FIND["on"]:
        import fnmatch
        return [p for p in _input_index() if fnmatch.fnmatchcase(p[len(IN) + 1:], rest)
                or fnmatch.fnmatchcase(p[len(IN) + 1:], "*/" + rest)]
    return sorted(glob.glob(IN + "/**/" + rest, recursive=True))


def _pid_alive(pid):
    """alive and not a zombie (a dead prefetcher stays a zombie child of this process until reaped)"""
    try:
        return open(f"/proc/{pid}/stat").read().rpartition(")")[2].split()[0] != "Z"
    except Exception:
        return False


def _stage_progress():
    """(bytes copied, time of the last update) from the robust copier's progress file, else (0, 0)"""
    try:
        x = open(Path(os.environ["ARC_STAGE_DIR"]) / ".progress").read().split()
        return int(x[0]), float(x[1])
    except Exception:
        return 0, 0.0


def _staged_local(dst, what):
    """(guard stage_env) wait until the copier has replaced the symlink dst by the local copy, while the copy is running
    (marker present, prefetcher alive) and the solver deadline is not near; return dst (local copy, or the symlink to the
    mount if the copy failed or was abandoned: the same bytes)."""
    t = time.time()
    marker = Path(os.environ.get("ARC_STAGE_DIR", "/nonexistent")) / ".staging"
    pid = int(os.environ.get("ARC_PREFETCH_PID") or 0)
    while os.path.islink(dst) and marker.exists() and (not pid or _pid_alive(pid)) and time.time() < deadline_epoch() - 600:
        time.sleep(0.5)
    log(f"stage: {what} {'local' if not os.path.islink(dst) else 'from the MOUNT'} after {time.time() - t:.0f} s wait")
    return dst


def _drop_local(p):
    """remove a staged local tar once unpacked (disk + page cache); a symlink to the mount is left alone"""
    try:
        if not os.path.islink(p):
            os.unlink(p)
    except OSError:
        pass


def start_prefetch():
    """MODE["prefetch"] (default off): detached page-cache prefetcher (_prefetch_main) started before the env unpack, so
    the env tar, the CUDA toolkit tar and the checkpoint (safetensors + lowbit-L*.pt, ~62 GB) stream from NFS in
    parallel while the env unpacks and the startup gates run. Survives the re-exec (own session). Byte-identical loads:
    only the page cache is warmed. MODE["prefetch"] may be a dict: threads, block_mb, max_gib, nice, timeout_s,
    ckpt (default true), env_tar (default true), extra (globs)."""
    cfg = MODE.get("prefetch")
    if not cfg or os.environ.get("ARC_PREFETCH_PID"):
        return
    cfg = cfg if isinstance(cfg, dict) else {}
    files = []
    try:
        if cfg.get("env_tar", True) and not Path(PY).exists() and not _glob_in("arc_py312/bin/python3.12"):
            files += _glob_in("arc_py312.tar*")[:1]
        if (code / "cuda_toolkit.tarbin").exists() and not Path("/tmp/cuda13").exists():
            files.append(str(code / "cuda_toolkit.tarbin"))
        if cfg.get("ckpt", True):
            ck = _ckpt_dir()
            files += sorted(str(p) for p in ck.glob("*.safetensors"))   # vLLM reads these first, then the experts
            files += sorted(str(p) for p in ck.glob("lowbit-L*.pt"))     # layer order = ArcLowbitExperts load order
        for g in cfg.get("extra", []):
            files += sorted(glob.glob(g, recursive=True))
        stage = _stage_plan(cfg, files)
        spec = W / "prefetch_spec.json"
        json.dump({"files": stage.get("files", files), "log": str(W / "prefetch_log.txt"), **{k: cfg[k] for k in
                   ("threads", "block_mb", "max_gib", "nice", "timeout_s", "stage_threads", "seg_mb", "throttle_mbps",
                    "robust", "stall_s", "max_attempts", "stall_emul", "barrier_stall_s")
                   if k in cfg}, **{k: stage[k] for k in ("copy", "stage_marker") if k in stage},
                   **({"deadline_epoch": deadline_epoch()} if S0 is not None else {}),
                   **({"progress": str(Path(stage["stage_marker"]).parent / ".progress")} if stage else {}),
                   **({"barrier": stage["barrier"]} if stage.get("barrier") else {}),
                   **({"barrier_seg_mb": cfg["barrier_seg_mb"]} if "barrier_seg_mb" in cfg else {})},
                  open(spec, "w"))
        p = subprocess.Popen([sys.executable, str(code / "run_l4.py"), "--prefetch", str(spec)],
                             stdout=subprocess.DEVNULL, stderr=open(W / "prefetch_log.txt", "a"), start_new_session=True)
        os.environ["ARC_PREFETCH_PID"] = str(p.pid)   # survives the re-exec: no second prefetcher
        log("prefetch started pid", p.pid, f"{len(files)} files",
            f"(S0 = T0 - {T0 - S0:.1f} s, solver deadline S0 + {deadline_epoch() - S0:.0f} s)" if S0 is not None else "")
    except Exception as ex:
        log("prefetch not started", repr(ex)[:200])
        if os.environ.pop("ARC_STAGE_DIR", None):   # nobody will copy: do not let the ranks wait for it
            try:
                os.unlink(Path(os.path.expandvars(cfg.get("stage_dir", "/tmp/arc_ckpt"))) / ".staging")
            except OSError:
                pass


if os.path.realpath(sys.executable) != os.path.realpath(PY):
    start_prefetch()
    sh("nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem --format=csv; nvidia-smi topo -m; free -g; nproc; df -h /dev/shm /tmp /kaggle/working; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true")
    if not Path(PY).exists():
        pre = _glob_in("arc_py312/bin/python3.12")
        if pre:
            sh(f"cp -a {Path(pre[0]).parents[1]} /tmp/arc_py312")
        elif os.environ.get("ARC_STAGE_ENV_TAR"):   # (guard stage_env) the parallel copier stages it first
            _et = _staged_local(os.environ["ARC_STAGE_ENV_TAR"], "env tar")
            sh(["tar", "-C", "/tmp", "-xf", _et])
            _drop_local(_et)
        else:
            sh(f"tar -C /tmp -xf {find(IN + '/**/arc_py312.tar*')}")
    log("re-exec under", PY); LOG.close()
    os.execv(PY, [PY, str(code / "run_l4.py")])

# ---------------------------------------------------------------- install code into the bundled env
SP = "/tmp/arc_py312/lib/python3.12/site-packages"
for f in list(code.glob("arcstats__*.py")) + list(code.glob("arcstats__*.so")):
    shutil.copy(f, f"{SP}/arcstats/{f.name[len('arcstats__'):]}")
for f in code.glob("vllm__*.py"):
    dst = Path(SP) / f.name.replace("__", "/")
    shutil.copy(f, dst); log("patched", dst.relative_to(SP))
if not Path("/tmp/cuda13").exists():
    if os.environ.get("ARC_STAGE_CUDA_TAR"):   # (guard stage_env) staged local copy (or its symlink: same bytes)
        _ct = _staged_local(os.environ["ARC_STAGE_CUDA_TAR"], "cuda tar")
        sh(["mkdir", "-p", "/tmp/cuda13"]); sh(["tar", "-C", "/tmp/cuda13", "-xf", _ct])
        _drop_local(_ct)
    else:
        sh(f"mkdir -p /tmp/cuda13 && tar -C /tmp/cuda13 -xf {code / 'cuda_toolkit.tarbin'}")
CUDA_HOME = "/tmp/cuda13/nvidia/cu13"
# JIT links (FlashInfer/DeepGEMM) need unversioned libs: expose the env's own CUDA 13 libs (shipped with torch)
CU13_LIB = f"{SP}/nvidia/cu13/lib"
for so in glob.glob(f"{CU13_LIB}/lib*.so.*"):
    base = re.sub(r"\.so\..*$", ".so", so)
    if not os.path.exists(base):
        os.symlink(os.path.basename(so), base)
# Kaggle images ship only the versioned driver lib (libcuda.so.1); JIT links (FlashInfer: -lcuda) need libcuda.so
CULINK = "/tmp/culink"
os.makedirs(CULINK, exist_ok=True)
if not os.path.exists(f"{CULINK}/libcuda.so"):
    cands = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + glob.glob("/usr/local/nvidia/lib64/libcuda.so*")
                   + glob.glob("/usr/lib64/libcuda.so*") + glob.glob("/usr/local/cuda*/compat/libcuda.so*"))
    if not cands:
        try:
            cands = re.findall(r"=> (\S*libcuda\.so\S*)", subprocess.run(["ldconfig", "-p"], capture_output=True,
                                                                         text=True).stdout)
        except Exception:
            cands = []
    if cands:
        os.symlink(cands[0], f"{CULINK}/libcuda.so")
    log("libcuda for JIT links:", cands[:1] or "NOT FOUND")
log("mode", json.dumps(MODE))
if S0 is not None:
    log("startup guard:", json.dumps(SG), f"S0 = T0 - {T0 - S0:.1f} s (first run_l4 start = T0 - "
        f"{T0 - float(os.environ.get('ARC_T0', T0)):.1f} s), solver deadline S0 + {deadline_epoch() - S0:.0f} s")
    # the TP ranks' per-file stage wait (arcstats.stage_wait, default cap 1800 s) may last as long as the copy runs:
    # reading the mount instead is never faster than the parallel copier, and the copier itself ends at the deadline
    os.environ["ARC_STAGE_WAIT_S"] = str(int(max(600.0, deadline_epoch() - time.time())))
    if os.environ.get("ARC_STAGE_DIR") and os.environ.get("ARC_PREFETCH_PID"):
        def _stage_monitor():   # a dead copier must release the waiters: reap it (no zombie) and drop the marker
            import threading  # noqa: F401
            pid = int(os.environ["ARC_PREFETCH_PID"])
            mk = Path(os.environ["ARC_STAGE_DIR"]) / ".staging"
            while mk.exists():
                try:
                    r, _st = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    r = pid if not _pid_alive(pid) else 0
                if r == pid:
                    if mk.exists():
                        log("stage: copier exited with the marker still present -> marker removed (mount reads)")
                        try:
                            mk.unlink()
                        except OSError:
                            pass
                    return
                time.sleep(2.0)
        import threading
        threading.Thread(target=_stage_monitor, daemon=True).start()
TRITON_DIRS = [os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")]
if MODE.get("triton_cache_tar"):
    # (default off) Triton kernel cache harvested from an earlier run on the same GPU/software (triton_cache_save):
    # compiled binaries are reused instead of re-JITted by every gate and every server rank; a key mismatch only means
    # a recompile, never different code. Path relative to the code dataset, or a glob under the inputs.
    # v27: the bundle ships it as "<name>.tarbin" (Kaggle auto-extracts uploaded *.tar files: v26's .tar arrived as a
    # directory and the restore failed). The archive is recognised by content (tarfile.is_tarfile), not extension; an
    # already-extracted directory "<stem>/" (old .tar upload) is copied instead.
    try:
        import tarfile
        t_ = code / MODE["triton_cache_tar"]
        if not t_.exists():
            try:
                t_ = Path(find(IN + "/**/" + MODE["triton_cache_tar"]))
            except FileNotFoundError:
                stem = re.sub(r"\.tar[^/]*$", "", MODE["triton_cache_tar"])
                t_ = next((p for p in [code / stem] + [Path(x) for x in glob.glob(IN + "/**/" + stem, recursive=True)]
                           if p.is_dir()), t_)
        if os.environ.get("ARC_STAGE_TRITON_TAR"):
            t_ = Path(_staged_local(os.environ["ARC_STAGE_TRITON_TAR"], "triton tar"))
        os.makedirs(TRITON_DIRS[0], exist_ok=True)
        if t_.is_dir():
            shutil.copytree(t_, TRITON_DIRS[0], dirs_exist_ok=True)
        elif tarfile.is_tarfile(str(t_)):
            sh(["tar", "-C", TRITON_DIRS[0], "-xf", str(t_)])
        else:
            raise FileNotFoundError(f"{t_}: not a tar archive")
        log("triton cache restored:", sum(len(f) for _, _, f in os.walk(TRITON_DIRS[0])), "files")
    except Exception as ex:
        log("triton cache restore failed (JIT as usual)", repr(ex)[:200])


def save_triton_cache(tag):
    """MODE["triton_cache_save"] (default off): tar the Triton cache into the working dir for a later triton_cache_tar"""
    if not MODE.get("triton_cache_save"):
        return
    try:
        out = W / "triton_cache.tar"
        subprocess.run(["tar", "-C", TRITON_DIRS[0], "-cf", str(out), "."], check=True, timeout=600)
        log("triton cache saved", tag, f"{out.stat().st_size / 2**20:.0f} MiB")
    except Exception as ex:
        log("triton cache save failed", tag, repr(ex)[:200])

# ---------------------------------------------------------------- link hog (emulation only, default off)
def start_hog(cfg, tag):
    """dots3/link_hog.py: rate-paced copy-engine H2D/D2H traffic on each GPU, so a wide GPU<->host link behaves more like
    Kaggle's narrow shared one. Never set in a submission mode."""
    args = [PY, str(code / "link_hog.py"), "--gpus", str(cfg.get("gpus", "0,1,2,3")), "--h2d", str(cfg.get("h2d", "max")),
            "--d2h", str(cfg.get("d2h", "max")), "--streams", str(cfg.get("streams", 4)),
            "--chunk-mb", str(cfg.get("chunk_mb", 4)), "--report", str(cfg.get("report", 60))]
    env = dict(os.environ, PYTHONPATH=SP)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    p = subprocess.Popen(args, env=env, stdout=open(W / f"link_hog_{tag}.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    time.sleep(float(cfg.get("warm_s", 5)))
    log("link_hog start", tag, json.dumps(cfg), "pid", p.pid, "alive" if p.poll() is None else f"EXITED {p.returncode}")
    return p


def stop_hog(p, tag):
    if p is None:
        return
    try:
        os.killpg(p.pid, 15)
        p.wait(10)
    except Exception:
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass
    log("link_hog stop", tag)


HOG = start_hog(MODE["link_hog"], "global") if MODE.get("link_hog") else None
if HOG is not None:
    import atexit
    atexit.register(stop_hog, HOG, "global")

# ---------------------------------------------------------------- microbench
if MODE.get("microbench", True):
    try:
        env = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP)
        sh([PY, str(code / "l4_microbench.py"), str(W / "microbench.json"), "--gpus", "4"], env=env, timeout=1800,
           cwd=str(code))
    except Exception as ex:
        log("microbench failed", repr(ex)[:200])

# ---------------------------------------------------------------- startup autotune on the real GPUs (seconds-minutes)
TUNE_ENV = {}
if MODE.get("lowbit_tune_file"):   # table measured on a real L4 by an earlier run (no startup cost)
    TUNE_ENV["ARC_LOWBIT_TUNE"] = str(code / MODE["lowbit_tune_file"])
if MODE.get("lowbit_tune_table"):   # same, inline in the mode ({proj: {M: [bm, bn, warps, stages]}}, e.g. a Kaggle pick)
    json.dump(MODE["lowbit_tune_table"], open(W / "lowbit_tune_pinned.json", "w"))
    TUNE_ENV["ARC_LOWBIT_TUNE"] = str(W / "lowbit_tune_pinned.json")
if MODE.get("gate_pins"):
    # Startup-gate picks fixed in the mode (default off): what the gates chose on every Kaggle 4xL4 run, applied as if
    # the gate had picked it (TUNE_ENV, so pick-dependent logic below -- kv_bonus_lmhead_gib, gate_merge's W8A8
    # precondition, topk's V2 pattern -- sees it). Turn the matching *_gate flags off: a gate left on still runs and
    # can overwrite keys, but its 'off' verdict does not remove a pin. the design notes lists the stable picks.
    TUNE_ENV.update({k: str(v) for k, v in MODE["gate_pins"].items()})
    if TUNE_ENV.get("ARC_FP8_W8A8") == "1" and not MODE.get("w8a8_gate"):   # the gate's side effect: tile configs
        os.makedirs("/tmp/w8a8_configs", exist_ok=True)
        for fn, cfg in json.load(open(code / "w8a8_configs.json")).items():
            json.dump(cfg, open(os.path.join("/tmp/w8a8_configs", fn), "w"))
        TUNE_ENV.setdefault("ARC_W8A8_CONFIG_DIR", "/tmp/w8a8_configs")
    log("gate_pins", json.dumps(MODE["gate_pins"]))
if MODE.get("tune_lowbit"):
    try:
        env_t = dict(os.environ, **{k: v for k, v in MODE.get("env", {}).items() if k.startswith("ARC_LOWBIT")},
                     CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0")
        sh([PY, str(code / "tune_lowbit.py"), str(W / "lowbit_tune.json")] +
           (["--ms", MODE["tune_lowbit_ms"]] if MODE.get("tune_lowbit_ms") else []),   # decode M = 4 x seqs
           env=env_t, timeout=900, cwd=str(code))
        TUNE_ENV["ARC_LOWBIT_TUNE"] = str(W / "lowbit_tune.json")
    except Exception as ex:
        log("lowbit tune failed (defaults kept)", repr(ex)[:200])
if MODE.get("nccl_candidates"):
    best = None
    for cand in MODE["nccl_candidates"]:
        try:
            env_n = dict(os.environ, **cand, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, NCCL_P2P_DISABLE="1")
            r = run_bench([PY, str(code / "nccl_pick.py"), "4"], env_n, 300, cwd=str(code))
            us = json.loads(r.stdout.strip().splitlines()[-1])["us"]
            score = 86 * us["ar_decode"] + 3 * us["ar_prefill"] + 20 * us["ag_index"]   # per decode step
            log("nccl", json.dumps(cand), json.dumps(us), f"score {score / 1e3:.1f} ms/step")
            if best is None or score < best[0]:
                best = (score, cand)
        except Exception as ex:
            log("nccl candidate failed", json.dumps(cand), repr(ex)[:200])
    if best:
        TUNE_ENV.update(best[1]); log("nccl pick", json.dumps(best[1]))

if MODE.get("host_ar_gate"):   # host-staged all-reduce (arcstats.host_ar) where it beats NCCL on these GPUs
    try:
        env_h = dict(os.environ, **{k: v for k, v in TUNE_ENV.items() if k.startswith("NCCL_")}, CUDA_HOME=CUDA_HOME,
                     PYTHONPATH=SP, NCCL_P2P_DISABLE="1", ARC_HOST_AR_TIMEOUT_MS="30000", ARC_HOST_AR_MONITOR="0",
                     **{k: str(v) for k, v in MODE.get("env", {}).items()
                        if k in ("ARC_HOST_AR_BITS", "ARC_HOST_AR_RED_BITS")})   # int6/int5 check (qb_* fields)
        # (device-side spin-wait watchdog: a hang becomes garbage results -> err check fails -> NCCL kept)
        r = run_bench([PY, str(code / "bench_host_ar.py"), "4", "64,160,256,512,1024"], env_h, 600, cwd=str(code))
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        ok_mb = 0.0
        ag = [d for d in rows if "op" in d]
        for d in ag:
            log("host_ag", json.dumps(d))
        ag_ok = bool(ag) and all(d["exact"] and d["host_us"] < 0.95 * d["nccl_us"] for d in ag)
        # host AR for the contiguous run of measured sizes where it wins (decode all-reduces are 64-256 tokens; a tie
        # at the smallest size must not disable it for the larger ones)
        ar = sorted([d for d in rows if "tokens" in d], key=lambda d: d["tokens"])
        q8 = bool(MODE.get("host_ar_q8")) and all("q8_graph_us" in d for d in ar) and \
            all(d["q8_same"] and d["q8_err"] < 0.02 for d in ar)
        # int8 payloads: same model-level deviation as NCCL vs bf16 host (summation-order noise), ~0.55x time
        key = "q8_graph_us" if q8 else "host_graph_us"
        win = [d["err"] < 0.02 and d["graph_err"] < 0.02 and d[key] < 0.95 * d["nccl_us"] for d in ar]
        for d, w in zip(ar, win):
            log("host_ar", json.dumps(d), "win" if w else "")
        best, best_gain, cur = None, 0.0, None   # contiguous run of wins with the largest total gain
        for i, w in enumerate(win + [False]):
            if w and cur is None:
                cur = i
            elif not w and cur is not None:
                gain = sum(ar[j]["nccl_us"] - ar[j][key] for j in range(cur, i))
                if gain > best_gain:
                    best, best_gain = (cur, i - 1), gain
                cur = None
        min_mb = 0.0
        if best:
            lo, hi = best
            ok_mb = ar[hi]["MB"]
            min_mb = 0.0 if lo == 0 else (ar[lo - 1]["MB"] + ar[lo]["MB"]) / 2
        if ok_mb > 0:
            TUNE_ENV.update(ARC_HOST_AR="1", ARC_HOST_AR_MAX_MB=f"{ok_mb + 0.01:.2f}", ARC_HOST_AR_MIN_MB=f"{min_mb:.3f}",
                            ARC_HOST_AR_Q8="1" if q8 else "0")
        elif ag_ok:   # all-gather only: all-reduce sizes go to NCCL
            TUNE_ENV.update(ARC_HOST_AR="1", ARC_HOST_AR_MAX_MB="0.001")
        TUNE_ENV["ARC_HOST_AG"] = "1" if ag_ok else "0"
        log("host_ar pick", (f"AR {'int8' if q8 else 'bf16'} {min_mb:.2f}..{ok_mb} MB") if ok_mb else "AR off (NCCL)", f"AG {'on' if ag_ok else 'off'}",
            r.stderr[-300:] if not rows else "")
        if str(MODE.get("env", {}).get("ARC_HOST_AR_BITS", "8")) != "8":   # packed low-bit payloads (mode env): sanity
            qb_ok = q8 and bool(ar) and all(d.get("qb_same") and d.get("qb_err", 1.0) < 0.1 for d in ar)
            log("host_ar bits", MODE["env"]["ARC_HOST_AR_BITS"], "ok" if qb_ok else "FAILED -> int8 kept",
                json.dumps({d["tokens"]: [d.get("qb_err"), d.get("qb_graph_us"), d.get("q8_graph_us")] for d in ar}))
            if not qb_ok:
                MODE["env"].pop("ARC_HOST_AR_BITS", None); MODE["env"].pop("ARC_HOST_AR_RED_BITS", None)
        if MODE.get("host_ar_timeout_ms") and ok_mb > 0:   # serving watchdog (logs; ACTION=exit kills a hung rank)
            TUNE_ENV.update(ARC_HOST_AR_TIMEOUT_MS=str(MODE["host_ar_timeout_ms"]),
                            ARC_HOST_AR_TIMEOUT_ACTION=MODE.get("host_ar_timeout_action", "log"))
        if q8 and ok_mb > 0 and MODE.get("host_ar_pipe"):   # q8 tile/warps + bit-identical int8 kernel candidates
            try:   # (parallel peer polling, K-stage skewed pipeline, ld polling, pipelined CTA roles)
                nbp = str(MODE.get("host_ar_pipe_blocks", 16))
                tmo = int(MODE.get("host_ar_pipe_timeout", 480))
                rp = run_bench([PY, str(code / "bench_host_ar_pipe.py"), "4", "64,160,256,1024"],
                               dict(env_h, BENCH_GPUS="0,1,2,3", ARC_HOST_AR_PIPE_BLOCKS=nbp,
                                    BENCH_PIPE="1" if MODE.get("host_ar_pipe_roles", True) else "0",
                                    BENCH_DEADLINE_S=str(tmo - 30), ARC_HOST_AR_TIMEOUT_MS="20000"), tmo, cwd=str(code))
                lines = rp.stdout.splitlines()
                chk = [l for l in lines if l.startswith("har check")]
                bad = [l for l in lines if l.startswith("har TIMEOUT") or l.startswith("har FAILED")]
                ok_all = bool(chk) and not bad and rp.returncode == 0 and all(
                    "ranks_equal True graph_equal True" in l for l in chk)
                ex_pipe = ok_all and all("pipe_vs_q8_equal True" in l for l in chk)
                ex_skew = ok_all and all("skew_equal True" in l for l in chk)
                ex_ldpw = ok_all and all("ld_pw_equal True" in l for l in chk)
                kok = {}   # per-kind exactness (benches with candidate isolation: 'kinds pipe=1,skew=0,...')
                for l in chk:
                    mk = re.search(r" kinds (\S+)", l)
                    for kv in (mk.group(1).split(",") if mk else []):
                        kk, vv = kv.split("=")
                        kok[kk] = kok.get(kk, True) and vv == "1"
                for l in lines:
                    if l.startswith("har kindfail"):
                        log("host_ar_pipe", l)
                if not ok_all:   # what hung / failed: the bench's own report + every watchdog line of the workers
                    log("host_ar_pipe stderr", " | ".join(l for l in rp.stderr.splitlines()
                                                          if "TIMEOUT" in l or "kindfail" in l or "Error" in l)[-3000:])
                tq, tv, cands = {}, {}, {}   # q8 default (graph), q8 tile grid, candidates {name: {T: us}}
                for l in lines:
                    if l.startswith("har time T"):
                        log("host_ar_pipe", l)
                        T = int(l.split()[3])
                        tq[T] = float(re.search(r"q8_graph (\S+)", l).group(1))
                        if ex_pipe:
                            for cfg, us in re.findall(r"pipe (r\d+ch\d+w\d+ns\d+) (\S+)", l):
                                cands.setdefault("pipe " + cfg, {})[T] = float(us)
                    elif l.startswith("har q8tile T"):
                        log("host_ar_q8tile", l)
                        T = int(l.split()[3])
                        for cfg, us in re.findall(r"q8 (t\d+w\d+) (\S+)", l):
                            tv.setdefault(cfg, {})[T] = float(us)
                    elif l.startswith("har cand T"):
                        log("host_ar_cand", l)
                        T = int(l.split()[3])
                        for kind, cfg, us in re.findall(r"(ld|pw|skewpw|skew) (\S+) (\S+)", l.split(None, 3)[3]):
                            if (kok.get(kind, False) and ok_all) if kok else \
                                    ((kind.startswith("skew") and ex_skew and (kind == "skew" or ex_ldpw)) or
                                     (kind in ("ld", "pw") and ex_ldpw)):
                                cands.setdefault(f"{kind} {cfg}", {})[T] = float(us)
                    elif l.startswith("har probe"):
                        log("host_ar_probe", l)
                dec = [T for T in (64, 160, 256) if T in tq]
                tot = lambda d: sum(d.get(T, 1e9) for T in dec)
                bt = min(tv.items(), key=lambda kv: tot(kv[1])) if ok_all and tv and dec and \
                    not any("tiles_equal False" in l for l in chk) else None
                if bt and tot(bt[1]) < 0.95 * tot(tq):
                    t_, w_ = re.match(r"t(\d+)w(\d+)", bt[0]).groups()
                    TUNE_ENV.update(ARC_HOST_AR_TILE=t_, ARC_HOST_AR_WARPS=w_)
                    tq = {T: bt[1].get(T, tq[T]) for T in tq}     # candidates must beat the tuned q8
                log("host_ar q8 tile pick", bt[0] if bt else None, TUNE_ENV.get("ARC_HOST_AR_TILE", "default"),
                    "" if chk else (rp.stdout[-300:] + rp.stderr[-600:]))
                best = min(cands.items(), key=lambda kv: tot(kv[1])) if cands and dec else None
                pick = None
                if best and tot(best[1]) < 0.95 * tot(tq) and best[1].get(1024, 0) <= 1.05 * tq.get(1024, 1e9):
                    pick = best[0]
                    kind, cfg = pick.split()
                    if kind == "pipe":
                        rr, ch, w, ns = re.match(r"r(\d+)ch(\d+)w(\d+)ns(\d+)", cfg).groups()
                        TUNE_ENV.update(ARC_HOST_AR_PIPE="1", ARC_HOST_AR_PIPE_BLOCKS=nbp, ARC_HOST_AR_PIPE_ROUNDS=rr,
                                        ARC_HOST_AR_PIPE_CH=ch, ARC_HOST_AR_PIPE_WARPS=w, ARC_HOST_AR_PIPE_NS=ns)
                    else:
                        k_, t_, w_ = re.match(r"(?:k(\d+))?t(\d+)w(\d+)", cfg).groups()
                        TUNE_ENV.update(ARC_HOST_AR_TILE=t_, ARC_HOST_AR_WARPS=w_)
                        if kind.startswith("skew"):
                            TUNE_ENV["ARC_HOST_AR_SKEW"] = k_
                        if kind in ("pw", "skewpw"):
                            TUNE_ENV["ARC_HOST_AR_PWAIT"] = "1"
                        if kind == "ld":
                            TUNE_ENV["ARC_HOST_AR_POLL"] = "ld"
                log("host_ar candidate pick", "exact" if ok_all else "NOT EXACT/FAILED", f"q8 {tot(tq):.0f} us",
                    f"best {best[0]} {tot(best[1]):.0f} us" if best else "none", "->", pick or "q8 kept",
                    "" if ok_all else (rp.stdout[-300:] + rp.stderr[-600:]))
            except Exception as ex:
                log("host_ar_pipe gate failed (q8 kernel kept)", repr(ex)[:200])
    except Exception as ex:
        log("host_ar gate failed (NCCL kept)", repr(ex)[:200])

if MODE.get("offload_fill_gate"):   # row-cache scan/fill/resolve variants (arcstats.offload_fill) if faster here
    try:   # synthetic drifting row stream at the serving footprint (12 x 4.5 GB host buffers), CUDA-graph replay
        env_f = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", ARC_OFFLOAD_WAYS="8",
                     ARC_OFFLOAD_RESOLVE="1")
        r = subprocess.run([PY, str(code / "bench_pcie_fill.py"), "--parts", "serve", "--bufs", "12",
                            "--serve-new", str(MODE.get("offload_fill_gate_new", 8000))], env=env_f, timeout=1800,
                           capture_output=True, text=True, cwd=str(code))
        res = {}
        for l in r.stdout.splitlines():
            m = re.match(r"pcie serve (\S+) .* ([0-9.]+) us/layer-call", l)
            if m:
                log("offload_fill_gate", l)
                res[m.group(1)] = float(m.group(2))
        base = res.get("default")
        best = min(res.items(), key=lambda kv: kv[1]) if res else None
        if base and best and best[1] < 0.95 * base:
            nm = best[0]
            if "epoch" in nm:
                TUNE_ENV["ARC_OFFLOAD_HT_EPOCH"] = "1"
            mb = re.search(r"_B(\d+)w(\d+)c(\d+)$", nm)
            if mb:
                TUNE_ENV.update(ARC_OFFLOAD_FILL_SPLIT="1", ARC_OFFLOAD_FILL_B=mb.group(1),
                                ARC_OFFLOAD_FILL_WARPS=mb.group(2), ARC_OFFLOAD_FILL_CTAS=mb.group(3))
            if "sort64k" in nm:
                TUNE_ENV["ARC_OFFLOAD_FILL_SORT"] = "65536"
            if "two" in nm:
                TUNE_ENV["ARC_OFFLOAD_TWO_CHOICE"] = "1"
        log("offload_fill_gate pick", f"default {base} best {best}",
            {k: v for k, v in TUNE_ENV.items() if k.startswith(("ARC_OFFLOAD_HT", "ARC_OFFLOAD_FILL", "ARC_OFFLOAD_TWO"))},
            "" if res else (r.stdout[-300:] + r.stderr[-500:]))
    except Exception as ex:
        log("offload fill gate failed (default kept)", repr(ex)[:200])

if MODE.get("swa_fused_gate"):   # fused SWA decode attention (arcstats.swa_fused) only if faster on this GPU
    try:   # candidates: 1 (per-token split-K), 2 (request-packed Triton), 2+cudaC (single-pass FP8 CUDA, ARC_SWA_CUDA=C)
        env_s = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                     ARC_SWA_CUDA_CANDS=MODE.get("swa_cuda_cands", "1,11"))
        r = subprocess.run([PY, str(code / "bench_swa_fused.py")], env=env_s, timeout=600, capture_output=True,
                           text=True, cwd=str(code))
        out = r.stdout
        t_o = float(re.search(r"^orig ([0-9.]+) ms", out, re.M).group(1))
        num = lambda pat: (lambda m: float(m.group(1)) if m else float("inf"))(re.search(pat, out, re.M))
        t_f, err = num(r"^fused ([0-9.]+) ms"), num(r"^rel err \(excluding the qpos<0 token\) ([0-9.e+-]+)")
        t_p, err_p = num(r"^packed ([0-9.]+) ms"), num(r"^packed rel err \(excluding the qpos<0 token\) ([0-9.e+-]+)")
        allc = [(t_f, err, "1"), (t_p, err_p, "2")]
        for c in re.findall(r"^cuda(\d+) [0-9.]+ ms", out, re.M):
            allc.append((num(rf"^cuda{c} ([0-9.]+) ms"), num(rf"^cuda{c} rel err \(excluding the qpos<0 token\) ([0-9.e+-]+)"),
                         f"2+cuda{c}"))
        cand = [(t, m) for t, e, m in allc if e < float(MODE.get("swa_err_max", 0.03)) and t < 0.9 * t_o]
        pick = min(cand)[1] if cand else None
        if pick:
            TUNE_ENV["ARC_SWA_FUSED"] = pick.split("+")[0]
            TUNE_ENV["ARC_SWA_CUDA"] = pick[len("2+cuda"):] if "+cuda" in pick else "0"
        log("swa_fused", f"orig {t_o} ms", " ".join(f"{m} {t} ms (err {e:.2e})" for t, e, m in allc), "->",
            f"ARC_SWA_FUSED={TUNE_ENV['ARC_SWA_FUSED']} ARC_SWA_CUDA={TUNE_ENV['ARC_SWA_CUDA']}" if pick else "off")
    except Exception as ex:
        log("swa_fused gate failed (off)", repr(ex)[:200], (r.stdout[-200:] + r.stderr[-300:]) if 'r' in dir() else "")

if MODE.get("w8a8_gate"):   # Triton W8A8 block-FP8 dense linears instead of FP8 Marlin, only if faster on this GPU
    try:
        cfgdir = "/tmp/w8a8_configs"                   # unpacked from the bundled w8a8_configs.json {filename: config}
        os.makedirs(cfgdir, exist_ok=True)
        for fn, cfg in json.load(open(code / "w8a8_configs.json")).items():
            json.dump(cfg, open(os.path.join(cfgdir, fn), "w"))
        env_w = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                     ARC_W8A8_CONFIG_DIR=cfgdir, ARC_W8A8_SHAPES=MODE.get("w8a8_shapes", ""))
        r = subprocess.run([PY, str(code / "bench_fp8_gemm.py"), "160"], env=env_w, timeout=900, capture_output=True,
                           text=True, cwd=str(code))
        m = re.search(r"T=160 total marlin ([0-9.]+) us\s+w8a8 ([0-9.]+) us", r.stdout)
        errs = [float(e) for e in re.findall(r"w8a8 +[0-9.]+ us \(err ([0-9.e+-]+)\)", r.stdout)]
        t_m, t_w = float(m.group(1)), float(m.group(2))
        on = t_w < 0.9 * t_m and errs and max(errs) < 0.06
        if on:
            TUNE_ENV.update(ARC_FP8_W8A8="1", ARC_W8A8_CONFIG_DIR=cfgdir)
        log("w8a8", f"marlin {t_m} us w8a8 {t_w} us per step-weighted layer set, max err {max(errs) if errs else None} ->",
            "ON" if on else "off")
    except Exception as ex:
        log("w8a8 gate failed (Marlin kept)", repr(ex)[:200], (r.stdout[-300:] + r.stderr[-300:]) if 'r' in dir() else "")

if MODE.get("w8a8_ws_gate") and TUNE_ENV.get("ARC_FP8_W8A8") == "1" and (code / "w8a8_ws.json").exists():
    # arcstats.w8a8_ws decode GEMM (weight-stationary M sub-tiles / split-K / quant fused into the prologue) per shape,
    # only where faster than quant + vLLM's kernel with weights cold in L2 (as in serving)
    try:
        env_ws = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                      ARC_W8A8_CONFIG_DIR="/tmp/w8a8_configs", ARC_W8A8_WS_CONFIG=str(code / "w8a8_ws.json"),
                      ARC_W8A8_SHAPES=MODE.get("w8a8_shapes", ""))
        r = subprocess.run([PY, str(code / "bench_w8a8_ws.py"), "gate", str(MODE.get("w8a8_ws_ms", "192"))], env=env_ws,
                           timeout=1500, capture_output=True, text=True, cwd=str(code))
        for l in r.stdout.splitlines():
            if l.startswith("N="):
                log("w8a8_ws", l)
        m = re.search(r"WS_GATE base ([0-9.]+) us ws ([0-9.]+) us .* shapes (\S+)", r.stdout)
        t_b, t_w, shp = float(m.group(1)), float(m.group(2)), m.group(3)
        on = shp != "-" and t_w < 0.97 * t_b
        if on:
            TUNE_ENV.update(ARC_W8A8_WS="1", ARC_W8A8_WS_CONFIG=str(code / "w8a8_ws.json"), ARC_W8A8_WS_SHAPES=shp)
        log("w8a8_ws", f"base {t_b} us ws {t_w} us per step-weighted layer set (cold L2) ->", f"ON {shp}" if on else "off")
    except Exception as ex:
        log("w8a8_ws gate failed (off)", repr(ex)[:200], (r.stdout[-300:] + r.stderr[-300:]) if 'r' in dir() else "")

if MODE.get("moe_combine_gate"):   # fused fp32-atomic top-k combine in the low-bit MoE down GEMM, only if faster here
    try:
        env_c = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                     **{k: v for k, v in MODE.get("env", {}).items() if k.startswith("ARC_LOWBIT")})
        r = subprocess.run([PY, str(code / "bench_moe_combine.py"), "160"], env=env_c, timeout=600, capture_output=True,
                           text=True, cwd=str(code))
        m = re.search(r"sum ([0-9.]+) ms\s+combine ([0-9.]+) ms\s+rel err ([0-9.e+-]+)", r.stdout)
        t_s, t_c, err = float(m.group(1)), float(m.group(2)), float(m.group(3))
        on = t_c < 0.95 * t_s and err < 0.02
        if on:
            TUNE_ENV["ARC_LOWBIT_COMBINE"] = "1"
        log("moe_combine", f"sum {t_s} ms combine {t_c} ms err {err:.1e} ->", "ON" if on else "off")
    except Exception as ex:
        log("moe_combine gate failed (off)", repr(ex)[:200], (r.stdout[-200:] + r.stderr[-300:]) if 'r' in dir() else "")

if MODE.get("lowbit_fuse_gate") and MODE.get("env", {}).get("ARC_LOWBIT_PAIRS") == "1":
    # fused MoE small kernels (arcstats.lowbit_fuse: routing, x -> fp16 + group sums, silu + fp16 + group sums, combine):
    # ON only if the top-k ids match, the layer output is within bf16 noise of the current path (rel L2 < 5e-3) and
    # the fused part is >= 5% faster and the whole layer faster, at every M (interleaved medians, real weights)
    try:
        ms = str(MODE.get("lowbit_fuse_ms", "256"))
        env_f = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", BENCH_REAL="1",
                     **{k: v for k, v in MODE.get("env", {}).items() if k.startswith("ARC_LOWBIT")},
                     **{k: v for k, v in TUNE_ENV.items() if k.startswith("ARC_LOWBIT")})
        r = run_bench([PY, str(code / "bench_lowbit_fuse.py"), ms], env_f, 600, cwd=str(code))
        for l in r.stdout.splitlines():
            if l.startswith(("lbf ", "FUSE_GATE")):
                log("lowbit_fuse", l)
        g = [l for l in r.stdout.splitlines() if l.startswith("FUSE_GATE")]
        on = len(g) == len(ms.split(",")) and all(l.endswith("ok True") for l in g)
        if on:
            TUNE_ENV["ARC_LOWBIT_FUSE"] = "1"
        log("lowbit_fuse_gate ->", "ON" if on else "off", "" if g else (r.stdout[-300:] + r.stderr[-600:]))
    except Exception as ex:
        log("lowbit_fuse gate failed (off)", repr(ex)[:200])
if MODE.get("lowbit2_tune_file"):   # fixed per-M lowbit2 pipeline (A/B runs without the gate); the gate overrides it
    TUNE_ENV.update(ARC_LOWBIT2="1", ARC_LOWBIT2_TUNE=str(code / MODE["lowbit2_tune_file"]))
if MODE.get("lowbit2_gate") and MODE.get("env", {}).get("ARC_LOWBIT_PAIRS") == "1":
    # per-M low-bit MoE pipeline (arcstats.lowbit2: tiles + evict-first weights, fused bit-exact top-k combine with L2
    # discard, expert-stationary kernel): full MoE layer old vs best variant on this GPU at the decode batch sizes, real
    # expert weights when present, interleaved medians; ON only if bit-identical at every M and >= 5% faster overall
    try:
        ms = str(MODE.get("lowbit2_ms", "256,320,384"))
        env_l = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", BENCH_REAL="1",
                     BENCH_OUT=str(W / "lowbit2_tune.json"), BENCH_DEADLINE_S=str(MODE.get("lowbit2_gate_s", 900)),
                     **{k: v for k, v in MODE.get("env", {}).items() if k.startswith("ARC_LOWBIT")},
                     **{k: v for k, v in TUNE_ENV.items() if k.startswith("ARC_LOWBIT")})
        r = run_bench([PY, str(code / "bench_lowbit2.py"), ms, "--gate"], env_l, int(MODE.get("lowbit2_gate_s", 900)) + 300,
                      cwd=str(code))
        for l in r.stdout.splitlines():
            if l.startswith(("lb2 ", "LB2_GATE")):
                log("lowbit2", l)
        g = [re.search(r"LB2_GATE M=(\d+) old ([0-9.]+) new ([0-9.]+) exact (\w+)", l)
             for l in r.stdout.splitlines() if l.startswith("LB2_GATE")]
        g = [m for m in g if m]
        t_o, t_n = sum(float(m.group(2)) for m in g), sum(float(m.group(3)) for m in g)
        on = bool(g) and len(g) == len(ms.split(",")) and all(m.group(4) == "True" for m in g) and t_n < 0.95 * t_o
        if on:
            TUNE_ENV.update(ARC_LOWBIT2="1", ARC_LOWBIT2_TUNE=str(W / "lowbit2_tune.json"))
        log("lowbit2_gate", f"old {t_o:.3f} ms new {t_n:.3f} ms over M={ms} exact {[m.group(4) for m in g]} ->",
            "ON" if on else "off", "" if g else (r.stdout[-300:] + r.stderr[-600:]))
    except Exception as ex:
        log("lowbit2 gate failed (off)", repr(ex)[:200])

if MODE.get("offload_packed_gate"):   # request-packed DSA offload attention only if faster on this GPU
    try:
        menv = MODE.get("env", {})
        base_e = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", BENCH_DISTINCT="1",
                      BENCH_RANDHOST="1",   # random host rows + dense reference check: wrong variants are rejected
                      ARC_OFFLOAD_WAYS=menv.get("ARC_OFFLOAD_WAYS", "8"), ARC_OFFLOAD_RESOLVE="1")
        variants = [("split", {"ARC_OFFLOAD_PACKED": "0"}),
                    ("split_bh32", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32", "ARC_OFFLOAD_SPLIT_WARPS": "8"}),
                    ("split_bh32_s4", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32", "ARC_OFFLOAD_SPLIT_WARPS": "8",
                                       "ARC_OFFLOAD_SPLIT": "4"}),   # fewer split-K partials: less combine traffic
                    ("split_bh32_s2", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32", "ARC_OFFLOAD_SPLIT_WARPS": "8",
                                       "ARC_OFFLOAD_SPLIT": "2"}),
                    ("split_bh32_s2_st3", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32", "ARC_OFFLOAD_SPLIT_WARPS": "8",
                                           "ARC_OFFLOAD_SPLIT": "2", "ARC_OFFLOAD_SPLIT_STAGES": "3"}),
                    # fp16 MMA inputs: sm89 converts e4m3 pairs to f16 in one instruction (bf16 needs more)
                    ("split_bh32_s2_f16", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32",
                                           "ARC_OFFLOAD_SPLIT_WARPS": "8", "ARC_OFFLOAD_SPLIT": "2",
                                           "ARC_OFFLOAD_SPLIT_F16": "1"}),
                    ("split_bh32_s2_st3_f16", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_SPLIT_BH": "32",
                                               "ARC_OFFLOAD_SPLIT_WARPS": "8", "ARC_OFFLOAD_SPLIT": "2",
                                               "ARC_OFFLOAD_SPLIT_STAGES": "3", "ARC_OFFLOAD_SPLIT_F16": "1"}),
                    # CUDA kernel (arcstats/offload_cuda.py; also takes the prefill chunks), split-K x pipeline/warps
                    *[(f"cuda_s{sp}_c{cf}", {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_CUDA": "1",
                                              "ARC_OFFLOAD_CUDA_SPLIT": str(sp), "ARC_OFFLOAD_CUDA_STAGES": str(cf)})
                      for sp in ((1, 2, 4) if MODE.get("offload_cuda_gate", True) else ()) for cf in (2, 3, 13)],
                    # request-packed CUDA kernel (offload_cuda.pk_*): each request's union of MTP top-k rows resolved and
                    # loaded once, 64 query rows per CTA, split-K over the union merged in-kernel; _q2 = token pairs
                    *[(f"cudapk_k{ks}" + (f"_q{qq}" if qq else ""),
                       {"ARC_OFFLOAD_PACKED": "0", "ARC_OFFLOAD_CUDA": "1", "ARC_OFFLOAD_CUDA_SPLIT": "1",
                        "ARC_OFFLOAD_CUDA_STAGES": "2", "ARC_OFFLOAD_CUDA_PK": "1", "ARC_OFFLOAD_CUDA_PK_KS": str(ks),
                        "ARC_OFFLOAD_CUDA_PK_QL": str(qq)})
                      for ks, qq in (((1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (4, 2))
                                     if MODE.get("offload_cudapk_gate", False) else ())],
                    ("packed", {"ARC_OFFLOAD_PACKED": "1"}),
                    ("packed_bn32", {"ARC_OFFLOAD_PACKED": "1", "ARC_OFFLOAD_PK_BN": "32", "ARC_OFFLOAD_PK_BD": "64"})]
        res = {}
        for nm, ev in variants:
            r = subprocess.run([PY, str(code / "bench_offload_cache.py"), menv.get("ARC_OFFLOAD_CACHE_ROWS", "131072"),
                                "16"], env=dict(base_e, **ev), timeout=600, capture_output=True, text=True,
                               cwd=str(code))
            m = re.search(r"([0-9.]+) ms/call", r.stdout)
            e = re.search(r"ref check: max rel err ([0-9.e+-]+)", r.stdout)
            res[nm] = float(m.group(1)) if m and e and float(e.group(1)) < 0.02 else None
            if not m:
                log("offload_packed gate", nm, "failed", (r.stdout[-200:] + r.stderr[-300:]))
        best = min((t, nm) for nm, t in res.items() if t is not None and nm != "split") if any(
            t is not None for nm, t in res.items() if nm != "split") else None
        on = res.get("split") and best and best[0] < 0.9 * res["split"]
        if on:
            TUNE_ENV.update({k: v for k, v in dict(variants)[best[1]].items()})
        log("offload_packed", res, "->", best[1] if on else "split")
    except Exception as ex:
        log("offload_packed gate failed (split kept)", repr(ex)[:200])

if MODE.get("lmhead_gate"):   # block-FP8 LM head (arcstats.lmhead_fp8) only if faster here (per step: 1 verify + 3 drafts)
    try:
        env_l = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0")
        if os.path.isdir("/tmp/w8a8_configs"):
            env_l["ARC_W8A8_CONFIG_DIR"] = "/tmp/w8a8_configs"
        r = subprocess.run([PY, str(code / "bench_lmhead.py")], env=env_l, timeout=900, capture_output=True, text=True,
                           cwd=str(code))
        t = {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
             for m in re.finditer(r"^lmhead M=(\d+) bf16 ([0-9.]+) ms fp8 ([0-9.]+) ms", r.stdout, re.M)}
        bf = t[160][0] + 3 * t[40][0]
        f8 = t[160][1] + 3 * t[40][1]
        on = f8 < 0.85 * bf
        if on:
            TUNE_ENV["ARC_LMHEAD_FP8"] = "1"
        log("lmhead", r.stdout.strip().replace("\n", " | "), f"step bf16 {bf:.3f} fp8 {f8:.3f} ms ->", "ON" if on else "off")
    except Exception as ex:
        log("lmhead gate failed (bf16 kept)", repr(ex)[:200], (r.stdout[-200:] + r.stderr[-300:]) if 'r' in dir() else "")

if MODE.get("idx_gate"):   # sharded indexer decode logits variant (arcstats.shard_logits_v2) only if faster here
    try:
        menv = MODE.get("env", {})
        env_i = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", BENCH_IDX_QUICK="1",
                     ARC_SHARD_KLOC=menv.get("ARC_SHARD_KLOC", "768"))
        r = subprocess.run([PY, str(code / "bench_idx_l4.py"), "40", menv.get("ARC_SHARD_BS", "80")], env=env_i,
                           timeout=900, capture_output=True, text=True, cwd=str(code))
        res = {}
        for m in re.finditer(r"^idx (orig_row|(?:row8|grp) [^\n]*?) ([0-9.]+) ms maxrel ([0-9.e+-]+) infmis (\d+)$",
                             r.stdout, re.M):
            if float(m.group(3)) < 1e-3 and int(m.group(4)) == 0:
                res[m.group(1)] = float(m.group(2))
        for l in r.stdout.splitlines():
            if l.startswith("idx"):
                log("idx_gate", l)
        t_o = res.pop("orig_row", None)
        best = min((t, nm) for nm, t in res.items()) if res else None
        on = t_o and best and best[0] < 0.9 * t_o
        if on:   # "grp f8=0 BN=128 KB=2 w=4" -> "grp,f8=0,BN=128,KB=2,w=4"
            TUNE_ENV["ARC_SHARD_V2"] = ",".join(best[1].split())
            if MODE.get("idx_f16out"):   # fp16 logits buffer: ~1.8x faster local top-k (decode KL at noise level)
                TUNE_ENV["ARC_SHARD_F16OUT"] = "1"
        log("idx_gate", f"orig {t_o} ms best {best} ->", TUNE_ENV.get("ARC_SHARD_V2", "orig"),
            "" if res else (r.stdout[-300:] + r.stderr[-500:]))
    except Exception as ex:
        log("idx gate failed (orig kept)", repr(ex)[:200])

if MODE.get("topk_gate"):   # exact radix top-k for the sharded indexer (arcstats.fast_topk) only if exact and >=5% faster
    try:
        menv = MODE.get("env", {})
        env_k = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", BENCH_TOPK_QUICK="1",
                     ARC_SHARD_KLOC=menv.get("ARC_SHARD_KLOC", "768"))
        r = subprocess.run([PY, str(code / "bench_topk_l4.py"), "63", menv.get("ARC_SHARD_BS", "128"),
                            str(MODE.get("topk_gate_mml", 262144))], env=env_k, timeout=900, capture_output=True,
                           text=True, cwd=str(code))
        out = r.stdout
        for l in out.splitlines():
            if l.startswith("topk"):
                log("topk_gate", l)
        ex = re.search(r"^topk exact1 (\d) exact2 (\d)", out, re.M)
        # with the shard_logits_v2 kernel the fast path also drops the logits' -inf tail stores: count that in
        pat = r"^topk orig ([0-9.]+) ms fast1 ([0-9.]+) ms fast2 ([0-9.]+) ms" if TUNE_ENV.get("ARC_SHARD_V2") or \
            menv.get("ARC_SHARD_V2") else r"^topk sel orig ([0-9.]+) ms fast1 ([0-9.]+) ms fast2 ([0-9.]+) ms"
        tm = re.search(pat, out, re.M)
        bm = re.search(r"^topk best nt=(\d+) agg=(\d+) mnt=(\d+)", out, re.M)
        t_o, t_1, t_2 = (float(tm.group(i)) for i in (1, 2, 3))
        cand = [(t, lvl) for t, lvl, ok in ((t_1, 1, ex.group(1) == "1"), (t_2, 2, ex.group(2) == "1"))
                if ok and t < 0.95 * t_o and lvl <= int(MODE.get("topk_gate_max_level", 2))]
        if cand:
            lvl = min(cand)[1]
            TUNE_ENV.update(ARC_FAST_TOPK=str(lvl), ARC_FAST_TOPK_NT=bm.group(1), ARC_FAST_TOPK_AGG=bm.group(2),
                            ARC_FAST_TOPK_MNT=bm.group(3))
        log("topk_gate", f"orig {t_o} ms fast1 {t_1} ms fast2 {t_2} ms exact {ex.group(1)}{ex.group(2)} ->",
            {k: v for k, v in TUNE_ENV.items() if k.startswith("ARC_FAST_TOPK")} or "torch.topk kept")
    except Exception as ex_:
        log("topk gate failed (torch.topk kept)", repr(ex_)[:200], (r.stdout[-300:] + r.stderr[-500:]) if 'r' in dir() else "")

for nb in MODE.get("host_ar_blocks_sweep", []):   # diagnostics: host all-reduce vs blocks per rank (log only)
    try:
        env_b = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, NCCL_P2P_DISABLE="1", ARC_HOST_AR_BLOCKS=str(nb))
        r = run_bench([PY, str(code / "bench_host_ar.py"), "4", "64,160,256,1024"], env_b, 600, cwd=str(code))
        for l in r.stdout.splitlines():
            if l.startswith("{"):
                log(f"host_ar_nb{nb}", l)
    except Exception as ex:
        log("host_ar blocks sweep failed", nb, repr(ex)[:200])
if MODE.get("offload_tune"):   # diagnostics: offload decode attention tile sweep on this GPU (log only)
    try:
        env_o = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0", ARC_OFFLOAD_WAYS="8",
                     ARC_OFFLOAD_RESOLVE="1", **{k: str(v) for k, v in MODE.get("offload_tune_env", {}).items()})
        r = subprocess.run([PY, str(code / "tune_offload_l4occ.py"), "split"], env=env_o, timeout=3000,
                           capture_output=True, text=True, cwd=str(code))
        for l in r.stdout.splitlines():
            log("offload_tune", l)
        if not r.stdout:
            log("offload_tune stderr", r.stderr[-800:])
    except Exception as ex:
        log("offload tune failed", repr(ex)[:200])
for bc in MODE.get("bench_cmds", []):   # diagnostics: {"cmd": [script, args...], "env": {...}} on GPU 0 (log only)
    try:
        env_b = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                     **{k: str(v).replace("$W", str(W)) for k, v in bc.get("env", {}).items()})
        hb = start_hog(bc["hog"], bc["cmd"][0]) if bc.get("hog") else None   # per-bench hog (link emulation sweeps)
        try:
            r = run_bench([PY, str(code / bc["cmd"][0])] + [str(a).replace("$W", str(W)) for a in bc["cmd"][1:]],
                          env_b, bc.get("timeout", 2400), cwd=str(code))
        finally:
            stop_hog(hb, bc["cmd"][0])
        if bc.get("hog"):
            log("bench", bc["cmd"][0], "with link_hog", json.dumps(bc["hog"]))
        for l in r.stdout.splitlines():
            log("bench", bc["cmd"][0], l)
        if r.returncode:
            log("bench", bc["cmd"][0], "rc", r.returncode, r.stderr[-800:])
    except Exception as ex:
        log("bench failed", bc.get("cmd"), repr(ex)[:200])
if MODE.get("shard_sample"):   # verify sampling on the vocab shard, no logits all-gather (arcstats.shard_sample)
    TUNE_ENV["ARC_SHARD_SAMPLE"] = "1"
if MODE.get("mla_qcat"):   # MLA decode query concat in place (arcstats.qcat, bit-identical)
    TUNE_ENV["ARC_MLA_QCAT"] = "1"
if MODE.get("draft_local_argmax"):   # MTP drafts: per-rank argmax + (value, id) all-gather instead of the logits
    for i_, a_ in enumerate(MODE["args"][:-1]):
        if a_ == "--speculative-config":
            MODE["args"][i_ + 1] = json.dumps(dict(json.loads(MODE["args"][i_ + 1]), use_local_argmax_reduction=True))
if MODE.get("gate_merge_gate") and TUNE_ENV.get("ARC_FP8_W8A8") == "1":
    # g_proj merged into the fused q/kv-a GEMM (arcstats.gmerge): ON only if bit-exact here and >= 5% faster per step
    try:
        env_g = dict(os.environ, CUDA_HOME=CUDA_HOME, PYTHONPATH=SP, CUDA_VISIBLE_DEVICES="0",
                     ARC_W8A8_CONFIG_DIR=TUNE_ENV.get("ARC_W8A8_CONFIG_DIR", ""),
                     GMERGE_DECODE_M=str(4 * int(MODE.get("gate_merge_seqs", 64))))
        r = run_bench([PY, str(code / "test_gmerge.py"), "1,16,64,256,1024"], env_g, 600, cwd=str(code))
        g = [l for l in r.stdout.splitlines() if l.startswith("GMERGE_GATE")]
        for l in r.stdout.splitlines():
            if l.startswith("gmerge") and ("us" in l or "!=" in l):
                log("gate_merge", l)
        on = bool(g) and g[-1].endswith("ok True")
        if on:
            TUNE_ENV["ARC_GATE_MERGE"] = "1"
        log("gate_merge_gate ->", "ON" if on else "off", g[-1] if g else (r.stdout[-300:] + r.stderr[-600:]))
    except Exception as ex:
        log("gate_merge gate failed (off)", repr(ex)[:200])
if MODE.get("gates_only"):   # startup benchmarks only (no checkpoint, no server)
    log("gates_only: done", json.dumps(TUNE_ENV))
    sys.exit(0)

# ---------------------------------------------------------------- server
ckpt = str(_ckpt_dir())
env = dict(os.environ, **{k: str(v).replace("$W", str(W)) for k, v in MODE.get("env", {}).items()}, **TUNE_ENV, CUDA_HOME=CUDA_HOME, NCCL_P2P_DISABLE="1",
           LIBRARY_PATH=f"{CULINK}:{CU13_LIB}:{CUDA_HOME}/lib:" + os.environ.get("LIBRARY_PATH", ""),
           VLLM_USE_FLASHINFER_SAMPLER="0",   # PyTorch top-k/top-p sampler: no JIT build at startup
           VLLM_ALLREDUCE_USE_FLASHINFER="0", VLLM_ALLREDUCE_USE_SYMM_MEM="0",
           # vLLM 0.30 applies this only to the engine->client socket handshake AFTER the engine reported READY (model
           # load is waited for without a timeout, v1/engine/utils.wait_for_engine_startup); the guard bounds startup itself
           VLLM_ENGINE_READY_TIMEOUT_S="43200" if SG else "5400",
           PATH=f"/tmp/arc_py312/bin:{CUDA_HOME}/bin:{os.environ['PATH']}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
ckpt_serve = ckpt
if os.environ.get("ARC_STAGE_DIR") and Path(os.environ["ARC_STAGE_DIR"]).is_dir():
    # MODE["prefetch"]["stage"]: serve from the local mirror (same files; symlinks to the mount until copied). The delta
    # overlay and the safetensors are first in the copy order; wait for them here (bounded), the TP ranks wait per
    # lowbit-L*.pt file (arcstats.stage_wait)
    ckpt_serve = os.environ["ARC_STAGE_DIR"]
    _sw0 = time.time()
    _need = [p for p in Path(ckpt_serve).glob("*.safetensors")] + \
        ([Path(os.environ["ARC_STAGE_DELTA"])] if os.environ.get("ARC_STAGE_DELTA") else [])
    _swmax = float(MODE["prefetch"].get("stage_wait_s", 300)) if isinstance(MODE.get("prefetch"), dict) else 300.0
    if S0 is not None:   # guard: as long as the copier runs (it ends at the deadline); 4 ranks reading the mount is slower
        _swmax = max(_swmax, deadline_epoch() - 600 - time.time())
    _pp = int(os.environ.get("ARC_PREFETCH_PID") or 0)
    while any(p.is_symlink() for p in _need) and (Path(ckpt_serve) / ".staging").exists() and time.time() - _sw0 < _swmax \
            and (S0 is None or not _pp or _pid_alive(_pp)):
        time.sleep(0.5)
    log("stage: serving from", ckpt_serve, f"(waited {time.time() - _sw0:.0f} s;",
        f"{sum(not p.is_symlink() for p in _need)}/{len(_need)} early files local)")
args = [PY, "-m", "vllm.entrypoints.cli.main", "serve", ckpt_serve, "--port", "8000", "--served-model-name", "m",
        "--tensor-parallel-size", "4", "--gpu-memory-utilization", os.environ.get("ARC_UTIL", str(MODE.get("util", 0.95)))] + MODE["args"]
if MODE.get("draft_vocab") and (code / MODE["draft_vocab"]).exists():   # MTP drafts from a frequent-token subset
    env["ARC_DRAFT_VOCAB"] = str(code / MODE["draft_vocab"])
if MODE.get("delta_file"):   # in-format fine-tune overlay on the base checkpoint (arcstats/delta_overlay.py)
    _df = find(IN + "/**/" + MODE["delta_file"])                 # code bundle or any attached input
    if os.environ.get("ARC_STAGE_DELTA") and ckpt_serve != ckpt and Path(os.environ["ARC_STAGE_DELTA"]).exists():
        _df = os.environ["ARC_STAGE_DELTA"]                         # staged local copy (or its symlink: same bytes)
    env["ARC_DELTA_FILE"] = _df
    log("delta overlay", _df, f"{Path(_df).stat().st_size / 2**30:.2f} GiB")
if MODE.get("kv_cache_gib"):   # explicit KV pool size (the util/graph-profiling estimate left ~1.2 GiB/GPU idle on L4)
    kv_gib = float(MODE["kv_cache_gib"])
    if TUNE_ENV.get("ARC_LMHEAD_FP8") == "1":   # the FP8 LM head frees ~0.18 GiB/GPU of bf16 weight: give it to the KV pool
        kv_gib += float(MODE.get("kv_bonus_lmhead_gib", 0))
    if MODE.get("env", {}).get("ARC_MLA_FP8") == "1" and TUNE_ENV.get("ARC_FP8_W8A8") == "1":   # FP8 MLA matmuls free
        kv_gib += float(MODE.get("kv_bonus_mla_gib", 0))                                     # ~0.21 GiB/GPU of bf16
    log("kv pool", f"{kv_gib:.2f} GiB/GPU")
    args += ["--kv-cache-memory", str(int(kv_gib * 2 ** 30))]
nsys = None
if MODE.get("nsys_at_s") is not None:
    try:
        if not Path("/tmp/nsys").exists():
            sh(f"dpkg-deb -x {find(IN + '/**/nsys_*.debbin')} /tmp/nsys_pkg && ln -s $(dirname $(find /tmp/nsys_pkg -name nsys -type f -path '*target-linux-x64*' | head -1)) /tmp/nsys")
        nsys = "/tmp/nsys/nsys"
        args = [nsys, "launch", "--session-new=arc", "--trace=cuda,nvtx,osrt", "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node"] + args
    except Exception as ex:
        log("nsys unavailable", repr(ex)[:200]); nsys = None
if os.environ.get("ARC_PREFETCH_PID") and (W / "prefetch_log.txt").exists():   # how far the page-cache prefetch got
    log("prefetch at server start:", ((W / "prefetch_log.txt").read_text().strip().splitlines() or ["-"])[-1])
SRV_PRE = None
if MODE.get("cpu_slow"):   # emulation only, never in a submission mode: slow-vCPU stand-in
    # the server tree (API server, EngineCore, TP workers) is confined to the first `ncores` allowed CPUs, and each of those
    # CPUs also runs `hogs_per_core` busy loops, so every server thread gets ~1/(1+hogs) of a core
    _cs = MODE["cpu_slow"]
    _cpus = sorted(os.sched_getaffinity(0))[:int(_cs.get("ncores", 8))]
    _hogs = []
    for _c in _cpus * int(_cs.get("hogs_per_core", 1)):
        _hogs.append(subprocess.Popen([PY, "-c", "while True: pass"], start_new_session=True,
                                      preexec_fn=lambda c=_c: os.sched_setaffinity(0, {c})))
    SRV_PRE = lambda: os.sched_setaffinity(0, set(_cpus))
    import atexit
    atexit.register(lambda: [h.kill() for h in _hogs])
    log("cpu_slow", json.dumps(_cs), "server cpus", _cpus, "hogs", len(_hogs))
srv_log = open(W / "server_log.txt", "w")
srv = subprocess.Popen(args, env=env, stdout=srv_log, stderr=subprocess.STDOUT, preexec_fn=SRV_PRE,
                       **({"start_new_session": True} if SG else {}))   # guard: own session, so its tree can be killed
log("server pid", srv.pid)
import urllib.request


def get(url, timeout=10):
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def _gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def _kill_server_tree(p):
    """(guard) SIGKILL the server tree (own session: API server + EngineCore + TP workers, plus any descendant that left
    the group), wait (<= 180 s) until the GPUs are released, and remove the offload host-row files (/dev/shm/arcoff_*,
    ~57 GiB) whose owner is gone, so a restart does not double the host memory."""
    pids = _descendants(p.pid)
    for k in [p.pid] + pids:
        try:
            os.kill(k, 9)
        except OSError:
            pass
    try:
        os.killpg(p.pid, 9)
    except OSError:
        pass
    try:
        p.wait(60)
    except Exception:
        pass
    t = time.time()
    while time.time() - t < 180:
        u = _gpu_used_mib()
        if u and max(u) < 1500:
            break
        time.sleep(3)
    freed = 0
    for f in glob.glob("/dev/shm/arcoff_*"):
        m = re.match(r"arcoff_(\d+)_", os.path.basename(f))
        if m and not _pid_alive(int(m.group(1))):
            try:
                sz = os.path.getsize(f); os.unlink(f); freed += sz
            except OSError:
                pass
    log("server tree killed:", len(pids) + 1, "processes,", f"gpu used MiB {_gpu_used_mib()} after {time.time() - t:.0f} s,",
        f"removed {freed / 2**30:.1f} GiB of dead-engine offload files")


t_start = time.time()
if not SG:
    while True:
        try:
            get("http://127.0.0.1:8000/v1/models"); break
        except Exception:
            if srv.poll() is not None:
                log("server died"); sys.exit(0 if MODE.get("workload") == "submit" else 1)   # keep the fallback submission
            time.sleep(float(MODE.get("ready_poll_s", 15)))   # the design notes: 15 s idles 7.5 s on average
else:
    # guard: a server that dies or hangs before it is ready is restarted (retries, if retry_min_left_s remain: a restart
    # reloads from the local staged copy); when the solver could no longer get solve_min_s, give up with the fallback
    _n, _emul, _t_srv, _slog = 0, list(SG.get("emul_kill_s") or []), time.time(), W / "server_log.txt"
    _last, _last_prog = None, time.time()
    while True:
        try:
            get("http://127.0.0.1:8000/v1/models"); break
        except Exception:
            pass
        _now = time.time()
        _why = f"died rc {srv.returncode}" if srv.poll() is not None else None
        if _why is None and _n < len(_emul) and _emul[_n] is not None and _now - _t_srv >= float(_emul[_n]):
            log(f"TEST emul_kill_s: killing the server tree {_now - _t_srv:.0f} s after start {_n + 1}")
            _kill_server_tree(srv); _why = "killed (test)"
        try:
            _cur = (os.path.getsize(_slog), _stage_progress()[0])
        except OSError:
            _cur = None
        if _cur != _last:
            _last, _last_prog = _cur, _now
        _staging = bool(os.environ.get("ARC_STAGE_DIR")) and (Path(os.environ["ARC_STAGE_DIR"]) / ".staging").exists()
        if _why is None and not _staging and _now - _last_prog > float(SG.get("hang_s", 2400)):
            # (while the copy runs a quiet server is waiting for files: a restart could not help, the deadline bounds it)
            _why = f"startup hang (no server log / staging progress for {_now - _last_prog:.0f} s)"
        _left = deadline_epoch() - _now
        if _why is None and _left < float(SG.get("solve_min_s", 120)):
            _why = f"not ready with {_left:.0f} s left to the solver deadline"
            _n = 1 << 30   # no retry
        if _why is not None:
            if _n < int(SG.get("retries", 1)) and _left > float(SG.get("retry_min_left_s", 1800)):
                _n += 1
                log("server", _why, f"-> restart {_n} ({_left:.0f} s left to the solver deadline)")
                _kill_server_tree(srv)
                _slog = W / f"server_log_r{_n}.txt"
                srv = subprocess.Popen(args, env=env, stdout=open(_slog, "w"), stderr=subprocess.STDOUT,
                                       preexec_fn=SRV_PRE, start_new_session=True)
                log("server pid", srv.pid, f"(restart {_n})")
                _t_srv, _last, _last_prog = time.time(), None, time.time()
                continue
            log("server", _why, f"-> no restart ({_left:.0f} s left): fallback submission,", ensure_submission())
            _kill_server_tree(srv)
            log("children killed at exit:", _kill_all_children())
            sys.exit(0 if MODE.get("workload") == "submit" else 1)
        time.sleep(float(MODE.get("ready_poll_s", 15)))
log(f"server ready after {time.time() - t_start:.0f}s" + (f" (S0 + {time.time() - S0:.0f} s)" if S0 is not None else ""))
save_triton_cache("after server ready")


def scrape():
    m = get("http://127.0.0.1:8000/metrics")
    g = lambda k: sum(float(x) for x in re.findall(rf"^{k}(?:{{[^}}]*}})? ([0-9.e+]+)", m, re.M))
    return {"t": round(time.time() - T0, 1), "gen": g("vllm:generation_tokens_total"), "prompt": g("vllm:prompt_tokens_total"),
            "running": g("vllm:num_requests_running"), "waiting": g("vllm:num_requests_waiting"),
            "kv": g("vllm:kv_cache_usage_perc"), "acc": g("vllm:spec_decode_num_accepted_tokens_total"),
            "drafts": g("vllm:spec_decode_num_drafts_total"), "preempt": g("vllm:num_preemptions_total")}


import httpx  # noqa: E402

mlog = open(W / "metrics.jsonl", "w")


def host_stats():
    """host resources a 12 h run could exhaust: RAM (incl. the container cgroup), /dev/shm, disks"""
    r = {}
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        r["mem_avail_gib"] = round(mi["MemAvailable"] / 2**20, 2)
        for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            if os.path.exists(f):
                r["cg_mem_gib"] = round(int(open(f).read()) / 2**30, 2); break
        for k, d in (("shm", "/dev/shm"), ("tmp", "/tmp"), ("work", str(W))):
            st = os.statvfs(d)
            r[k + "_used_gib"] = round((st.f_blocks - st.f_bfree) * st.f_frsize / 2**30, 2)
            r[k + "_free_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.split("\n")
        g = [[int(x) for x in l.split(",")] for l in g if l.strip()]
        r["gpu_used_mib"] = [u for u, _ in g]
        r["gpu_free_mib_min"] = min(t - u for u, t in g)
    except Exception:
        pass
    return r


async def monitor(stop):
    n = 0
    while not stop.is_set():
        try:
            rec = scrape()
            if n % 12 == 0:
                rec.update(host_stats())
            n += 1
            mlog.write(json.dumps(rec) + "\n"); mlog.flush()
        except Exception as e:
            log("metrics error", repr(e)[:100])
        try:
            await asyncio.wait_for(stop.wait(), 5)
        except asyncio.TimeoutError:
            pass


async def sweep():
    res = open(W / "sweep.jsonl", "w")
    vocab_lo, vocab_hi = 1000, 100000
    async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 3600.0)) as c:
        for case in MODE["sweep"][: int(os.environ.get("ARC_SWEEP_LIMIT", "999"))]:
            B, P, G = case["batch"], case["prompt"], case["gen"]
            m0 = scrape(); t0 = time.time()
            bodies = [{"model": "m", "prompt": [random.randint(vocab_lo, vocab_hi) for _ in range(P)], "max_tokens": G,
                       "ignore_eos": True, "temperature": 0.7, "top_p": 1.0} for _ in range(B)]
            if nsys and case.get("nsys"):
                async def capture():
                    await asyncio.sleep(case["nsys"])
                    await asyncio.to_thread(subprocess.run, [nsys, "start", "--session=arc", "-o", str(W / "trace"),
                                                             "--force-overwrite=true", "--sample=none", "--cpuctxsw=none"])
                    await asyncio.sleep(case.get("nsys_len", 30))
                    await asyncio.to_thread(subprocess.run, [nsys, "stop", "--session=arc"])
                    log("nsys window captured")
                cap = asyncio.create_task(capture())
            rs = await asyncio.gather(*[c.post("http://127.0.0.1:8000/v1/completions", json=b) for b in bodies],
                                      return_exceptions=True)
            m1 = scrape(); dt = time.time() - t0
            ok = sum(1 for r in rs if not isinstance(r, Exception) and r.status_code == 200)
            rec = {**case, "ok": ok, "wall_s": round(dt, 1), "gen_tok": m1["gen"] - m0["gen"],
                   "prompt_tok": m1["prompt"] - m0["prompt"], "drafts": m1["drafts"] - m0["drafts"],
                   "accepted": m1["acc"] - m0["acc"], "preempt": m1["preempt"] - m0["preempt"]}
            res.write(json.dumps(rec) + "\n"); res.flush(); log("sweep", json.dumps(rec))


async def submit():
    """Competition workload: deadline-aware solver (solve.py) writing /kaggle/working/submission.json."""
    split = MODE.get("challenges", "test")
    data = Path(find(IN + "/**/arc-agi_" + split + "_challenges.json")).parent
    out = W / ("submission.json" if split == "test" else f"smoke_{split}.json")   # never overwrite with eval answers
    sol = data / f"arc-agi_{split}_solutions.json"
    left = MODE.get("deadline_total_s", 41400) - (time.time() - T0)
    if S0 is not None:   # guard: whatever remains until the S0-anchored deadline
        left = deadline_epoch() - time.time()
        if left < float(SG.get("solve_min_s", 120)):
            log(f"solver skipped: {left:.0f} s left; submission", ensure_submission())
            return
    cmd = [PY, str(code / "solve.py"), "--port", "8000", "--challenges", str(data / f"arc-agi_{split}_challenges.json"),
           "--out", str(out), "--log", str(W / "solve_log.txt"), "--template",
           str(code / "system_prompt.txt"), "--tokenizer", ckpt, "--deadline-s", str(int(left))] + \
          [str(x) for x in MODE.get("solve_args", [])] + (["--solutions", str(sol)] if sol.exists() and MODE.get("score") else []) + \
          (["--tasks", str(code / MODE["tasks_file"])] if MODE.get("tasks_file") else [])
    log("solver", "RERUN" if RERUN else "save-run", " ".join(cmd),
        f"(S0 + {time.time() - S0:.0f} s, deadline S0 + {deadline_epoch() - S0:.0f} s)" if S0 is not None else "")
    p = await asyncio.create_subprocess_exec(*cmd)
    await p.wait()
    if S0 is not None:
        log("solver exit rc", p.returncode, "| submission:", ensure_submission())


async def load_sweep():
    """Throughput-vs-concurrency ramp (load_sweep.py) on the live server: one run answers 'how many sequences'."""
    split = MODE.get("challenges", "evaluation")
    if MODE.get("ctx_file"):   # long-context emulation: per-request decode context lengths from a real workload
        src = ["--ctx-file", str(code / MODE["ctx_file"])]
    elif MODE.get("synthetic_prompt_tokens"):
        src = ["--synthetic", str(MODE["synthetic_prompt_tokens"])]
    else:
        data = Path(find(IN + "/**/arc-agi_" + split + "_challenges.json")).parent
        src = ["--challenges", str(data / f"arc-agi_{split}_challenges.json"), "--template", str(code / "system_prompt.txt")]
    cmd = [PY, str(code / "load_sweep.py"), "--port", "8000", *src, "--tokenizer", ckpt, "--out", str(W / "load_sweep.json")] + \
          [str(x) for x in MODE.get("sweep_args", [])]
    log("load_sweep", " ".join(cmd))
    p = await asyncio.create_subprocess_exec(*cmd)
    await p.wait()


async def logprob_dump():
    """Serving-fidelity A/B (logprob_dump.py): MODE["dump_ids"] = staged sequence file to re-score, else generate."""
    src = ["--ids", str(code / MODE["dump_ids"])] if MODE.get("dump_ids") else ["--gen", str(W / "fid_seqs.json")]
    cmd = [PY, str(code / "logprob_dump.py"), "--port", "8000", "--tokenizer", ckpt, *src, "--out", str(W / "fid.npz")] + \
          [str(x) for x in MODE.get("dump_args", [])]
    log("logprob_dump", " ".join(cmd))
    p = await asyncio.create_subprocess_exec(*cmd)
    await p.wait()


async def replay():
    """Closed-loop long-generation replay (replay.py) with solve.py's admission policy: KV pressure / preemptions."""
    cmd = [PY, str(code / "replay.py"), "--port", "8000", "--tokenizer", ckpt, "--lengths", str(code / "req_lengths.json"),
           "--out", str(W / "replay.json")] + [str(x) for x in MODE.get("replay_args", [])]
    log("replay", " ".join(cmd))
    p = await asyncio.create_subprocess_exec(*cmd)
    await p.wait()


async def nsys_window():
    """MODE["nsys_window"] = [start_s, length_s] (seconds since T0): profile the live workload once.
    MODE["nsys_window_after_prefill"] = s: instead start s seconds after the server is in pure decode (requests running,
    none waiting, prompt-token counter unchanged for 60 s), for MODE["nsys_len_s"] seconds."""
    if MODE.get("nsys_window_after_prefill") is not None and MODE.get("workload") == "load_sweep":
        ln = MODE.get("nsys_len_s", 20)
        while not (W / "load_sweep.json.decode").exists():   # load_sweep: every request has its first token
            await asyncio.sleep(10)
        log("nsys: pure decode reached (load_sweep marker)")
        await asyncio.sleep(MODE["nsys_window_after_prefill"])
    elif MODE.get("nsys_window_after_prefill") is not None:
        ln = MODE.get("nsys_len_s", 20)
        last, stable = None, 0
        while True:
            await asyncio.sleep(10)
            try:
                mt = await asyncio.to_thread(get, "http://127.0.0.1:8000/metrics")
                v = lambda n: sum(float(x) for x in re.findall(rf"^vllm:{n}(?:{{[^}}]*}})? ([0-9.e+-]+)", mt, re.M))
                cur = (v("num_requests_running"), v("num_requests_waiting"), v("prompt_tokens_total"))
            except Exception:
                continue
            stable = stable + 10 if (last and cur[0] > 0 and cur[1] == 0 and cur[2] == last[2]) else 0
            last = cur
            if stable >= 60:
                break
        log("nsys: pure decode reached", cur)
        await asyncio.sleep(MODE["nsys_window_after_prefill"])
    else:
        st, ln = MODE["nsys_window"]
        await asyncio.sleep(max(0.0, st - (time.time() - T0)))
    await asyncio.to_thread(subprocess.run, [nsys, "start", "--session=arc", "-o", str(W / "trace"), "--force-overwrite=true",
                                             "--sample=none", "--cpuctxsw=none"])
    await asyncio.sleep(ln)
    await asyncio.to_thread(subprocess.run, [nsys, "stop", "--session=arc"])
    log("nsys window captured")


async def main():
    stop = asyncio.Event()
    mon = asyncio.create_task(monitor(stop))
    if nsys and (MODE.get("nsys_window") or MODE.get("nsys_window_after_prefill") is not None):
        asyncio.create_task(nsys_window())
    if MODE.get("workload", "sweep") == "sweep":
        await sweep()
    elif MODE["workload"] == "submit":
        await submit()
    elif MODE["workload"] == "load_sweep":
        await load_sweep()
    elif MODE["workload"] == "replay":
        await replay()
    elif MODE["workload"] == "logprob_dump":
        await logprob_dump()
    stop.set(); await mon

asyncio.run(main())
log("workload done")
save_triton_cache("after workload")
srv.terminate()
try:
    srv.wait(120)
except Exception:
    srv.kill()
if SG:   # own session: make sure no worker outlives the run
    try:
        os.killpg(srv.pid, 9)
    except OSError:
        pass
if nsys and (W / "trace.nsys-rep").exists():
    for rep in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum"):
        try:
            sh(f"{nsys} stats --force-export=true --report {rep} --format csv -o {W}/nsys_{rep} {W}/trace.nsys-rep",
               timeout=1800)
        except Exception as ex:
            log("nsys stats failed", rep, repr(ex)[:120])
log("done")
