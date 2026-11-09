"""ARC_STAGE_DIR (set by run_l4.py MODE["prefetch"]["stage"], default off): the checkpoint is
served from a local mirror whose files start as symlinks to the (FUSE) input mount and are replaced by local copies,
in load order, by the prefetcher. wait_staged(path) blocks while `path` is still a symlink and the copy is running
(marker <ARC_STAGE_DIR>/.staging present, prefetcher pid alive), at most ARC_STAGE_WAIT_S (default 1800) s; then the
caller opens whatever is there (a local copy, or the symlink = the old mount read). Same bytes either way."""
import os
import time


def wait_staged(path):
    sd = os.environ.get("ARC_STAGE_DIR")
    if not sd or not os.path.islink(path):
        return 0.0
    marker = os.path.join(sd, ".staging")
    pid = int(os.environ.get("ARC_PREFETCH_PID") or 0)
    t0, tmax = time.time(), float(os.environ.get("ARC_STAGE_WAIT_S", "1800"))
    while os.path.islink(path) and os.path.exists(marker) and time.time() - t0 < tmax:
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            except OSError:
                pass
        time.sleep(0.2)
    return time.time() - t0
