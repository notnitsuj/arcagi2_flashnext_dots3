"""ARC_STEPLOG=<dir> (or 1 = $ARC_WORK/steplog): per-step speculative-decoding log from the scheduler (EngineCore process), for offline analysis
of draft acceptance. Read-only: it never changes scheduling or outputs.

Files in <dir> (raw little-endian int64, appended in blocks):
  steps.i64   per update_from_output call: [step, t_us, num_reqs, total_scheduled_tokens, total_draft_tokens]
  reqs.i64    per (step, request): [step, rid, pos_before, n_new, n_draft, n_sched]
              pos_before = request.num_tokens (prompt + real output tokens) before this step's tokens are appended,
              n_new = tokens appended by this step (accepted drafts + bonus), n_draft = drafts verified this step.
  toks.i64    per freed request: [rid, n, tok_0 .. tok_{n-1}] (request.all_token_ids at finish)
  rids.tsv    rid -> request id string
"""
import os
import time

import numpy as np

DIR = os.environ.get("ARC_STEPLOG", "")
if DIR == "1":   # mode files: log next to the run's other outputs
    DIR = os.path.join(os.environ.get("ARC_WORK", "/tmp"), "steplog")
FLUSH = int(os.environ.get("ARC_STEPLOG_FLUSH", "200"))


class _Log:
    def __init__(self):
        os.makedirs(DIR, exist_ok=True)
        tag = f".{os.getpid()}"
        self.f_steps = open(os.path.join(DIR, f"steps{tag}.i64"), "ab")
        self.f_reqs = open(os.path.join(DIR, f"reqs{tag}.i64"), "ab")
        self.f_toks = open(os.path.join(DIR, f"toks{tag}.i64"), "ab")
        self.f_rids = open(os.path.join(DIR, f"rids{tag}.tsv"), "a")
        self.rid = {}
        self.step = 0
        self.steps, self.reqs = [], []
        self.t0 = time.monotonic()
        with open(os.path.join(DIR, f"t0{tag}.txt"), "w") as f:   # wall clock of t_us = 0 (aligns steps with run windows)
            f.write(f"{time.time():.6f} {os.environ.get('ARC_T0', '0')}\n")

    def rid_of(self, req_id):
        r = self.rid.get(req_id)
        if r is None:
            r = self.rid[req_id] = len(self.rid)
            self.f_rids.write(f"{r}\t{req_id}\n")
        return r

    def flush(self):
        if self.steps:
            np.asarray(self.steps, dtype=np.int64).tofile(self.f_steps); self.steps.clear()
        if self.reqs:
            np.asarray(self.reqs, dtype=np.int64).tofile(self.f_reqs); self.reqs.clear()
        for f in (self.f_steps, self.f_reqs, self.f_toks, self.f_rids):
            f.flush()


def install():
    from vllm.v1.core.sched import scheduler as sch
    L = [None]

    def log():
        if L[0] is None:
            L[0] = _Log()
        return L[0]

    orig_update = sch.Scheduler.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        lg = log()
        spec = scheduler_output.scheduled_spec_decode_tokens or {}
        before = []
        for req_id, n_sched in scheduler_output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req is not None:
                before.append((req_id, req, req.num_tokens, len(spec.get(req_id, ())), n_sched))
        out = orig_update(self, scheduler_output, model_runner_output)
        s = lg.step
        tot_d = 0
        for req_id, req, pos, nd, ns in before:
            tot_d += nd
            lg.reqs.append((s, lg.rid_of(req_id), pos, req.num_tokens - pos, nd, ns))
        lg.steps.append((s, int((time.monotonic() - lg.t0) * 1e6), len(before),
                         scheduler_output.total_num_scheduled_tokens, tot_d))
        lg.step += 1
        if lg.step % FLUSH == 0:
            lg.flush()
        return out

    orig_free = sch.Scheduler._free_request

    def _free_request(self, request, *a, **kw):
        try:
            lg = log()
            toks = np.asarray(list(request.all_token_ids), dtype=np.int64)
            np.concatenate([[lg.rid_of(request.request_id), toks.size], toks]).astype(np.int64).tofile(lg.f_toks)
        except Exception:
            pass
        return orig_free(self, request, *a, **kw)

    sch.Scheduler.update_from_output = update_from_output
    sch.Scheduler._free_request = _free_request
    import atexit
    atexit.register(lambda: L[0] is not None and L[0].flush())
