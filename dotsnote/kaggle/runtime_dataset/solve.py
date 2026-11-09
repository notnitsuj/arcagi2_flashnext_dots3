"""Deadline-aware ARC-AGI-2 solver against a local vLLM server (the Kaggle submission workload).

  python solve.py --port 8000 --challenges arc-agi_test_challenges.json --out submission.json --deadline-s 39000

Every test input gets a first attempt; second attempts are scheduled only while time remains. The token cap of each
new attempt adapts to the measured generation rate so all queued attempts can finish before the deadline. An attempt
that hits its cap while still thinking gets a short forced conclusion ("</think>" + answer) on top of its own
context (prefix cached). At the deadline everything is aborted and submission.json is written; inputs without a
parsed grid fall back to the test input itself. submission.json is rewritten after every finished attempt.
Prompt and parser follow the ARC Prize benchmark harness (arc/direct.py).
"""
import argparse, asyncio, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_detect import LoopDetector  # noqa: E402  (ship dots3/loop_detect.py next to solve.py)
import os  # noqa: E402
try:
    from conf_gate import ConfGate  # noqa: E402  (ship dots3/conf_gate.py next to solve.py)
except ImportError:
    ConfGate = None

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--challenges", required=True)
ap.add_argument("--solutions", default=None, help="optional, for local scoring")
ap.add_argument("--tasks", default=None, help="optional file of task ids to restrict to")
ap.add_argument("--out", required=True)
ap.add_argument("--log", default=None)
ap.add_argument("--template", required=True, help="ARC benchmark system prompt template")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--deadline-s", type=float, required=True, help="seconds from start until the submission must be final")
ap.add_argument("--concurrency", type=int, default=48)
ap.add_argument("--cap1", type=int, default=240000, help="max tokens for a first attempt (upper bound)")
ap.add_argument("--cap2", type=int, default=120000, help="max tokens for a second attempt (upper bound)")
ap.add_argument("--min-cap", type=int, default=24000)
ap.add_argument("--min-cap2", type=int, default=40000)
ap.add_argument("--protect", type=int, default=0,
                help="a running first attempt is never cut by the elastic budget before this many tokens (endgame and "
                     "the hard cap still cut); 0 = v11 behaviour. Cutting running attempts made their remaining work "
                     "look free, and the forced-conclusion prefill fed back into the rate (run 37 cap collapse)")
ap.add_argument("--rate-mode", default="window", choices=["window", "long"],
                help="long: generated tokens/s since T0 + 1200 s (at least 600 s of history) instead of the last 600 s")
ap.add_argument("--no-loop-detect", action="store_true")
ap.add_argument("--kv-admit", type=float, default=0.85,
                help="start new attempts only while server KV usage is below this (avoids recompute preemptions)")
ap.add_argument("--min-running", type=int, default=8, help="always keep at least this many attempts running")
ap.add_argument("--endgame-s", type=float, default=900,
                help="this long before the deadline every running attempt is cut and force-concluded (none are lost)")
ap.add_argument("--budget-margin-s", type=float, default=600,
                help="elastic caps size the remaining work to finish this long before the deadline (default 600)")
ap.add_argument("--max-model-len", type=int, default=262144)
ap.add_argument("--force-tokens", type=int, default=3000)
ap.add_argument("--force-ctx-chars", type=int, default=24000,
                help="forced conclusions see the prompt + only the last N chars of the reasoning: bounded re-prefill "
                     "(vLLM's hybrid prefix cache loses SWA checkpoints under 40-way decode churn on a small KV pool)")
ap.add_argument("--attempts", type=int, default=2)
ap.add_argument("--alt", default="off", choices=["off", "full", "tail", "session"],
                help="after a finished attempt 1, one greedy continuation asks for the most plausible alternative grid "
                     "(attempt_2 filler while no real second attempt exists). full: whole reasoning (prefix-cache hit "
                     "only if the SWA checkpoints survive); tail: prompt + last --force-ctx-chars of it (bounded prefill); "
                     "session (needs --session-chunk): pushed into the attempt's own session right after its natural or "
                     "budget-forced conclusion, so only ALT_NOTE is prefilled; not "
                     "after the endgame; used as attempt_2 only when attempt_2 would otherwise repeat attempt_1")
ap.add_argument("--alt-tokens", type=int, default=3000)
ap.add_argument("--alt-timeout-s", type=float, default=240,
                help="--alt session: give up on the alternative (abort the session, keep the answer) after this long")
ap.add_argument("--session-chunk", type=int, default=0,
                help="> 0: run each attempt as a server-side session (arcstats ARC_SESSION_EP) in chunks of this many "
                     "tokens; a budget/endgame cut then forces the conclusion inside the same sequence (whole reasoning in "
                     "context, ~25 tokens to prefill) instead of a new request that re-prefills prompt + tail uncached")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--grid-fmt", default="json", choices=["json", "compact", "compact_noinstr"],
                help="prompt grid format (gridfmt.py, ship next to solve.py when not json). compact: separator-free "
                     "digit rows with a (HxW) header + an instruction to write grids compactly and refer to known rows by "
                     "name; compact_noinstr: compact prompt grids only. Non-json also parses compact answers/trace grids "
                     "(JSON answers still accepted; forced conclusions keep the JSON prefill). the design notes")
a = ap.parse_args()

T0 = time.time()
DEADLINE = T0 + a.deadline_s
KV = [0.0]                                 # latest server KV-cache usage (0..1)
ENDGAME = DEADLINE - a.endgame_s
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.tokenizer)
TEMPLATE = Path(a.template).read_text()
chal = json.load(open(a.challenges))
if a.tasks:
    keep = set(open(a.tasks).read().split())
    chal = {k: v for k, v in chal.items() if k in keep}
sols = json.load(open(a.solutions)) if a.solutions else {}
logf = open(a.log, "a") if a.log else None


def log(*x):
    s = f"[solve {time.time() - T0:7.0f}s] " + " ".join(str(v) for v in x)
    print(s, flush=True)
    if logf:
        logf.write(s + "\n"); logf.flush()


if a.grid_fmt != "json":
    import gridfmt  # noqa: E402


def make_prompt(task, ti):
    if a.grid_fmt != "json":
        body = gridfmt.prompt_body(TEMPLATE, task, ti, a.grid_fmt)
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    ex = ""
    for i, p in enumerate(task["train"]):
        ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(p["input"]) + "\n\n"
        ex += "OUTPUT: \n\n" + json.dumps(p["output"]) + "\n\n"
    body = TEMPLATE.format(training_examples=ex, test_input=json.dumps(task["test"][ti]["input"]))
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def parse(ans):
    return parse_json(ans) if a.grid_fmt == "json" else gridfmt.parse_answer(ans, parse_json)


def parse_json(ans):
    m = re.search(r"\\boxed\{(.*?)\}", ans, re.DOTALL)
    for cand in ([m.group(1)] if m else []) + [ans]:
        end = cand.rfind("]")
        depth, start = 0, -1
        for i in range(end, -1, -1):
            depth += cand[i] == "]"
            depth -= cand[i] == "["
            if depth == 0:
                start = i; break
        if start >= 0:
            try:
                g = json.loads(cand[start:end + 1])
                if isinstance(g, list) and g and all(isinstance(r, list) and r and all(isinstance(v, int) for v in r) for r in g):
                    return g
            except Exception:
                pass
    return None


FORCE_PREFIX = "\n\nTime is up; I will commit to my best hypothesis.\n</think>\n\nFinal answer:\n```json\n[["


def parse_partial(txt):
    """Grid from a (possibly truncated) JSON list of rows: keep complete rows of one common length."""
    g = parse_json(txt)
    if g is not None:
        return g
    rows = re.findall(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]", txt)
    rows = [[int(v) for v in r.split(",")] for r in rows]
    if not rows:
        return None
    w = len(rows[0])
    out = []
    for r in rows:
        if len(r) != w:
            break
        out.append(r)
    return out if out and len(out) <= 30 and w <= 30 else None


def expected_shape(task, ti):
    """Output shape implied by the train pairs (all same-as-input, or all one fixed shape); None if unknown."""
    shp = lambda g: (len(g), len(g[0]) if g else 0)
    tr = task["train"]
    if all(shp(p["input"]) == shp(p["output"]) for p in tr):
        return shp(task["test"][ti]["input"])
    outs = {shp(p["output"]) for p in tr}
    return outs.pop() if len(outs) == 1 else None


GRID_RE = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")


def trace_grid(text, shape, test_in):
    if a.grid_fmt != "json":
        g = gridfmt.trace_grid_compact(text, shape, test_in)
        if g is not None:
            return g
    return trace_grid_json(text, shape, test_in)


def trace_grid_json(text, shape, test_in):
    """Last well-formed grid of the expected shape written in the reasoning (answers often appear mid-trace even when
    the attempt runs out of tokens); never the test input itself."""
    for m in reversed(list(GRID_RE.finditer(text))):
        try:
            g = json.loads(m.group(0))
        except Exception:
            continue
        if not (isinstance(g, list) and g and all(isinstance(r, list) and r and all(isinstance(v, int) for v in r)
                                                  for r in g)):
            continue
        if len({len(r) for r in g}) != 1 or g == test_in:
            continue
        if shape is None or (len(g), len(g[0])) == tuple(shape):
            return g
    return None


# ---------------------------------------------------------------- state
items = [(t, ti) for t in chal for ti in range(len(chal[t]["test"]))]
answers = {k: [] for k in items}          # parsed grids in attempt order
backup = {k: [] for k in items}           # grids recovered from capped reasoning traces (used only to fill gaps)
shapes = {k: expected_shape(chal[k[0]], k[1]) for k in items}
prompts = {k: make_prompt(chal[k[0]], k[1]) for k in items}
pids = {k: tok(prompts[k], add_special_tokens=False).input_ids for k in items}
plen = {k: len(v) for k, v in pids.items()}
queue = [(k, 1) for k in items]            # (item, attempt number)
# confidence-gated termination, default off: env ARC_CONF_KILL="x:thr[,x:thr]" (+ ARC_CONF_FEATURE,
# ARC_CONF_K, ARC_CONF_GROUP, see conf_gate.py). A first attempt whose confidence at checkpoint x is below thr is cut
# while thinking; its forced conclusion (ARC_CONF_FORCE=1, default) only fills gaps (backup), and a fresh first attempt
# for the same item goes to the front of the queue. At most ARC_CONF_RETRIES (1) kills per item; retries are not gated.
# Requests that carry `logprobs` take vLLM's stock sampling path (no arcstats shard_sample for that step).
CONF = ConfGate.from_env() if ConfGate is not None else None
CONF_RETRIES = int(os.environ.get("ARC_CONF_RETRIES", "1"))
CONF_FORCE = os.environ.get("ARC_CONF_FORCE", "1") == "1"
conf_kills = {k: 0 for k in items}
RATE0 = [None]                             # (time, tokens) at T0 + 1200 s for --rate-mode long
gen_hist = []                              # (time, tokens generated so far) for the rate estimate
done_tokens = 0
inflight = {}                              # task -> expected remaining tokens


alt = {}                                   # item -> alternative grid (attempt_2 filler, see --alt)
ALT_SESSION = a.alt == "session" and a.session_chunk > 0
alt_res = {}                               # (item, attempt) -> (alt text | None, alt tokens, seconds, status) (--alt session)
alt_stats = []                             # (att1 correct?, status, alt correct?, alt used, tokens, seconds) (--alt session)
ALT_NOTE = ("\n\nIf that answer turns out to be wrong, my most plausible alternative output (a different grid) is:\n"
            "```json\n[[")


def write_submission():
    sub = {}
    for t in chal:
        outs = []
        for ti in range(len(chal[t]["test"])):
            g = answers[(t, ti)] + [b for b in backup[(t, ti)] if b not in answers[(t, ti)]]
            if len(g) == 1 and (t, ti) in alt and alt[(t, ti)] != g[0]:
                g = g + [alt[(t, ti)]]
            fb = chal[t]["test"][ti]["input"]
            a1 = g[0] if g else fb
            a2 = g[1] if len(g) > 1 else (g[0] if g else fb)
            outs.append({"attempt_1": a1, "attempt_2": a2})
        sub[t] = outs
    tmp = a.out + ".tmp"
    json.dump(sub, open(tmp, "w"))
    Path(tmp).replace(a.out)


# natural completion-token quantiles of lb144 attempts on holdout40 (runs/hold_lb144_{fixed,bias}; 240001 = hit the
# 240k log cap); used to project E[min(ct, L)] for the elastic caps (dots3/budget_sim.py)
NAT = [17e3, 25e3, 31e3, 37e3, 43e3, 47e3, 55e3, 66e3, 77e3, 87e3, 96e3, 101e3, 124e3, 140e3, 157e3, 205e3, 239e3,
       240e3, 240e3, 240e3]
prog = {}                                  # (item, attempt) -> completion tokens so far (streamed usage)
att1_ct = {}                               # item -> completion tokens of its first attempt
LIVE = {1: a.cap1, 2: a.cap2}              # live caps; running attempts beyond them are cut + forced to conclude
ADMIT2 = [True]


def exp_rem(L, p=0.0):
    """expected tokens still to generate under cap L for an attempt at progress p"""
    if p >= L:
        return 0.0
    m = [min(n, L) for n in NAT if n > p]
    return (sum(m) / len(m) - p) if m else L - p


def best_cap(budget, n_new, progs, hi, lo, fixed=0.0):
    for L in range(int(hi), int(lo) - 1, -8000):
        # protected running first attempts are not free below their progress: they run on to max(L, protect)
        run = sum(exp_rem(max(L, a.protect) if p < a.protect else L, p) for p in progs)
        if fixed + n_new * (exp_rem(L) + a.force_tokens / 5) + run <= budget:
            return L
    return lo - 1


def update_live_caps():
    """elastic caps: first attempts get the largest cap whose expected total work fits the remaining budget (cutting
    running ones only while first attempts are still queued); second attempts only spend the surplus"""
    r = rate()
    if r is None:
        return
    B = r * max(0.0, DEADLINE - time.time() - a.budget_margin_s)
    pend1 = sum(1 for _, n in queue if n == 1)
    p1 = [v for (k, n), v in prog.items() if n == 1]
    p2 = [v for (k, n), v in prog.items() if n == 2]
    L1 = max(a.min_cap, best_cap(B, pend1, p1, a.cap1, a.min_cap)) if pend1 else a.cap1
    need1 = pend1 * exp_rem(L1) + sum(exp_rem(max(L1, a.protect) if p < a.protect else L1, p) for p in p1)
    L2 = best_cap(B - need1, 0, p2, a.cap2, a.min_cap2)
    LIVE[1], LIVE[2] = L1, max(L2, a.min_cap2)
    ADMIT2[0] = L2 >= a.cap2 and B - need1 - sum(exp_rem(a.cap2, p) for p in p2) >= exp_rem(a.cap2)


def rate():
    """generated tokens/s over the last ~10 minutes (server metric), None until known."""
    if len(gen_hist) < 2:
        return None
    t1, g1 = gen_hist[-1]
    t0, g0 = next(((t, g) for t, g in gen_hist if t >= t1 - 600), gen_hist[0])
    if a.rate_mode == "long" and RATE0[0] is not None and t1 - RATE0[0][0] >= 600:
        t0, g0 = RATE0[0]
    return (g1 - g0) / max(1.0, t1 - t0) if t1 > t0 else None


def live_cap(attempt, n):
    """elastic cap for a running attempt at n tokens (protected first attempts run on to --protect)"""
    if attempt == 1 and n < a.protect:
        return max(LIVE[1], a.protect)
    return LIVE[attempt]


def cap_for(attempt):
    """cap for a new attempt (0 = do not start a second attempt now)"""
    if attempt == 1:
        return LIVE[1]
    return LIVE[2] if ADMIT2[0] else 0


async def metrics_loop(c):
    while time.time() < DEADLINE:
        try:
            m = (await c.get(f"http://127.0.0.1:{a.port}/metrics", timeout=10)).text
            g = sum(float(x) for x in re.findall(r"^vllm:generation_tokens_total(?:\{[^}]*\})? ([0-9.e+]+)", m, re.M))
            gen_hist.append((time.time(), g))
            if RATE0[0] is None and time.time() - T0 >= 1200:
                RATE0[0] = gen_hist[-1]
            kv = re.findall(r"^vllm:kv_cache_usage_perc(?:\{[^}]*\})? ([0-9.e+-]+)", m, re.M)
            if kv:
                KV[0] = max(float(x) for x in kv)
            del gen_hist[:-400]
            update_live_caps()
        except Exception:
            pass
        await asyncio.sleep(20)


FORCE_IDS = tok(FORCE_PREFIX, add_special_tokens=False).input_ids
FENCE_IDS = sorted(i for t, i in tok.get_vocab().items() if "```" in tok.convert_tokens_to_string([t]))
ALT_IDS = tok(ALT_NOTE, add_special_tokens=False).input_ids
SPECIAL_IDS = set(tok.all_special_ids)


async def _timed_lines(resp, alt_t0):
    """resp.aiter_lines(), but once alt_t0[0] is set (an --alt session push is pending) a line must arrive before
    alt_t0[0] + --alt-timeout-s (and DEADLINE - 90), else asyncio.TimeoutError"""
    it = resp.aiter_lines().__aiter__()
    while True:
        if alt_t0[0] is None:
            nxt = await it.__anext__()
        else:
            lim = min(alt_t0[0] + a.alt_timeout_s, DEADLINE - 90) - time.time()
            nxt = await asyncio.wait_for(it.__anext__(), timeout=max(0.01, lim))
        yield nxt


async def run_session(c, k, attempt, body, max_tok, det):
    """One attempt as a server session (see --session-chunk). Returns (text, fin, ct, cut, forced_text | None)."""
    rid = f"{k[0]}_{k[1]}_{attempt}_{time.time():.3f}"
    U = f"http://127.0.0.1:{a.port}/arc/session"
    parts, ids, fin, cut, closed, forced, ftext = [], [], None, "", False, False, []
    n_chunk = min(a.session_chunk, max_tok, max(64, live_cap(attempt, 0)))
    # --alt session: the server keeps the session open after a stop (EOS / fence) so one more chunk can be pushed
    keep = ALT_SESSION and attempt == 1
    kx = {"keep": True} if keep else {}
    alt_t0, alt_parts, alt_n, alt_status = [None], [], 0, "none"
    body0 = dict({"rid": rid, "prompt_ids": pids[k], "max_tokens": n_chunk, "temperature": a.temp, "top_p": a.top_p},
                 **kx)
    async with c.stream("POST", U, json=body0) as resp:
      try:
        async for line in (_timed_lines(resp, alt_t0) if keep else resp.aiter_lines()):
            if not line:
                continue
            ev = json.loads(line)
            if alt_t0[0] is not None:                       # --alt session: the alternative's chunk
                if "ids" in ev:
                    alt_n += len(ev["ids"])
                    alt_parts.append(ev["text"])
                elif "chunk_end" in ev:
                    alt_status = "ok"
                    await c.post(U + "/close", json={"rid": rid})
                elif "end" in ev:
                    break
                continue
            if keep and "chunk_end" in ev and (forced or ev["chunk_end"] != "length"):
                # a conclusion just ended (natural EOS or the forced answer's fence): one alternative, else close
                fin = fin or ev["chunk_end"]
                body_t = "".join(ftext) if forced else "".join(parts)
                g = parse_partial("[[" + body_t) if forced else \
                    (parse(body_t.rpartition("</think>")[2]) if "</think>" in body_t else None)
                now = time.time()
                if g is not None and ids and now < ENDGAME and now < DEADLINE - 120:
                    last = [] if ids[-1] in SPECIAL_IDS else [ids[-1]]   # re-add the dropped fence, not an EOS
                    alt_t0[0] = now
                    alt_status = "pushed"
                    r = await c.post(U + "/push", json={"rid": rid, "prompt_ids": last + ALT_IDS,
                                                        "max_tokens": max(16, min(a.alt_tokens, a.max_model_len -
                                                                                  plen[k] - len(ids) - 64)),
                                                        "temperature": 0.0, "stop_token_ids": FENCE_IDS, "keep": True})
                    if r.status_code != 200:
                        alt_status = f"http{r.status_code}"
                        break                                     # leaving the stream aborts the session
                else:
                    alt_status = "skip"
                    await c.post(U + "/close", json={"rid": rid})
                continue
            if "ids" in ev:
                ids += ev["ids"]
                (ftext if forced else parts).append(ev["text"])
                if not forced:
                    prog[(k, attempt)] = len(ids)
                    closed = closed or "</think>" in "".join(parts[-2:])
                    if det is not None and not cut and det.feed(ev["text"]):
                        cut = "loop"
            elif "chunk_end" in ev:
                if forced or ev["chunk_end"] != "length":
                    fin = fin or ev["chunk_end"]
                    await c.post(U + "/close", json={"rid": rid})
                    continue
                if not cut and not closed and len(ids) >= live_cap(attempt, len(ids)):
                    cut = "budget"
                if not cut and not closed and time.time() >= ENDGAME:
                    cut = "endgame"
                if cut == "loop" or (not cut and len(ids) >= max_tok):
                    fin = "length"                          # loop: conclude from the excised text (old path)
                    await c.post(U + "/close", json={"rid": rid})
                elif cut:
                    if time.time() < DEADLINE - 120:        # force inside the sequence: only FORCE_PREFIX to prefill
                        forced, fin = True, "length"
                        await c.post(U + "/push", json=dict({"rid": rid, "prompt_ids": [ids[-1]] + FORCE_IDS,
                                                             "max_tokens": a.force_tokens, "temperature": 0.0,
                                                             "stop_token_ids": FENCE_IDS},
                                                            **({"keep": True} if keep and cut == "budget" else {})))
                    else:
                        fin = "length"
                        await c.post(U + "/close", json={"rid": rid})
                else:
                    await c.post(U + "/push", json=dict({"rid": rid, "prompt_ids": [ids[-1]],
                                                         "max_tokens": min(a.session_chunk, max_tok - len(ids),
                                                                           max(64, live_cap(attempt, len(ids)) -
                                                                               len(ids)))}, **kx))
            elif "end" in ev:
                break
      except (asyncio.TimeoutError, httpx.HTTPError, ValueError) as e:
        if alt_t0[0] is None:
            raise                                                   # not the alternative's fault: old behaviour
        alt_status = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
    if keep:
        alt_res[(k, attempt)] = ("".join(alt_parts) if alt_status == "ok" else None, alt_n,
                                 time.time() - alt_t0[0] if alt_t0[0] else 0.0, alt_status)
    text = "".join(parts)
    return text, fin, len(ids), cut, ("".join(ftext) if forced else None)


async def one(c, k, attempt, cap):
    t, ti = k
    # request the upper-bound cap; the live (elastic) cap is enforced client-side while streaming
    max_tok = max(256, min(a.cap1 if attempt == 1 else a.cap2, a.max_model_len - plen[k] - a.force_tokens - 64))
    inflight[(k, attempt)] = max_tok
    body = {"model": "m", "prompt": prompts[k], "max_tokens": max_tok, "temperature": a.temp, "top_p": a.top_p,
            "skip_special_tokens": False, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    gate = None
    if CONF is not None and attempt == 1 and conf_kills[k] < CONF_RETRIES and a.session_chunk <= 0:
        gate = CONF.clone()
        body["logprobs"] = gate.k
    t_start = time.time()
    try:
        det = None if a.no_loop_detect else LoopDetector()
        parts, fin, ct, cut, closed = [], None, 0, "", False
        prog[(k, attempt)] = 0
        ftext = None
        if a.session_chunk > 0:
            text, fin, ct, cut, ftext = await run_session(c, k, attempt, body, max_tok, det)
            parts = [text]
        else:
          # leaving the `async with` early closes the connection -> vLLM aborts the request
          async with c.stream("POST", f"http://127.0.0.1:{a.port}/v1/completions", json=body) as resp:
              async for line in resp.aiter_lines():
                  if not line.startswith("data: ") or line == "data: [DONE]":
                      continue
                  ev = json.loads(line[6:])
                  if ev.get("usage"):
                      ct = prog[(k, attempt)] = ev["usage"]["completion_tokens"]
                  for ch in ev.get("choices") or []:
                      parts.append(ch.get("text") or "")
                      closed = closed or "</think>" in "".join(parts[-2:])
                      fin = ch.get("finish_reason") or fin
                      if det is not None and det.feed(ch.get("text") or ""):
                          cut = "loop"
                      if gate is not None and not cut and not closed and gate.feed_openai(ch.get("logprobs")):
                          cut = "conf"
                  if not cut and fin is None and not closed and ct >= live_cap(attempt, ct):
                      cut = "budget"
                  if not cut and fin is None and not closed and time.time() >= ENDGAME:
                      cut = "endgame"
                  if cut:
                      break
        text = "".join(parts)
        if cut:
            fin = "length"                                         # treated like a capped attempt below
            if cut == "loop" and det.loop_start is not None:
                text = text[:det.loop_start]                       # do not condition the conclusion on the loop
        if cut == "conf":
            conf_kills[k] += 1
            queue.insert(0, (k, 1))                                # fresh first attempt gets the freed budget
            log("conf_kill", t, ti, f"ct={ct}", f"value={gate.last_value:.4f}")
        elif attempt == 1:
            att1_ct[k] = ct
        forced = False
        grid = None
        if ftext is not None:                                       # session: forced inside the sequence
            grid = parse_partial("[[" + ftext)
            forced = True
            tg = trace_grid(text, shapes[k], chal[t]["test"][ti]["input"])
            if tg is not None and tg not in backup[k]:
                backup[k].append(tg)
        elif cut == "conf" and not CONF_FORCE:
            pass
        elif "</think>" not in text and fin == "length" and time.time() < DEADLINE - 120:
            # forced conclusion: close the thinking and prefill the answer's opening so the grid comes next
            tail = text if len(text) <= a.force_ctx_chars else \
                "[... earlier reasoning omitted ...]\n" + text[-a.force_ctx_chars:]
            body2 = dict(body, prompt=prompts[k] + tail + FORCE_PREFIX, max_tokens=a.force_tokens, temperature=0.0,
                         stop=["```"], stream=False)
            body2.pop("stream_options", None)
            body2.pop("logprobs", None)
            r2 = (await c.post(f"http://127.0.0.1:{a.port}/v1/completions", json=body2,
                               timeout=max(10.0, DEADLINE - 90 - time.time()))).json()
            grid = parse_partial("[[" + r2["choices"][0]["text"])
            ct += r2["usage"]["completion_tokens"]
            forced = True
            tg = trace_grid(text, shapes[k], chal[t]["test"][ti]["input"])
            if tg is not None and tg not in backup[k]:
                backup[k].append(tg)
        elif "</think>" in text:
            grid = parse(text.rpartition("</think>")[2])
        if cut == "conf":
            if grid is not None and grid not in backup[k]:
                backup[k].append(grid)                             # a doomed trace's answer only fills gaps
        elif grid is not None and grid not in answers[k]:
            answers[k].append(grid)
        elif grid is not None and attempt > 1:
            pass   # duplicate answer: keep a single copy (attempt_2 falls back to it)
        alt_note = ""
        if a.alt in ("full", "tail") and attempt == 1 and not forced and grid is not None and time.time() < ENDGAME:
            try:
                body_text = text.rstrip()
                if a.alt == "tail" and len(body_text) > a.force_ctx_chars:
                    body_text = "[... earlier reasoning omitted ...]\n" + body_text[-a.force_ctx_chars:]
                body3 = dict(body, prompt=prompts[k] + body_text + ALT_NOTE, max_tokens=a.alt_tokens, temperature=0.0,
                             stop=["```"], stream=False)
                body3.pop("stream_options", None)
                body3.pop("logprobs", None)
                t3 = time.time()
                r3 = (await c.post(f"http://127.0.0.1:{a.port}/v1/completions", json=body3,
                                   timeout=max(10.0, DEADLINE - 90 - time.time()))).json()
                g3 = parse_partial("[[" + r3["choices"][0]["text"])
                ct += r3["usage"]["completion_tokens"]
                alt_note = f"alt{time.time() - t3:.0f}s"
                if g3 is not None and g3 != grid:
                    alt[k] = g3
                    alt_note += "+"
            except Exception as e:
                log("alt error", t, ti, repr(e)[:120])
        gt = sols.get(t, [None] * 9)[ti] if sols else None
        if (k, attempt) in alt_res:                                 # --alt session result of this attempt
            atext, an, asec, ast = alt_res.pop((k, attempt))
            g3 = parse_partial("[[" + atext) if atext is not None else None
            if ast == "ok" and g3 is None:
                ast = "nogrid"
            elif g3 is not None and g3 == grid:
                ast = "same"
            elif g3 is not None and grid is not None and cut != "conf":
                alt[k] = g3
                ast = "new"
            alt_note = f"salt_{ast}_{an}t_{asec:.0f}s"
            a1c = (grid == gt) if gt is not None and grid is not None else None
            a2c = (g3 == gt) if gt is not None and ast == "new" else None
            alt_stats.append((k, a1c, ast, a2c, an, asec, forced))
            log("alt", t, ti, f"att{attempt}", "forced" if forced else "natural", f"status={ast}", f"tok={an}",
                f"{asec:.1f}s", "att1=" + ({True: "correct", False: "wrong"}.get(a1c, "nogt")),
                "alt=" + ({True: "correct", False: "wrong"}.get(a2c, "none")))
        log("done", t, ti, f"att{attempt}", f"cap={max_tok}", f"ct={ct}", f"{time.time() - t_start:.0f}s", cut,
            "forced" if forced else "", "parsed" if grid is not None else "NOGRID",
            ("correct" if grid == gt else "wrong") if gt is not None and grid is not None else "", alt_note +
            (("_correct" if alt.get(k) == gt else "_wrong") if k in alt and gt is not None else ""))
    except Exception as e:
        log("error", t, ti, attempt, repr(e)[:200])
    finally:
        inflight.pop((k, attempt), None)
        prog.pop((k, attempt), None)
        write_submission()


async def main():
    write_submission()
    # no keep-alive reuse: session push/close POSTs leave idle pooled connections that uvicorn closes after 5 s;
    # reusing one as it closes raises ReadError and kills a multi-hour attempt (run 72: 4 lost attempt-1s)
    limits = httpx.Limits(max_connections=a.concurrency + 8, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.deadline_s), limits=limits) as c:
        mt = asyncio.create_task(metrics_loop(c))
        running = set()
        while time.time() < DEADLINE - 60:
            while queue and len(running) < a.concurrency and time.time() < ENDGAME and \
                    (KV[0] < a.kv_admit or len(running) < a.min_running):
                if queue[0][1] > 1:
                    # second attempts: items whose first attempt finished, shortest first attempt first
                    ready = [q for q in queue if q[0] in att1_ct]
                    if not ready:
                        break
                    q = min(ready, key=lambda q: att1_ct[q[0]])
                    cap = cap_for(q[1])
                    if cap < a.min_cap2:
                        break                                 # no surplus now; re-checked every 15 s
                    queue.remove(q)
                    k, n = q
                else:
                    k, n = queue.pop(0)
                    cap = cap_for(n)
                running.add(asyncio.create_task(one(c, k, n, cap)))
                if n < a.attempts and not any(q[0] == k for q in queue):
                    queue.append((k, n + 1))              # second attempts after all first attempts
            if not running:
                break
            done, running = await asyncio.wait(running, timeout=15, return_when=asyncio.FIRST_COMPLETED)
        for tsk in running:
            tsk.cancel()
        mt.cancel()
    write_submission()
    n_ans = sum(1 for k in items if answers[k])
    log(f"final: {n_ans}/{len(items)} test inputs answered")
    if sols:
        sc = 0.0
        for t in chal:
            gts = sols.get(t, [])
            hit = [any(g == gts[ti] for g in answers[(t, ti)][:2]) or
                   (len(answers[(t, ti)]) == 1 and alt.get((t, ti)) == gts[ti])
                   for ti in range(len(chal[t]["test"])) if ti < len(gts)]
            sc += sum(hit) / max(1, len(hit))
        log(f"local score (pass@2): {sc:.2f}/{len(chal)} = {100 * sc / len(chal):.1f}%")
    if ALT_SESSION:
        # q = P(alternative correct | attempt-1 answer wrong), over conclusions whose alternative is a new grid; 'used' =
        # the alternative is attempt_2 in submission.json (write_submission's rule); gain = inputs it turns correct
        from collections import Counter
        st = Counter(x[2] for x in alt_stats)
        new = [x for x in alt_stats if x[2] == "new"]
        w = [x for x in new if x[1] is False]
        wf = [x for x in w if x[6]]
        used = gain = 0
        for k, a1c, ast, a2c, *_ in new:
            g = answers[k] + [b for b in backup[k] if b not in answers[k]]
            if len(g) == 1 and alt.get(k) is not None and alt[k] != g[0]:
                used += 1
                gain += bool(a2c) and not any(x == sols.get(k[0], [None] * 9)[k[1]] for x in g) if sols else 0
        ntok = sum(x[4] for x in alt_stats)
        log(f"alt-q: conclusions {len(alt_stats)} status {dict(st)} | att1 wrong with a new alt: {len(w)} "
            f"(forced {len(wf)}) | alt correct: {sum(1 for x in w if x[3])} -> q = "
            f"{sum(1 for x in w if x[3]) / max(1, len(w)):.3f} (forced {sum(1 for x in wf if x[3]) / max(1, len(wf)):.3f}"
            f", natural {sum(1 for x in w if x[3] and not x[6]) / max(1, len(w) - len(wf)):.3f}) | used as attempt_2: "
            f"{used}, inputs gained: {gain} | alt tokens {ntok} ({ntok / max(1, len(alt_stats)):.0f}/conclusion), "
            f"mean {sum(x[5] for x in alt_stats) / max(1, len(alt_stats)):.1f}s")

asyncio.run(main())
