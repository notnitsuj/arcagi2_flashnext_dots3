import argparse, asyncio, contextvars, json, os, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import repl_sandbox as sandbox
try:
    import arckit as ARCKIT
except Exception:
    ARCKIT = None
import repl_vote
BACKENDS = ('qwen', 'dots3', 'fnext')
SAMPLING = {'qwen': {'temp': 1.0, 'top_p': 0.95, 'top_k': 20}, 'dots3': {'temp': 0.7, 'top_p': 1.0, 'top_k': 0}, 'fnext': {'temp': 1.0, 'top_p': 0.95, 'top_k': 20}}

def parse_args(argv):
    ap = argparse.ArgumentParser(prog='solve.py --mode repl')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--challenges', required=True)
    ap.add_argument('--solutions', default=None, help='optional, for local scoring (never shown to the model)')
    ap.add_argument('--tasks', default=None, help='optional file of task ids to restrict to')
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', default=None)
    ap.add_argument('--traces-out', default=None, help="optional jsonl.gz: each session's transcript (debugging)")
    ap.add_argument('--samples-out', default=None, help='jsonl, one row per session (default: repl_samples.jsonl next to --log)')
    ap.add_argument('--template', default=None, help='accepted for launcher compatibility (direct-mode prompt; unused)')
    ap.add_argument('--tokenizer', required=True, help='model directory (tokenizer + chat template)')
    ap.add_argument('--deadline-s', type=float, required=True)
    ap.add_argument('--backend', default='qwen', choices=BACKENDS, help='qwen: Qwen3.8 native tool calls (<tool_call><function=python>); dots3: <dots_function_call>; fnext: Qwen3.8-Flash-Next, the qwen harness over server sessions (vLLM 0.32 streaming input)')
    ap.add_argument('--concurrency', type=int, default=40, help='sessions in flight')
    ap.add_argument('--min-running', type=int, default=8)
    ap.add_argument('--kv-admit', type=float, default=0.9, help='start sessions only while server KV use is below this')
    ap.add_argument('--import-fix', default='off', choices=('off', 'carry', 'alias'), help="harness (default off): carry = with --fn-fixes, an import carried from an earlier cell is wrapped in try/except ImportError, so an import that failed there (e.g. `import ak`) no longer kills every later cell that uses the name; alias = carry + `ak` importable in the sandbox and AUX runners (sys.modules alias of arckit)")
    ap.add_argument('--tb-fix', default='off', choices=('off', 'on'), help="harness (default off): with --fn-fixes, a cell that ran with carried earlier imports/defs prepended gets its traceback rewritten to the model's own numbering: <tool> line numbers minus the prefix length, each frame followed by its source line, and a frame inside the carried prefix named as such (carried import / definition from an earlier cell). Cells without a prefix are unchanged")
    ap.add_argument('--answer-order', default='off', choices=('off', 'shape'), help="session answers (default off): shape = a session's parsed answer grids (the model's final / forced answer, not program outputs) are re-assigned to test inputs by output shape when every arckit size rule that fits the training pairs predicts the same shape for each test input, those shapes are pairwise distinct, and every answer grid matches exactly one of them (the model sometimes writes test input 2's grid first, or only that one); otherwise the answers keep their positions")
    ap.add_argument('--kv-agg', choices=('max', 'mean'), default='max', help='combine the per-engine KV use for admission')
    ap.add_argument('--ctx-frac', type=float, default=0.9, help="context-aware admission: start a session only while the projected context of the running sessions (each max(its live context, --ctx-typ), + one more) fits this share of the server's KV token capacity (vllm:cache_config_info kv_cache_size_tokens); 0 = off")
    ap.add_argument('--ctx-typ', type=int, default=32768, help='projected context of a session (raised to 0.6 x the mean final context of finished sessions)')
    ap.add_argument('--sample-cap', type=int, default=65536, help='generated tokens per session (all turns)')
    ap.add_argument('--min-cap', type=int, default=16384)
    ap.add_argument('--turn-tokens', type=int, default=32768, help='per-turn generation budget (soft)')
    ap.add_argument('--task-cap-per-input', type=int, default=270000)
    ap.add_argument('--kmax', type=int, default=6, help='sessions per task')
    ap.add_argument('--agree', type=int, default=2)
    ap.add_argument('--policy', default='tiers', choices=['tiers', 'keep'], help='tiers: the shared harness tiers (unverified tasks, then confirmations while top votes < --agree). keep: every task gets a first session, then more sessions while budget remains, fewest-sessions-first (no settle rule unless --agree is reached by every input)')
    ap.add_argument('--max-inflight-per-task', type=int, default=2)
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--force-tokens', type=int, default=2048, help='forced-answer tokens per test input; 0 = off')
    ap.add_argument('--force-skip-verified', action='store_true', help='no forced answer for a session whose task already has a verified vote on every test input (decided when the session concludes)')
    ap.add_argument('--max-tool-calls', type=int, default=24)
    ap.add_argument('--exec-timeout', type=float, default=10.0)
    ap.add_argument('--exec-workers', type=int, default=0, help='concurrent sandbox processes (0 = from the CPU count)')
    ap.add_argument('--no-stop-on-verify', dest='stop_on_verify', action='store_false')
    ap.add_argument('--no-exec-on-cut', dest='exec_on_cut', action='store_false')
    ap.add_argument('--no-nudge', dest='nudge', action='store_false', help='dots3: no per-turn nudge into a call')
    ap.add_argument('--keep-code', action='store_true')
    ap.add_argument('--fn-fixes', default='auto', choices=['auto', 'on', 'off'], help="harness fixes (fnext default on): grid parser accepts quoted digits / row strings / digit rows; forced-answer tokens sized from the largest grid; tool stdout cap 6000 chars; pre-imported Counter/defaultdict/deque/collections/itertools/math + earlier cells' imports and defs carried into a cell that uses them; an unclosed <tool_call> inside thinking is run, not taken as the final answer; timeouts counted")
    ap.add_argument('--consistency-flags', default='off', choices=['on', 'off'], help="extra rules (default off): demote also on augmentation inconsistency (D8 / colour maps equivariant on train but not on a test input) or test-only line coverage of the verified program (one extra sandbox pass per verified session), and serve settled tasks (every input's top grid >= 2 verified votes, margin >= 1) last")
    ap.add_argument('--settle-last', default='off', choices=['on', 'off'], help="allocation only (--policy keep): a task whose every test input's top grid has >= 2 verified votes and leads the next verified grid by >= 1 is served after all unsettled tasks; no demotion (--consistency-flags on implies it)")
    ap.add_argument('--att2-fill', default='off', choices=['on', 'off'], help='submission only (default off): a test input whose selection has a single distinct candidate gets, as attempt_2, the most frequent OTHER grid any program of any session output for it (verified or not; ties: best train-pass fraction, then grid text) instead of a copy of attempt_1. attempt_1 never changes (an input with no candidate keeps its fallback attempt_1)')
    ap.add_argument('--carry', default='off', choices=['on', 'off'], help="session prompt (default off): a new session of a task none of whose finished sessions has a verified program gets a note with one earlier session's best non-verified transform (>= 1 example right) and its per-example result (carry_note); the earlier sessions' programs rotate")
    ap.add_argument('--tie-break', default='off', choices=['off', 'nif', 'flags'], help='submission only (default off): among verified grids with EQUAL vote counts, nif ranks first the grid whose supporting verified program has the most `if` statements (the program that handles more cases; off = earliest session); flags first prefers a grid with a supporter whose program passes the augmentation / test-only-coverage check (one extra sandbox pass per verified session, never a demotion), then nif. Vote counts and the fallback order never change')
    ap.add_argument('--arckit', default='off', choices=['off', 'lib', 'block', 'review'], help='helper library in the sandbox (arckit.py next to this file): lib = `ak` pre-imported + content flags shown in the feedback; block = as lib, and a program whose test outputs are flagged does not count as verified (the session continues); review = block + one review of test-only lines. Implies --prompt ak unless another --prompt is given')
    ap.add_argument('--demote-flagged', default='auto', choices=['auto', 'on', 'off'], help="a finished session whose last verified program's test outputs break a safe training invariant (arckit.flag, when arckit.py is next to this file) is demoted: its verified programs count as full-score partials (fallback pool, weight 0.5) instead of verified votes; auto = on for fnext")
    ap.add_argument('--tolerant-calls', default='auto', choices=['auto', 'on', 'off'], help='on: a tool call without the canonical <parameter=code> is parsed leniently (JSON arguments, code= attributes, bare programs; only code that parses as Python); auto = on for fnext')
    ap.add_argument('--effort', default='xhigh', help='qwen: reasoning_effort chat-template argument')
    ap.add_argument('--prompt', default='v1', help="the harness' prompt variant (repl_prompt.VARIANTS, if it has them)")
    ap.add_argument('--temp', type=float, default=None)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--top-k', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--session-ep', default='auto', choices=['auto', 'on', 'off'], help="on: every session (and direct attempt) is one server-side session (/arc/session, the Dots3 runtime's ARC_SESSION_EP): each tool turn pushes only the new tokens (tool response) into the kept KV instead of a new request that re-prefills the whole context; auto = on for dots3")
    ap.add_argument('--session-chunk', type=int, default=1024, help='session generation chunk (tokens): the granularity of solver cuts')
    ap.add_argument('--tok-cache', default='off', choices=['on', 'off'], help='client tokenizer memo (default off): identical tokenize / chat-template calls (the harness and the session endpoint both tokenize every tool response; every session of a task renders and tokenizes the same prompt) are answered from an LRU cache; same results')
    ap.add_argument('--start-stops', default='off', choices=['on', 'off'], help='session start (default off): send the stop token ids and a full chunk with /arc/session (the Flash-Next session route honours them) instead of a 1-token first chunk + a push with the stops (one round trip and one extra resume/mixed step per session less)')
    ap.add_argument('--turn-log', default=None, help='profiling (default off): jsonl, one row per session turn (client tokenize / push RTT / time to first token / chunk-end waits / harness gap incl. tool exec); never changes behaviour')
    ap.add_argument('--direct', default='off', choices=['off', 'on'], help="on: also one direct-answer attempt per test input (the direct solver's prompt from --template, no tools, forced conclusion at its cap), scheduled with the first sessions")
    ap.add_argument('--direct-cap', type=int, default=98304)
    ap.add_argument('--direct-min-cap', type=int, default=24000)
    ap.add_argument('--direct-force-tokens', type=int, default=3000)
    ap.add_argument('--direct-force-ctx-chars', type=int, default=24000)
    ap.add_argument('--direct-temp', type=float, default=0.7)
    ap.add_argument('--select', default='vote', choices=['vote', 'agree', 'toolfirst', 'route'], help="vote: attempts = the harness vote (verified outputs by votes, then answers). agree: a REPL output is attempt_1 only when >= --agree-n independent sessions' verified programs agree on it; otherwise the direct answer is attempt_1 and the top verified output (if any) attempt_2")
    ap.add_argument('--agree-n', type=int, default=2)
    ap.add_argument('--route-side', type=int, default=20)
    ap.add_argument('--direct-cap2', type=int, default=120000)
    ap.add_argument('--direct-min-cap2', type=int, default=40000)
    ap.add_argument('--tool-yield', type=float, default=0.7, help="toolfirst planner: expected share of a tool task's inputs that still need direct attempt 1")
    ap.add_argument('--budget-margin-s', type=float, default=900, help='the planner sizes new sessions to finish this long before the deadline (covers the endgame)')
    ap.add_argument('--endgame-s', type=float, default=600, help='this long before the deadline every running session is cut and concluded')
    ap.add_argument('--tail-fill', default='off', choices=['on', 'off'], help="end of run (default off): once a session started now could not use --sample-cap before the endgame (per-session rate x time to the endgame), later sessions (tier >= 1) are admitted with caps sized to end before the endgame (>= --tail-min-cap) instead of needing --min-cap of free token budget (in the tail, time binds, not tokens: the running sessions' cap reservations stop admission ~40 min early and the GPU ramps down). KV admission is unchanged; no effect earlier")
    ap.add_argument('--tail-min-cap', type=int, default=16384)
    ap.add_argument('--tail-frac', type=float, default=0.85, help='tail-fill: new caps = this fraction of (per-session rate x seconds to the endgame)')
    ap.add_argument('--drain-fill', default='off', choices=['on', 'off'], help='end of run (default off): keep the batch full until a late, measured endgame cut instead of the fixed --endgame-s one. The cut happens at deadline - G: G = time for every running session to end its current chunk (--session-chunk) before the last call (deadline - 120 s) and to generate its forced answer before deadline - 60 s, at the measured per-session rate, + --drain-guard-safety, at least --drain-guard-min and at most --endgame-s (so never earlier than the fixed cut). In the time-bound tail (where --tail-fill shrinks caps) tier>=1 sessions get their full cap (the cut concludes them) until --drain-min-s before the cut; KV / context admission unchanged. Sessions still running at deadline - 60 s keep their programs (recorded, cut=deadline) instead of being dropped')
    ap.add_argument('--drain-guard-min', type=float, default=180, help='drain-fill: the cut is at least this long before the deadline')
    ap.add_argument('--drain-guard-safety', type=float, default=45, help='drain-fill: seconds added to the measured conclusion time')
    ap.add_argument('--drain-min-s', type=float, default=300, help='drain-fill: no new session when less than this remains before the cut')
    ap.add_argument('--drain-rate-floor', type=float, default=8.0, help='drain-fill: per-session tok/s floor of the conclusion-time estimate')
    ap.add_argument('--endgame-flag', default=None, help='file created at the endgame cut (the launcher starts its DP keepalive on it)')
    ap.add_argument('--rate0', type=float, default=600.0, help='generated tok/s assumed until the server rate is known')
    ap.add_argument('--rate-window-s', type=float, default=600)
    ap.add_argument('--sandbox-uid', type=int, default=65534, help='when running as root, run programs as this uid/gid (-1: never switch user)')
    a = ap.parse_args(argv)
    if a.samples_out is None and a.log:
        a.samples_out = str(Path(a.log).with_name('repl_samples.jsonl'))
    if a.select in ('toolfirst', 'route'):
        a.direct = 'on'
    if a.session_ep == 'auto':
        a.session_ep = 'on' if a.backend in ('dots3', 'fnext') else 'off'
    if a.arckit != 'off' and a.prompt == 'v1':
        a.prompt = 'ak'
    if a.demote_flagged == 'auto':
        a.demote_flagged = 'on' if a.backend == 'fnext' else 'off'
    if a.fn_fixes == 'auto':
        a.fn_fixes = 'on' if a.backend == 'fnext' else 'off'
    if a.tolerant_calls == 'auto':
        a.tolerant_calls = 'on' if a.backend == 'fnext' else 'off'
    for k, v in SAMPLING[a.backend].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a
GUARD = '\nimport sys as _gs, os as _go\n_gs.dont_write_bytecode = True\n_GCWD = _go.path.realpath(_go.getcwd())\n_GDENY = {"os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "subprocess.Popen",\n          "os.kill", "os.killpg", "pty.spawn", "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",\n          "socket.getaddrinfo", "socket.gethostbyname", "resource.setrlimit", "os.setuid", "os.setgid"}\n_GPATH = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink",\n          "os.truncate", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copytree"}\n_GW = _go.O_WRONLY | _go.O_RDWR | _go.O_CREAT | _go.O_APPEND | _go.O_TRUNC\ndef _ginside(p):\n    try:\n        if isinstance(p, int):\n            return True\n        q = _go.path.realpath(_go.fsdecode(p))\n        return q == _GCWD or q.startswith(_GCWD + _go.sep)\n    except Exception:\n        return False\ndef _gguard(ev, args):\n    if ev == "open":\n        p, mode, flags = (tuple(args) + (None, None, 0))[:3]\n        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _GW)\n        if w and not _ginside(p):\n            raise PermissionError("sandbox: files can only be written in the working directory")\n    elif ev in _GDENY:\n        raise PermissionError("sandbox: " + ev + " is not allowed")\n    elif ev in _GPATH:\n        for x in args[:2]:\n            if isinstance(x, (str, bytes, _go.PathLike)) and not _ginside(x):\n                raise PermissionError("sandbox: " + ev + " outside the working directory is not allowed")\n_gs.addaudithook(_gguard)\ndel _gguard\n'
_ORIG_LIMITS = sandbox._limits
SANDBOX_UID = [None]

def _hardened_limits(timeout):
    base = _ORIG_LIMITS(timeout)
    uid = SANDBOX_UID[0]

    def f():
        base()
        try:
            os.nice(10)
        except OSError:
            pass
        if uid is not None:
            os.chown('.', uid, uid)
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
    return f

def harden_sandbox(a):
    sandbox.RUNNER = GUARD + sandbox.RUNNER
    sandbox._limits = _hardened_limits
    n = a.exec_workers or max(4, min(32, (os.cpu_count() or 8) - 8))
    sandbox.MAX_CONCURRENT = 1 << 20
    if a.sandbox_uid >= 0 and hasattr(os, 'geteuid') and (os.geteuid() == 0):
        SANDBOX_UID[0] = a.sandbox_uid
    return n
SELFTEST_TASK = {'train': [{'input': [[1, 2]], 'output': [[2, 1]]}], 'test': [{'input': [[3, 4]]}]}
SELFTEST = "def transform(g):\n    return [r[::-1] for r in g]\ntry:\n    open('/tmp/_sandbox_probe_should_fail', 'w')\n    print('WRITE-ALLOWED')\nexcept PermissionError:\n    print('write-blocked')\n"

async def sandbox_selftest(log):
    for attempt in range(2):
        res = await sandbox.run_code(SELFTEST, SELFTEST_TASK, 10.0)
        ok = sandbox.verified(res) and res['tests'][0]['out'] == [[4, 3]]
        blocked = 'write-blocked' in (res.get('stdout') or '')
        log('sandbox self-test', f'uid={SANDBOX_UID[0]}', 'verified' if ok else f'FAILED {repr(res)[:300]}', 'outside-write blocked' if blocked else 'outside-write NOT blocked')
        if ok or SANDBOX_UID[0] is None:
            return ok
        SANDBOX_UID[0] = None
    return False
CUR = contextvars.ContextVar('repl_sample', default=None)

class LiveCap:

    def __init__(self, cap):
        self.cap, self.limit = (int(cap), None)

    def __sub__(self, used):
        return (self.cap if self.limit is None else min(self.cap, self.limit)) - used

    def __int__(self):
        return self.cap

class Sample:

    def __init__(self, task, idx, cap, res, kind='repl', k=None):
        self.task, self.idx, self.res, self.kind, self.k = (task, idx, res, kind, k)
        self.cap = LiveCap(cap)
        self.used_done = 0
        self.live = 0
        self.cut = ''
        self.shadow = []
        self.n_exec = 0
        self.exec_s = 0.0
        self.wait_s = 0.0
        self.n_retry = 0
        self.last_code = None
        self.prev_codes = []
        self.ver_code = None
        self.n_carry = 0
        self.n_timeout = 0
        self.fmt_rec = 0
        self.fmt_fail = 0
        self.req_fail = 0
        self.ep = None
        self.ctx = 0
        self.t0 = time.time()

    def progress(self):
        return self.used_done + self.live

class Task:

    def __init__(self, tid, task, order, gt):
        self.tid, self.task, self.order, self.gt = (tid, task, order, gt)
        self.n_test = len(task['test'])
        self.recs, self.launched, self.inflight, self.used, self.reserved = ([], 0, 0, 0, 0)
        self.running = set()
        self._tier = None
        self.direct = [None] * self.n_test
        self.direct_todo = []
        self.direct_recs = []
        self.tool = True
        self.direct2 = [None] * self.n_test
        self.d1_natural = [None] * self.n_test
        self.d1_ct = [None] * self.n_test
        self.d2_launched = set()
        self.carry = []
        self.carry_ver = False
        self.n_carry = 0

    def carry_add(self, shadow):
        if any((p.get('verified') for p in shadow)):
            self.carry_ver = True
            return
        ps = [p for p in shadow if p.get('code') and (p.get('n_ok') or 0) >= 1]
        if ps:
            b = max(ps, key=lambda p: (p['n_ok'], p['k']))
            self.carry.append((b['code'], list(b['train_ok'])))

    def carry_note(self):
        if self.carry_ver or not self.carry:
            return None
        code, ok = self.carry[self.n_carry % len(self.carry)]
        self.n_carry += 1
        return carry_note(code, ok)

    def tier(self, a):
        if a.select in ('toolfirst', 'route'):
            return (0 if a.select == 'toolfirst' else 1) if self.tool and self.launched == 0 else None
        if self.launched >= a.kmax:
            return None
        if self.launched == 0:
            return 0
        if a.policy == 'keep':
            if self._tier is None:
                r = repl_vote.rank(self.recs, self.n_test)
                self._tier = -1 if r and all((x['top_votes'] >= a.agree for x in r)) else 1
                if self._tier == 1 and (a.consistency_flags == 'on' or a.settle_last == 'on') and settled(self.recs, self.n_test):
                    self._tier = 3
            return self._tier if self._tier > 0 else None
        if self._tier is None:
            r = repl_vote.rank(self.recs, self.n_test)
            self._tier = 1 if any((x['n_ver_samples'] == 0 for x in r)) else 2 if any((x['top_votes'] < a.agree for x in r)) else -1
        if self._tier == 1:
            return 1
        if self._tier == 2 and self.inflight == 0:
            return 2
        return None

class ServerError(Exception):
    pass
_DIGROW = re.compile('^\\s*[0-9](?:[ ,]?[0-9])*\\s*$')

def _as_grid(g):
    if isinstance(g, list) and g and all((isinstance(r, str) for r in g)):
        g = [[c for c in r if c.isdigit()] for r in g]
    if not (isinstance(g, list) and g and all((isinstance(r, list) and r for r in g))):
        return None
    try:
        out = [[int(v) for v in r] for r in g]
    except Exception:
        return None
    if len(out) > 30 or len({len(r) for r in out}) != 1 or len(out[0]) > 30 or any((v < 0 or v > 9 for r in out for v in r)):
        return None
    return out

def parse_grids_ext(text, _orig=None):
    out = _orig(text) if _orig else []
    if out:
        return out
    for blk in re.findall('```(?:json|text|python)?\\s*\\n(.*?)```', text, re.S):
        b = blk.strip()
        try:
            g = json.loads(b)
        except Exception:
            g = None
        if g is not None:
            one = _as_grid(g)
            if one is not None:
                out.append(one)
                continue
            if isinstance(g, list) and g and all((_as_grid(x) is not None for x in g)):
                out.extend((_as_grid(x) for x in g))
                continue
        rows = [ln for ln in b.splitlines() if ln.strip()]
        if rows and all((_DIGROW.match(ln) for ln in rows)):
            one = _as_grid([re.sub('[ ,]', '', ln) for ln in rows])
            if one is not None:
                out.append(one)
    if out:
        return out
    for m in reversed(list(re.finditer('\\[\\s*\\[[\\d\\s,\\"\'\\[\\]]*\\]\\s*\\]', text))):
        try:
            g = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            continue
        one = _as_grid(g)
        if one is not None:
            return [one]
    return []

def force_tokens_for(task, base):
    cells, rows = (0, 0)
    for g in [p['output'] for p in task['train']] + [t['input'] for t in task['test']]:
        cells, rows = (max(cells, len(g) * len(g[0])), max(rows, len(g)))
    return max(int(base), int(2.6 * cells + 3 * rows + 200))
_PRE = 'from collections import Counter, defaultdict, deque\nimport collections, itertools, math\n'

def _guard_import(src):
    return 'try:\n' + '\n'.join(('    ' + x for x in src.split('\n'))) + '\nexcept ImportError:\n    pass'

def carry_defs(prev_codes, code, guard=False, layout=None):
    import ast
    names = lambda src: {x.id for x in ast.walk(ast.parse(src)) if isinstance(x, ast.Name)}
    try:
        tree = ast.parse(code)
    except Exception:
        return code
    if any((isinstance(n, ast.ImportFrom) and n.module == '__future__' for n in tree.body)):
        return code
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    used = names(code)
    imports, last = ([], {})
    for prev in prev_codes:
        try:
            pt = ast.parse(prev)
        except Exception:
            continue
        for n in pt.body:
            if isinstance(n, ast.ImportFrom) and n.module == '__future__':
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                src = ast.get_source_segment(prev, n)
                imports.append(({(a.asname or a.name).split('.')[0] for a in n.names}, _guard_import(src) if guard and src else src))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name not in defined:
                last[n.name] = ast.get_source_segment(prev, n)
    need, frontier = (set(), used & set(last))
    while frontier:
        n = frontier.pop()
        need.add(n)
        try:
            sub = names(last[n])
        except Exception:
            sub = set()
        used |= sub
        frontier |= (sub & set(last)) - need
    pre, have, lbl = ([], set(), [])
    for bound, src in imports:
        if bound & used and (not bound <= have) and src:
            pre.append(src)
            lbl.append(('import', src.strip().splitlines()[1 if guard and src.startswith('try:\n') else 0].strip()))
            have |= bound
    for n in last:
        if n in need:
            pre.append(last[n])
            lbl.append(('definition', n))
    if layout is not None:
        ln = 1
        for p, l in zip(pre, lbl):
            if p:
                k = p.count('\n') + 1
                layout.append((ln, ln + k - 1, l[0], l[1], p))
                ln += k + 1
    return '\n\n'.join((p for p in pre if p)) + '\n\n' + code if pre else code

_TB_FRAME = re.compile('^(\\s*)File "<tool>", line (\\d+)(.*)$')

def tb_fix(err, code0, off, layout):
    if not err or off <= 0:
        return err
    src0 = code0.split('\n')
    lines = err.split('\n')
    lastl = next((x for x in reversed(lines) if x.strip()), '')
    syn = bool(re.match('\\s*(SyntaxError|IndentationError|TabError)\\b', lastl))
    out = []
    for x in lines:
        m = _TB_FRAME.match(x)
        if not m:
            out.append(x)
            continue
        ind, n, rest = (m.group(1), int(m.group(2)), m.group(3))
        if n > off:
            k = n - off
            out.append(f'{ind}File "<tool>", line {k}{rest}')
            if not syn and 0 < k <= len(src0) and src0[k - 1].strip():
                out.append(ind + '  ' + src0[k - 1].strip())
            continue
        seg = next((g for g in layout if g[0] <= n <= g[1]), None)
        if seg is None:
            out.append(x)
            continue
        out.append(f'{ind}File "<tool>", in a carried {seg[2]} from an earlier cell: {seg[3]} (line {n - seg[0] + 1} of it){rest}')
        pl = seg[4].split('\n')[n - seg[0]]
        if not syn and pl.strip():
            out.append(ind + '  ' + pl.strip())
    s = '\n'.join(out)
    return re.sub('\\(detected at line (\\d+)\\)', lambda m: f'(detected at line {int(m.group(1)) - off})' if int(m.group(1)) > off else m.group(0), s)

def state_with_open_call(orig):

    def _state(seg):
        st = orig(seg)
        if st != 'call':
            i = seg.rfind('<tool_call>')
            if i >= 0 and '</tool_call>' not in seg[i:]:
                return 'call'
        return st
    return _state
AUX_RUNNER = '\nimport json, sys, io, contextlib, copy, socket, signal\ndef _blocked(*a, **k): raise OSError("network disabled")\nfor _m in ("connect", "connect_ex", "bind", "sendto", "sendmsg"): setattr(socket.socket, _m, _blocked)\nimport numpy as np\nfrom collections import Counter, defaultdict, deque\nimport collections, itertools, math\nD = json.load(open("task.json"))\ntrain = D["train"]; test_inputs = D["test_inputs"]\nns = {"np": np, "numpy": np, "train": copy.deepcopy(train), "test_inputs": copy.deepcopy(test_inputs),\n      "test_input": copy.deepcopy(test_inputs[0]), "__name__": "__main__", "__builtins__": __builtins__,\n      "Counter": Counter, "defaultdict": defaultdict, "deque": deque, "collections": collections,\n      "itertools": itertools, "math": math}\nres = {}\ntry:\n    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n        exec(compile(open("code.py").read(), "<tool>", "exec"), ns)\nexcept BaseException as e:\n    res["load_error"] = repr(e)[:200]\nf = ns.get("transform")\nclass TO(Exception): pass\ndef _alarm(*a): raise TO()\nsignal.signal(signal.SIGALRM, _alarm)\ndef call(g):\n    signal.alarm(4)\n    try:\n        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n            try:\n                o = f(copy.deepcopy(g))\n            except TO:\n                raise\n            except Exception:\n                o = f(np.array(g))\n        a = np.asarray(o)\n        if a.ndim != 2 or a.size == 0: return None\n        return [[int(v) for v in r] for r in a.tolist()]\n    except BaseException:\n        return None\n    finally:\n        signal.alarm(0)\ndef d8(g, k):\n    a = np.array(g)\n    if k >= 4: a = a.T\n    return np.rot90(a, k % 4).tolist()\ndef cperm(g, p):\n    return [[p[v] for v in r] for r in g]\naug_flag = False\nif f is not None:\n    base = [call(t) for t in test_inputs]\n    import random\n    for kind, k in [("d8", k) for k in range(1, 8)] + [("col", s) for s in range(3)]:\n        if kind == "d8":\n            fx = lambda g, k=k: d8(g, k)\n        else:\n            rr = random.Random(1000 + k); nz = list(range(1, 10)); sh = nz[:]; rr.shuffle(sh)\n            p = {0: 0, **dict(zip(nz, sh))}\n            fx = lambda g, p=p: cperm(g, p)\n        if all(call(fx(q["input"])) == fx(q["output"]) for q in train):\n            if any(b is not None and call(fx(t)) != fx(b) for t, b in zip(test_inputs, base)):\n                aug_flag = True\n                res.setdefault("aug_maps", []).append(f"{kind}{k}")\nres["aug_flag"] = aug_flag\ncov_flag = False\nif f is not None:\n    def cov(g):\n        seen = set()\n        def tr(fr, ev, arg):\n            if fr.f_code.co_filename == "<tool>":\n                if ev == "line":\n                    seen.add(fr.f_lineno)\n                return tr\n            return None\n        sys.settrace(tr)\n        try:\n            call(g)\n        finally:\n            sys.settrace(None)\n        return seen\n    try:\n        trl = set()\n        for q in train:\n            trl |= cov(q["input"])\n        cov_flag = any(cov(t) - trl for t in test_inputs)\n    except BaseException as e:\n        res["cov_error"] = repr(e)[:200]\nres["cov_flag"] = cov_flag\njson.dump(res, open("result.json", "w"))\n'

AK_ALIAS = 'sys.modules.setdefault("ak", ak)\n'
AUX_ALIAS = [False]

def aux_runner(ak=False):
    if not ak:
        return AUX_RUNNER
    a, b = ('D = json.load(open("task.json"))', 'ns = {"np": np,')
    assert AUX_RUNNER.count(a) == 1 and AUX_RUNNER.count(b) == 1, 'AUX_RUNNER layout changed'
    r = AUX_RUNNER.replace(a, 'sys.path.insert(0, ".")\nimport arckit as ak\n' + (AK_ALIAS if ak == 'alias' else '') + a, 1)
    return r.replace(b, 'ns = {"ak": ak, "arckit": ak, "np": np,', 1)

async def aux_check(code, task, timeout=90):
    import shutil as _sh, signal as _sig, tempfile as _tf
    d = _tf.mkdtemp(prefix='qa_')
    try:
        json.dump({'train': task['train'], 'test_inputs': [t['input'] for t in task['test']]}, open(f'{d}/task.json', 'w'))
        open(f'{d}/code.py', 'w').write(code)
        ak = bool(sandbox.ARCKIT) and os.path.isfile(sandbox.ARCKIT_PATH)
        open(f'{d}/runner.py', 'w').write(GUARD + aux_runner('alias' if ak and AUX_ALIAS[0] else ak))
        if ak:
            _sh.copyfile(sandbox.ARCKIT_PATH, f'{d}/arckit.py')
        cmd = [sandbox.PY, '-I', 'runner.py']
        if sandbox.UNSHARE:
            cmd = [sandbox.UNSHARE, '-rn'] + cmd
        env = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        p = await asyncio.create_subprocess_exec(*cmd, cwd=d, env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, preexec_fn=sandbox._limits(timeout))
        try:
            await asyncio.wait_for(p.wait(), timeout + 10)
        except asyncio.TimeoutError:
            try:
                os.killpg(p.pid, _sig.SIGKILL)
            except ProcessLookupError:
                pass
            await p.wait()
        try:
            return json.load(open(f'{d}/result.json'))
        except Exception:
            return {'error': 'no result'}
    finally:
        _sh.rmtree(d, ignore_errors=True)

def is_grid(g):
    return isinstance(g, list) and 0 < len(g) <= 30 and isinstance(g[0], list) and (0 < len(g[0]) <= 30) and all((isinstance(r, list) and len(r) == len(g[0]) and all((isinstance(v, int) and (not isinstance(v, bool)) and (0 <= v <= 9) for v in r)) for r in g))
_IF_RE = re.compile('\\bif\\b')
CARRY_MAX_CODE = 6000

def carry_note(code, train_ok):
    n = len(train_ok)
    ok = [str(i + 1) for i, x in enumerate(train_ok) if x]
    bad = [str(i + 1) for i, x in enumerate(train_ok) if not x]
    code = code.strip()
    if len(code) > CARRY_MAX_CODE:
        code = code[:CARRY_MAX_CODE] + '\n# ... (truncated)'
    res = f'It reproduces {len(ok)} of {n} examples' + (f' (example{('s' if len(ok) > 1 else '')} {', '.join(ok)})' if ok else '') + (f' and fails on example{('s' if len(bad) > 1 else '')} {', '.join(bad)}.' if bad else '.')
    return f'\n\n## Notes from an earlier attempt\nAn earlier attempt at this task ran out of budget without a program that reproduces every example. Its best `transform` is below. {res} Its idea may be partly right or wrong: reuse, fix or discard it.\n```python\n' + code + '\n```'

def aux_clean(rec):
    x = rec.get('aux') or {}
    return 0 if x.get('aug_flag') or x.get('cov_flag') else 1

def rank_tiebreak(recs, n_test, flags=False):
    res = []
    cands = [repl_vote.sample_candidates(r, n_test) for r in recs]
    for k in range(n_test):
        vv, first, grids, nif, fb, ok = ({}, {}, {}, {}, {}, {})
        for i, c in enumerate(cands):
            g = c[k]['verified']
            if g is not None:
                q = repl_vote.key(g)
                vv[q] = vv.get(q, 0) + 1
                first.setdefault(q, i)
                grids[q] = g
                nif[q] = max(nif.get(q, 0), int(recs[i].get('ver_nif') or 0))
                ok[q] = max(ok.get(q, 0), aux_clean(recs[i]) if flags else 0)
        for i, c in enumerate(cands):
            for src, w in (('answer', 1.0), ('partial', None)):
                x = c[k][src]
                if x is None:
                    continue
                g, w = (x, w) if src == 'answer' else (x[0], 0.5 * x[1])
                q = repl_vote.key(g)
                if q in vv:
                    continue
                fb[q] = fb.get(q, 0.0) + w
                first.setdefault(q, i)
                grids[q] = g
        order = sorted(vv, key=lambda q: (-vv[q], -ok[q], -nif[q], first[q])) + sorted(fb, key=lambda q: (-fb[q], first[q]))
        res.append({'attempts': [grids[q] for q in order[:2]], 'n_verified_votes': [vv[q] for q in order[:2] if q in vv], 'n_ver_samples': sum(vv.values()), 'n_distinct_ver': len(vv), 'top_votes': max(vv.values()) if vv else 0, 'from': ['ver' if q in vv else 'fb' for q in order[:2]]})
    return res

def order_answers(task, ans):
    n = len(task['test'])
    if ARCKIT is None or n < 2 or not ans or len(ans) > n or not all((is_grid(g) for g in ans)):
        return None
    try:
        rules = ARCKIT.size_rule(task['train'], all_rules=True)
        preds = []
        for t in task['test']:
            ps = {r.predict(t['input']) for r in rules or []}
            if len(ps) != 1 or None in ps:
                return None
            preds.append(tuple(next(iter(ps))))
    except Exception:
        return None
    if len(set(preds)) != n:
        return None
    pos = []
    for g in ans:
        s = (len(g), len(g[0]))
        if s not in preds:
            return None
        pos.append(preds.index(s))
    if len(set(pos)) != len(pos) or pos == list(range(len(ans))):
        return None
    out = [None] * (max(pos) + 1)
    for g, k in zip(ans, pos):
        out[k] = g
    return out

def settled(recs, n_test):
    if not recs:
        return False
    for x in repl_vote.rank(recs, n_test):
        vv = x['n_verified_votes']
        if not vv or vv[0] < 2 or (len(vv) > 1 and vv[0] - vv[1] < 1):
            return False
    return True

def _ast_ok(code):
    import ast
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def _unjson(body):
    try:
        return json.loads('"' + body + '"', strict=False)
    except Exception:
        pass
    return body.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
_TQ = re.compile('code\\s*=\\s*(\\"\\"\\"|\'\'\')(.*?)(\\1|$)', re.S)

def tolerant_code(seg):
    i = seg.rfind('<tool_call>')
    if i < 0:
        return None
    t = seg[i + len('<tool_call>'):]
    for e in ('</tool_call>', '<|im_end|>'):
        j = t.find(e)
        if j >= 0:
            t = t[:j]
    cands = []
    m = re.search('\\\\?"code\\\\?"\\s*:\\s*\\\\?"', t)
    if m:
        body = t[m.end():]
        k = None
        for pat in ('"\\s*,\\s*"\\w+"\\s*:', '"\\s*\\}', '"?\\s*\\n\\s*</'):
            mm = re.search(pat, body)
            if mm and (k is None or mm.start() < k):
                k = mm.start()
        cands.append(_unjson(body[:k] if k is not None else body.rstrip().rstrip('"')))
    m = _TQ.search(t)
    if m:
        cands.append(m.group(2))
    m = re.search('\\bcode="((?:[^"\\\\]|\\\\.)*)"', t, re.S)
    if m:
        cands.append(_unjson(m.group(1)))
    m = re.search('<function=\\w+>?\\s*\\n(.*?)(?:\\n\\s*</parameter>|\\n\\s*</function>|$)', t, re.S)
    if m and (not m.group(1).lstrip().startswith(('{', '<'))):
        cands.append(m.group(1))
    rest = [ln for ln in t.splitlines() if not re.match('\\s*</?[^>\\n]*>\\s*$', ln) and ln.strip() not in ('python', 'code', 'call:python')]
    if rest and len('\n'.join(rest).strip()) > 8:
        cands.append('\n'.join(rest))
    for c in cands:
        c = (c or '').strip('\n')
        c = re.sub('^\\s*```(?:python)?\\s*\\n', '', c)
        c = re.sub('\\n```\\s*$', '', c)
        if c.strip() and _ast_ok(c):
            return c
    return None

class _SessCfg:

    def __init__(self, a, t):
        self._a, self._t = (a, t)

    def __getattr__(self, k):
        return getattr(self._a, k)

    @property
    def force_tokens(self):
        a, t = (self._a, self._t)
        if a.force_skip_verified and t.recs:
            r = repl_vote.rank(t.recs, t.n_test)
            if r and all((x['n_ver_samples'] > 0 for x in r)):
                return 0
        if a.fn_fixes == 'on' and a.force_tokens > 0:
            return force_tokens_for(t.task, a.force_tokens)
        return a.force_tokens

class Solver:

    def __init__(self, a):
        self.a = a
        self.T0 = time.time()
        self.DEADLINE = self.T0 + a.deadline_s
        self.ENDGAME = self.DEADLINE - a.endgame_s
        self.LAST_CALL = self.DEADLINE - min(120.0, a.endgame_s / 2)
        self.base = f'http://{a.host}:{a.port}'
        self.logf = open(a.log, 'a') if a.log else None
        self.samples_f = open(a.samples_out, 'a') if a.samples_out else None
        self.turn_f = open(a.turn_log, 'a') if getattr(a, 'turn_log', None) else None
        self.server_ok = True
        self.kv = 0.0
        self.kv_tokens = None
        self.final_ctx = []
        self.gen_hist = []
        self.gen_off = [0.0, 0.0]
        self.metrics = {}
        self.metrics0 = None
        self.running = {}
        self.n_seed = a.seed * 100000
        self.tool_stats = []
        self.endgame_done = False
        self._cut_cache = (0.0, None)
        self._drain_closed = False
        self.n_timeouts = 0
        self.n_demoted = 0

    def log(self, *x):
        s = f'[repl {time.time() - self.T0:7.0f}s] ' + ' '.join((str(v) for v in x))
        print(s, flush=True)
        if self.logf:
            self.logf.write(s + '\n')
            self.logf.flush()

    def load(self):
        a = self.a
        chal = json.load(open(a.challenges))
        if a.tasks:
            keep = set(open(a.tasks).read().split())
            chal = {k: v for k, v in chal.items() if k in keep}
        sols = json.load(open(a.solutions)) if a.solutions else {}
        self.chal, self.sols = (chal, sols)
        ids = list(chal)
        self.tasks = []
        for i, t in enumerate(ids):
            task = {'train': chal[t]['train'], 'test': [{'input': x['input']} for x in chal[t]['test']]}
            self.tasks.append(Task(t, task, i, sols.get(t)))
        self.n_inputs = sum((t.n_test for t in self.tasks))
        if a.direct == 'on':
            assert a.template and Path(a.template).exists(), "--direct on needs --template (the direct solver's prompt)"
            self.template = Path(a.template).read_text()
            for t in self.tasks:
                if a.select in ('toolfirst', 'route'):
                    side = max((max(len(g), len(g[0])) for g in [p['input'] for p in t.task['train']] + [x['input'] for x in t.task['test']]))
                    t.tool = a.select == 'toolfirst' or side >= a.route_side
                    t.direct_todo = [] if t.tool else list(range(t.n_test))
                else:
                    t.direct_todo = list(range(t.n_test))

    def load_backend(self):
        a = self.a
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.tokenizer)
        if getattr(a, 'tok_cache', 'off') == 'on':
            self.tok = TokCache(self.tok)
        import repl_prompt
        if a.arckit != 'off':
            path = str(HERE / 'arckit.py')
            if not hasattr(sandbox, '_runner') or not os.path.isfile(path):
                raise SystemExit('--arckit needs the harness hooks and arckit.py next to the solver')
            sandbox.ARCKIT, sandbox.ARCKIT_PATH = (a.arckit, path)
            if a.import_fix == 'alias':
                assert sandbox.AK_PRE.count('import arckit as ak\n') == 1, 'AK_PRE layout changed'
                sandbox.AK_PRE = sandbox.AK_PRE.replace('import arckit as ak\n', 'import arckit as ak\n' + AK_ALIAS, 1)
                AUX_ALIAS[0] = True
        if a.prompt != 'v1' and a.prompt not in getattr(repl_prompt, 'VARIANTS', {}):
            raise SystemExit(f'--prompt {a.prompt}: not a variant of this harness')
        if a.backend in ('qwen', 'fnext'):
            import repl_session as sess
            import repl_prompt as P
            self.sess = sess
            msgs = lambda task: [{'role': 'system', 'content': P.SYSTEM}, {'role': 'user', 'content': P.task_prompt(task, *([a.prompt] if hasattr(P, 'VARIANTS') else []))}]
            self.render = lambda task: self.tok.apply_chat_template(msgs(task), tools=[P.TOOL], tokenize=False, add_generation_prompt=True, reasoning_effort=a.effort)
            self.fmt = {'nudge': sess.NUDGE, 'close': sess.CLOSE_CALL, 'state': sess._state, 'extract': P.extract_code, 'force': sess.FORCE, 'force_after': sess.FORCE_AFTER_THINK, 'parse': P.parse_grids, 'turn_open': '<|im_start|>assistant\n<think>\n', 'force_stop': None, 'close_fence': False, 'call_open': '<tool_call>', 'call_end': '</tool_call>', 'tool_resp': lambda r: '<|im_end|>\n<|im_start|>user\n<tool_response>\n' + r + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'}
        else:
            import repl_dsession as sess
            import repl_d3fmt as F
            self.sess = sess
            self.render = lambda task: F.render_prompt(self.tok, task)
            self.fmt = {'nudge': sess.NUDGE, 'close': F.CLOSE_CALL, 'state': F.state, 'extract': F.extract_code, 'force': sess.FORCE, 'force_after': sess.FORCE_AFTER_THINK, 'parse': F.parse_grids, 'turn_open': F.RESP_POST, 'force_stop': ['<|endofassistant|>'], 'close_fence': True, 'call_open': F.CALL_OPEN, 'call_end': F.CALL_END, 'tool_resp': F.tool_response}
        sess.run_code = self.run_code_tracked
        if a.carry == 'on':
            tp0 = sess.task_prompt
            sess.task_prompt = lambda task, *v, _f=tp0: _f(task, *v) + (task.get('_carry') or '')
        if a.fn_fixes == 'on':
            pg = sess.parse_grids
            sess.parse_grids = lambda text, _o=pg: parse_grids_ext(text, _o)
            self.fmt['parse'] = sess.parse_grids
            sess._state = state_with_open_call(sess._state)
            self.fmt['state'] = sess._state
            sandbox.OUT_LIMIT = 6000
            fb0 = sandbox.feedback

            def feedback(res, task, timeout=10.0, *aa, _f=fb0, **kw):
                txt = _f(res, task, timeout, *aa, **kw)
                if 'NameError' in (res.get('error') or ''):
                    txt += '\n\nNote: every python call starts in a fresh process; nothing persists between calls. Include all imports and definitions in each call.'
                return txt
            sandbox.feedback = feedback
            sess.feedback = feedback
            a0 = '"__builtins__": __builtins__}\n'
            assert a0 in sandbox.RUNNER
            sandbox.RUNNER = sandbox.RUNNER.replace(a0, a0 + _PRE + 'ns.update(Counter=Counter, defaultdict=defaultdict, deque=deque, collections=collections, itertools=itertools, math=math)\n', 1)
        if a.tolerant_calls == 'on':
            orig = sess.extract_code

            def extract_code(seg, _orig=orig):
                c = _orig(seg)
                smp = CUR.get()
                if c is not None and (not _ast_ok(c)):
                    c2 = re.sub('(\\n\\s*</[^\\n]*)+\\s*$', '', c.rstrip())
                    if c2 and _ast_ok(c2):
                        c = c2
                if c is None and '<tool_call>' in seg:
                    c = tolerant_code(seg)
                    if smp is not None:
                        smp.fmt_rec += c is not None
                        smp.fmt_fail += c is None
                return c
            sess.extract_code = extract_code
            self.fmt['extract'] = extract_code

    async def run_code_tracked(self, code, task, timeout=10.0):
        smp = CUR.get()
        t0 = time.time()
        code0 = code
        if self.a.fn_fixes == 'on' and smp is not None:
            layout = [] if self.a.tb_fix == 'on' else None
            code = carry_defs(smp.prev_codes, code0, guard=self.a.import_fix != 'off', layout=layout)
            smp.n_carry += code != code0
            smp.prev_codes.append(code0)
        async with self.gate:
            t1 = time.time()
            res = await sandbox.run_code(code, task, timeout)
        t2 = time.time()
        if self.a.tb_fix == 'on' and code != code0 and res.get('error'):
            try:
                res['error'] = tb_fix(res['error'], code0, code.count('\n') - code0.count('\n'), layout or [])
            except Exception:
                pass
        self.n_timeouts += bool(res.get('timeout'))
        if smp is not None:
            smp.n_timeout += bool(res.get('timeout'))
        if smp is not None:
            smp.last_code = code0
        self.tool_stats.append((t2 - t1, t1 - t0))
        if smp is not None:
            smp.n_exec += 1
            smp.exec_s += t2 - t1
            smp.wait_s += t1 - t0
            if sandbox.verified(res):
                smp.ver_code = code
                smp.ver_nif = len(_IF_RE.findall(code0))
            if res.get('has_transform') and 'train' in res:
                ok = [bool(x.get('ok')) for x in res['train']]
                smp.shadow.append({'k': smp.n_exec, 'train_ok': ok, 'n_ok': sum(ok), 'verified': sandbox.verified(res), 'tests': [t.get('out') for t in res.get('tests', [])], 'used': smp.progress(), 'code': code0 if self.a.carry == 'on' else None})
        return res

    async def metrics_loop(self, c):
        while time.time() < self.DEADLINE:
            try:
                m = (await c.get(self.base + '/metrics', timeout=10)).text
                val = lambda k: sum((float(x) for x in re.findall(f'^{k}(?:{{[^}}]*}})? ([0-9.e+-]+)', m, re.M)))
                g = val('vllm:generation_tokens_total')
                if g < self.gen_off[1]:
                    self.gen_off[0] += self.gen_off[1]
                self.gen_off[1] = g
                self.gen_hist.append((time.time(), g + self.gen_off[0]))
                del self.gen_hist[:-400]
                kv = re.findall('^vllm:kv_cache_usage_perc(?:\\{[^}]*\\})? ([0-9.e+-]+)', m, re.M)
                if kv:
                    _kv = [float(x) for x in kv]
                    self.kv = max(_kv) if self.a.kv_agg == 'max' else sum(_kv) / len(_kv)
                cap = re.findall('^vllm:cache_config_info\\{[^}]*kv_cache_size_tokens="(\\d+)"', m, re.M)
                if cap:
                    self.kv_tokens = sum((int(x) for x in cap))
                self.metrics = {'gen': g + self.gen_off[0], 'prompt': val('vllm:prompt_tokens_total'), 'hits': val('vllm:prefix_cache_hits_total'), 'queries': val('vllm:prefix_cache_queries_total'), 'running': val('vllm:num_requests_running'), 'waiting': val('vllm:num_requests_waiting'), 'preempt': val('vllm:num_preemptions_total'), 't': time.time()}
                if self.metrics0 is None and self.metrics['gen'] > 0:
                    self.metrics0 = dict(self.metrics)
            except Exception:
                pass
            await asyncio.sleep(20)

    def rate(self):
        h = self.gen_hist
        if len(h) < 2:
            return None
        t1, g1 = h[-1]
        t0, g0 = next(((t, g) for t, g in h if t >= t1 - self.a.rate_window_s), h[0])
        if t1 - t0 < 120:
            return None
        return (g1 - g0) / (t1 - t0)

    async def health_loop(self, c):
        fails = 0
        while time.time() < self.DEADLINE:
            try:
                ok = (await c.get(self.base + '/health', timeout=10)).status_code == 200
            except Exception:
                ok = False
            fails = 0 if ok else fails + 1
            if ok and (not self.server_ok):
                self.log('server healthy again: admission resumes')
            if ok or fails >= 2:
                if self.server_ok and (not ok):
                    self.log('server unhealthy: admission paused, requests wait')
                self.server_ok = ok
            await asyncio.sleep(5)

    async def wait_server(self):
        while not self.server_ok:
            if time.time() > self.DEADLINE - 60:
                raise ServerError('deadline while the server is down')
            await asyncio.sleep(2)

    def free_budget(self):
        r = self.rate()
        r = self.a.rate0 if r is None else r
        horizon = max(0.0, self.DEADLINE - self.a.budget_margin_s - time.time())
        committed = sum((max(0, s.res - s.progress()) for s in self.running.values()))
        return r * horizon - committed

    def sess_rate(self):
        r = self.rate()
        r = self.a.rate0 if r is None else r
        return r / max(1, len(self.running))

    def guard_s(self):
        """drain-fill: seconds before the deadline at which the running sessions must be cut: each must end its current
        chunk before the last call (deadline - 120 s) and its forced answer before the loop exit (deadline - 60 s)"""
        a = self.a
        try:
            r = self.rate()
            r = a.rate0 if r is None else r
            rs = max(a.drain_rate_floor, r / max(1, len(self.running)))
            force = max([_SessCfg(a, s.task).force_tokens * s.task.n_test for s in self.running.values() if s.kind == 'repl'] or [0])
            need = max(120.0 + a.session_chunk / rs, 60.0 + (a.session_chunk + force) / rs) + a.drain_guard_safety
            return min(a.endgame_s, max(a.drain_guard_min, need))
        except Exception as ex:          # runs in the main loop: never let an estimate end the run; fall back to the fixed cut
            if not getattr(self, '_guard_err', False):
                self._guard_err = True
                self.log('drain: guard estimate failed, using the fixed cut:', repr(ex)[:160])
            return a.endgame_s

    def cut_time(self):
        if self.a.drain_fill != 'on':
            return self.ENDGAME
        now = time.time()
        t, v = self._cut_cache
        if v is None or now - t > 2.0:
            v = self.DEADLINE - self.guard_s()
            self._cut_cache = (now, v)
        return v

    def time_cap(self):
        if self.a.tail_fill != 'on' and self.a.drain_fill != 'on':
            return None
        c = int(self.a.tail_frac * self.sess_rate() * max(0.0, self.cut_time() - time.time()))
        return c if c < self.a.sample_cap else None

    def ctx_projection(self):
        fin = self.final_ctx[-50:]
        typ = max(self.a.ctx_typ, int(0.6 * sum(fin) / len(fin)) if fin else 0)
        return (typ, sum((max(s.ctx, typ) for s in self.running.values())))

    def ctx_room(self):
        if not self.a.ctx_frac or len(self.running) < self.a.min_running:
            return True
        if not self.kv_tokens:
            return time.time() - self.T0 > 180
        typ, proj = self.ctx_projection()
        return proj + typ <= self.a.ctx_frac * self.kv_tokens

    def pick_toolfirst(self):
        a = self.a
        free = self.free_budget()
        tool_pend = [t for t in self.tasks if t.tier(a) is not None]
        d1_pend = [(t, k) for t in self.tasks for k in t.direct_todo]
        future = sum((t.n_test for t in tool_pend)) + sum((t.n_test for t in self.tasks if t.tool and t.running))
        units = len(tool_pend) + len(d1_pend) + a.tool_yield * future
        share = free / max(1.0, units)
        best = None
        for t in tool_pend:
            kk = (t.tier(a), t.order)
            if best is None or kk < best[0]:
                best = (kk, t, int(max(a.min_cap, min(a.sample_cap, share - a.force_tokens * t.n_test))), 'repl', None)
        for t, k in d1_pend:
            kk = (0 if not t.tool else 2, t.order, k)
            if best is None or kk < best[0]:
                best = (kk, t, int(max(a.direct_min_cap, min(a.direct_cap, share - a.direct_force_tokens))), 'direct', k)
        if best is not None:
            return best[1:]
        reserve = a.tool_yield * future * a.direct_min_cap
        cap2 = int(min(a.direct_cap2, free - reserve - a.direct_force_tokens))
        if cap2 < a.direct_min_cap2:
            return None
        cands = []
        for t in self.tasks:
            if t.direct_todo or (t.running and t.tool) or len(t.d2_launched) == t.n_test:
                continue
            r = self.ranked(t) if t.recs or t.running else None
            for k in range(t.n_test):
                ver = bool(r) and r[k].get('from', [])[:1] == ['ver']
                if k in t.d2_launched or (t.direct[k] is None and (not ver)):
                    continue
                nat = bool(t.d1_natural[k]) if t.direct[k] is not None else False
                cands.append(((nat, t.d1_ct[k] or 0, t.order, k), t, k))
        if not cands:
            return None
        _, t, k = min(cands, key=lambda x: x[0])
        return (t, cap2, 'direct2', k)

    def pick(self):
        a = self.a
        if a.select in ('toolfirst', 'route'):
            return self.pick_toolfirst()
        free = self.free_budget()
        pend0 = [t for t in self.tasks if t.launched == 0] + [(t, k) for t in self.tasks for k in t.direct_todo]
        best = None
        for t in self.tasks:
            if t.direct_todo:
                share = free / max(1, len(pend0))
                cap = int(max(a.direct_min_cap, min(a.direct_cap, share - a.direct_force_tokens)))
                kk = (0, t.launched + t.n_test - len(t.direct_todo), t.inflight, t.order)
                if best is None or kk < best[0]:
                    best = (kk, t, cap, 'direct', t.direct_todo[0])
            tr = t.tier(a)
            if tr is None or (tr > 0 and t.inflight >= a.max_inflight_per_task):
                continue
            ft = a.force_tokens * t.n_test
            tc = self.time_cap()
            if a.drain_fill == 'on' and tc is not None:
                if self.cut_time() - time.time() < a.drain_min_s:
                    if not self._drain_closed:
                        self._drain_closed = True
                        self.log(f'drain: no new sessions, cut in {self.cut_time() - time.time():.0f}s (guard {self.DEADLINE - self.cut_time():.0f}s)')
                    continue
                tcap = a.task_cap_per_input * t.n_test - t.used - t.reserved
                cap = int(min(a.sample_cap, tcap - ft))
                if cap < a.tail_min_cap:
                    continue
            else:
                if tc is not None and tc < a.tail_min_cap and (tr > 0):
                    continue
                min_cap = a.min_cap if tc is None else min(a.min_cap, max(tc, a.tail_min_cap))
                if tr == 0:
                    share = free / max(1, len(pend0))
                    cap = int(max(min_cap, min(a.sample_cap, share - ft)))
                    if tc is not None:
                        cap = min(cap, max(tc, a.tail_min_cap))
                else:
                    tcap = a.task_cap_per_input * t.n_test - t.used - t.reserved
                    if tc is None:
                        cap = int(min(a.sample_cap, tcap - ft, free - ft))
                    else:
                        cap = int(min(a.sample_cap, tcap - ft, tc))
                    if cap < min_cap:
                        continue
            kk = (tr, t.launched + t.n_test - len(t.direct_todo), t.inflight, t.order)
            if best is None or kk < best[0]:
                best = (kk, t, cap, 'repl', None)
        return None if best is None else best[1:]

    def endpoint(self, c, smp):
        if smp.ep is None:
            smp.ep = (_SessionEndpoint if self.a.session_ep == 'on' else _Endpoint)(self, c, smp)
        return smp.ep

    async def one(self, c, t, smp):
        CUR.set(smp)
        a = self.a
        self.n_seed += 1
        seed = self.n_seed
        rec, text, err = (None, '', None)
        try:
            task = t.task
            if a.carry == 'on':
                note = t.carry_note()
                if note:
                    task = dict(t.task, _carry=note)
                    smp.carried = True
            rec, text = await self.sess.run_sample(_SessCfg(a, t), self.endpoint(c, smp), self.tok, task, smp.cap, seed=seed)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = repr(e)[:300]
        if rec is None:
            rec = {'used': smp.progress(), 'status': 'error', 'forced': False, 'progs': list(smp.shadow), 'answer': [], 'n_tool': smp.n_exec, 'error': err}
        elif rec.get('status') in ('cap', 'ctx', 'answer'):
            try:
                await self.conclude(c, t, smp, rec, text)
            except Exception as e:
                self.log('conclude error', t.tid, repr(e)[:200])
        if (a.consistency_flags == 'on' or a.tie_break == 'flags') and smp.ver_code and any((p.get('verified') for p in rec.get('progs') or [])) and (time.time() < self.LAST_CALL):
            try:
                async with self.gate:
                    rec['aux'] = await aux_check(smp.ver_code, t.task)
            except Exception as e:
                rec['aux'] = {'error': repr(e)[:120]}
        if a.traces_out:
            try:
                import gzip
                with gzip.open(a.traces_out, 'at') as f:
                    f.write(json.dumps({'task': t.tid, 'idx': smp.idx, 'status': rec.get('status'), 'text': text}) + '\n')
            except Exception:
                pass
        if smp.ep is not None:
            try:
                await smp.ep.close()
            except Exception:
                pass
        rec.update(fmt_rec=smp.fmt_rec, fmt_fail=smp.fmt_fail, n_carry=smp.n_carry, n_timeout=smp.n_timeout)
        if smp.ver_code is not None:
            rec['ver_nif'] = getattr(smp, 'ver_nif', 0)
        rec.update(cut=smp.cut, n_exec=smp.n_exec, exec_s=round(smp.exec_s, 2), wait_s=round(smp.wait_s, 2), n_retry=smp.n_retry, wall_s=round(time.time() - smp.t0, 1), cap=smp.cap.cap, seed=seed)
        return rec

    async def conclude(self, c, t, smp, rec, text):
        a, F = (self.a, self.fmt)
        if any((p.get('verified') for p in rec.get('progs') or [])) or rec.get('answer') or rec.get('forced'):
            return
        prompt = self.render(t.task)
        ctx = prompt + text
        ts = ctx.rfind(F['turn_open'])
        ts = len(prompt) if ts < len(prompt) else ts + len(F['turn_open'])
        if F['nudge'] and ctx.endswith(F['nudge']):
            ctx = ctx[:-len(F['nudge'])]
        for e in ('<|im_end|>', '<|endofassistant|>', '<|endoftext|>'):
            ctx = ctx.removesuffix(e)
        seg = ctx[ts:]
        st = F['state'](seg)
        code = None
        if a.exec_on_cut and time.time() < self.LAST_CALL:
            if st == 'call' or F['call_open'] in seg:
                if F['call_end'] not in seg[seg.rfind(F['call_open']):]:
                    ctx = ctx + F['close']
                code = F['extract'](ctx[ts:])
                if code:
                    code = re.sub('(\\n\\s*</[^\\n]*)+\\s*$', '', code.rstrip()) or None
            else:
                m = re.findall('```python\\n(.*?)(?:```|$)', seg, re.S)
                code = m[-1].strip() if m and 'def transform' in m[-1] else None
            res = None
            if code and code != smp.last_code:
                rec['salvaged'] = True
                res = await self.run_code_tracked(code, t.task, a.exec_timeout)
                p = smp.shadow[-1] if smp.shadow and smp.shadow[-1]['k'] == smp.n_exec else None
                if p is not None:
                    rec['progs'] = list(rec.get('progs') or []) + [dict(p, k=10 ** 6)]
                    if p['verified']:
                        return
            st = F['state'](ctx[ts:])
            if st == 'call' and F['call_end'] in ctx[ts:][ctx[ts:].rfind(F['call_open']):]:
                resp = sandbox.feedback(res, t.task, a.exec_timeout) if res is not None else '(not run: time is up)'
                ctx = ctx + F['tool_resp'](resp)
                st = 'think'
        ftok = _SessCfg(a, t).force_tokens
        if ftok <= 0 or st == 'call' or time.time() > self.LAST_CALL:
            return
        add = F['force'] if st == 'think' else F['force_after']
        body = {'prompt': ctx + add, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': ftok * t.n_test}
        if F['force_stop']:
            body['stop'] = F['force_stop']
        out, _, u = await self.endpoint(c, smp).complete(body)
        rec['used'] = rec.get('used', 0) + u.get('completion_tokens', 0)
        g = '```json\n' + out
        if F['close_fence']:
            g = self.sess.close_fences(g) if hasattr(self.sess, 'close_fences') else g + ('\n```' if '```' not in out else '')
        rec['answer'] = F['parse'](g)
        rec['forced'] = True

    def direct_prompt(self, task, ti):
        ex = ''
        for i, p in enumerate(task['train']):
            ex += f'--Example {i}-- \n\n INPUT: \n\n' + json.dumps(p['input']) + '\n\n'
            ex += 'OUTPUT: \n\n' + json.dumps(p['output']) + '\n\n'
        body = self.template.format(training_examples=ex, test_input=json.dumps(task['test'][ti]['input']))
        return self.tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True)

    async def one_direct(self, c, t, smp):
        CUR.set(smp)
        a = self.a
        k = smp.k
        prompt = self.direct_prompt(t.task, k)
        rec = {'kind': 'direct', 'k': k, 'att': getattr(smp, 'att', 1), 'forced': False, 'grid': None, 'used': 0}
        try:
            text, fin, u = await self.endpoint(c, smp).complete({'prompt': prompt, 'max_tokens': smp.cap.cap, 'temperature': a.direct_temp, 'top_p': 1.0})
            rec['used'] = u.get('completion_tokens', 0)
            if '</think>' in text:
                rec['grid'] = _parse_json(text.rpartition('</think>')[2])
            elif time.time() < self.LAST_CALL:
                tail = text if len(text) <= a.direct_force_ctx_chars or a.session_ep == 'on' else '[... earlier reasoning omitted ...]\n' + text[-a.direct_force_ctx_chars:]
                out, _, u2 = await self.endpoint(c, smp).complete({'prompt': prompt + tail + DIRECT_FORCE, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': a.direct_force_tokens, 'stop': ['```']})
                rec['used'] += u2.get('completion_tokens', 0)
                rec['grid'] = _parse_partial('[[' + out)
                rec['forced'] = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            rec['error'] = repr(e)[:300]
        if smp.ep is not None:
            try:
                await smp.ep.close()
            except Exception:
                pass
        rec.update(cut=smp.cut, wall_s=round(time.time() - smp.t0, 1), cap=smp.cap.cap)
        return rec

    def finish_direct(self, t, smp, rec):
        t.inflight -= 1
        t.reserved -= smp.res
        t.used += rec.get('used', 0)
        k = smp.k
        att = rec.get('att', 1)
        if rec.get('grid') is not None:
            if att == 2:
                t.direct2[k] = rec['grid']
            else:
                t.direct[k] = rec['grid']
                t.d1_natural[k] = not rec.get('forced')
                t.d1_ct[k] = rec.get('used')
        elif rec.get('error') and (not smp.cut) and (time.time() < self.ENDGAME):
            if att == 2:
                t.d2_launched.discard(k)
            else:
                t.direct_todo.append(k)
        rec.update(task=t.tid, t_rel=round(time.time() - self.T0, 1))
        t.direct_recs.append(rec)
        gt = t.gt[k] if t.gt and k < len(t.gt) else None
        self.log('direct', t.tid, k, f'att{att}', f'cap={smp.cap.cap}', f'used={rec.get('used')}', 'forced' if rec.get('forced') else '', 'parsed' if rec.get('grid') is not None else 'NOGRID', ('correct' if rec.get('grid') == gt else 'wrong') if gt is not None and rec.get('grid') is not None else '', f'cut={smp.cut}' if smp.cut else '', f'{rec['wall_s']:.0f}s', f'err={rec.get('error')}' if rec.get('error') else '')
        if self.samples_f:
            self.samples_f.write(json.dumps(dict(rec, correct=rec.get('grid') == gt if gt is not None else None)) + '\n')
            self.samples_f.flush()
        self.write_submission()

    def select(self, t):
        r = self.ranked(t) if t.recs or t.running else [{'attempts': [], 'from': [], 'n_verified_votes': []}] * t.n_test
        out = []
        for k in range(t.n_test):
            att, frm = (list(r[k]['attempts']), list(r[k].get('from', [])))
            nv = list(r[k].get('n_verified_votes', []))
            d = t.direct[k]
            if self.a.select in ('toolfirst', 'route'):
                ver = att[0] if frm[:1] == ['ver'] else None
                s1, src1 = (ver, 'ver') if ver is not None else (d, 'd1') if d is not None else (None, '')
                s2, src2 = (None, '')
                for g, src in ((d, 'd1'), (t.direct2[k], 'd2')):
                    if g is not None and g != s1 and (s2 is None):
                        s2, src2 = (g, src)
                for g, src in zip(att, frm):
                    if s1 is None:
                        s1, src1 = (g, src)
                    elif s2 is None and g != s1:
                        s2, src2 = (g, src)
                out.append(([x for x in (s1, s2) if x is not None], f'{src1 or 'none'}+{src2 or 'none'}'))
                continue
            if self.a.select == 'agree' and d is not None:
                if frm[:1] == ['ver'] and nv and (nv[0] >= self.a.agree_n):
                    a2 = d if d != att[0] else att[1] if len(att) > 1 else att[0]
                    out.append(([att[0], a2], f'agree{nv[0]}+' + ('direct' if d != att[0] else 'same')))
                else:
                    rest = [g for g in att if g != d]
                    src = frm[att.index(rest[0])] if rest else ''
                    out.append(([d] + rest[:1], 'direct+' + (f'{src}1' if src == 'ver' else src or 'none')))
            else:
                if d is not None and d not in att and (len(att) < 2):
                    att, frm = (att + [d], frm + ['direct'])
                out.append((att, 'vote:' + '+'.join(frm[:2]) if att else 'none'))
        if self.a.att2_fill == 'on':
            out = [self.fill2(t, k, att, rule) for k, (att, rule) in enumerate(out)]
        return out

    def fill2(self, t, k, att, rule):
        a1 = att[0] if att else t.task['test'][k]['input']
        if any((g != a1 for g in att[1:2])):
            return (att, rule)
        recs = list(t.recs) + [{'progs': list(s.shadow)} for s in t.running if s.shadow]
        n, frac, grids = ({}, {}, {})
        for r in recs:
            seen = set()
            for p in r.get('progs') or []:
                ts = p.get('tests') or []
                g = ts[k] if k < len(ts) else None
                if not is_grid(g) or g == a1:
                    continue
                q = repl_vote.key(g)
                fr = 1.0 if p.get('verified') else (p.get('n_ok') or 0) / max(1, len(p.get('train_ok') or []))
                frac[q] = max(frac.get(q, 0.0), fr)
                grids.setdefault(q, g)
                if q not in seen:
                    seen.add(q)
                    n[q] = n.get(q, 0) + 1
        if not n:
            return (att, rule)
        q = min(n, key=lambda q: (-n[q], -frac[q], q))
        return ([a1, grids[q]], rule + '+fill')

    def demote(self, t, rec):
        if self.a.demote_flagged != 'on':
            return
        ver = [p for p in rec.get('progs') or [] if p.get('verified')]
        if not ver:
            return
        why = []
        if ARCKIT is not None:
            try:
                why = list(ARCKIT.flag(t.task['train'], [x['input'] for x in t.task['test']], ver[-1]['tests']))
            except Exception as e:
                self.log('arckit flag error', t.tid, repr(e)[:120])
        aux = rec.get('aux') or {}
        if self.a.consistency_flags == 'on':
            why += ['augmentation-inconsistent'] if aux.get('aug_flag') else []
            why += ['test-only coverage'] if aux.get('cov_flag') else []
        if why:
            for p in ver:
                p['verified'] = False
                p['demoted'] = True
            rec['demoted'] = why[:4]
            self.n_demoted += 1

    def ranked(self, t):
        recs = list(t.recs) + [{'progs': list(s.shadow), 'answer': [], 'ver_nif': getattr(s, 'ver_nif', 0)} for s in t.running if s.shadow]
        if self.a.tie_break in ('nif', 'flags'):
            return rank_tiebreak(recs, t.n_test, flags=self.a.tie_break == 'flags')
        return repl_vote.rank(recs, t.n_test)

    def write_submission(self):
        sub = {}
        last = getattr(self, '_last_sub', {})
        for t in self.tasks:
            try:
                sel = self.select(t)
                outs = []
                for k in range(t.n_test):
                    att = sel[k][0]
                    fb = t.task['test'][k]['input']
                    a1 = att[0] if att else fb
                    a2 = att[1] if len(att) > 1 else a1
                    outs.append({'attempt_1': a1, 'attempt_2': a2})
            except Exception as e:
                outs = last.get(t.tid)
                try:
                    self.log('selection error', t.tid, repr(e)[:160])
                except Exception:
                    pass
                if outs is None:
                    continue
            sub[t.tid] = outs
        self._last_sub = sub
        for tid in self.chal:
            sub.setdefault(tid, [{'attempt_1': x['input'], 'attempt_2': x['input']} for x in self.chal[tid]['test']])
        tmp = self.a.out + '.tmp'
        json.dump(sub, open(tmp, 'w'))
        Path(tmp).replace(self.a.out)
        if getattr(self, 'turn_f', None) is not None:
            self.turn_f.flush()

    def score(self):
        if not self.sols:
            return None
        sc = 0.0
        for t in self.tasks:
            gts = t.gt or []
            sel = self.select(t)
            hit = [any((g == gts[k] for g in sel[k][0][:2])) for k in range(t.n_test) if k < len(gts)]
            sc += sum(hit) / max(1, len(hit))
        return sc

    def finish(self, t, smp, rec):
        a = self.a
        t.running.discard(smp)
        t.inflight -= 1
        t.reserved -= smp.res
        t.used += rec.get('used', 0)
        if smp.ctx or rec.get('n_ctx'):
            self.final_ctx.append(max(smp.ctx, rec.get('n_ctx') or 0))
        rec.update(task=t.tid, idx=len(t.recs), t_rel=round(time.time() - self.T0, 1))
        if a.answer_order == 'shape' and rec.get('answer'):
            o = order_answers(t.task, rec['answer'])
            if o is not None:
                rec['answer_raw'], rec['answer'], rec['answer_reordered'] = (rec['answer'], o, True)
        self.demote(t, rec)
        if a.carry == 'on':
            rec['carried'] = bool(getattr(smp, 'carried', False))
            t.carry_add(smp.shadow)
        t.recs.append(rec)
        t._tier = None
        if a.select in ('toolfirst', 'route') and (not t.running):
            r = repl_vote.rank(t.recs, t.n_test)
            for k in range(t.n_test):
                if r[k].get('from', [])[:1] != ['ver'] and t.direct[k] is None and (k not in t.direct_todo):
                    t.direct_todo.append(k)
        ver = [p for p in rec.get('progs') or [] if p.get('verified')]
        corr = ''
        if t.gt:
            if ver:
                corr = 'ver_correct' if all((o == g for o, g in zip(ver[-1]['tests'], t.gt))) else 'ver_wrong'
            elif rec.get('answer'):
                corr = 'ans_correct' if rec['answer'][:t.n_test] == t.gt[:len(rec['answer'][:t.n_test])] else 'ans_wrong'
        self.log('done', t.tid, f's{rec['idx']}', f'cap={smp.cap.cap}', f'used={rec.get('used', 0)}', f'status={rec.get('status')}', f'tools={rec.get('n_tool', 0)}', f'exec={smp.n_exec}/{smp.exec_s:.1f}s', *([f'fmt={smp.fmt_rec}/{smp.fmt_rec + smp.fmt_fail}'] if smp.fmt_rec + smp.fmt_fail else []), 'verified' if ver else 'forced' if rec.get('forced') else 'answer' if rec.get('answer') else 'none', f'cut={smp.cut}' if smp.cut else '', f'{rec['wall_s']:.0f}s', corr, f'demoted={'|'.join(rec['demoted'])[:80]}' if rec.get('demoted') else '', f'err={rec.get('error')}' if rec.get('error') else '')
        if self.samples_f:
            row = {k: v for k, v in rec.items() if k not in ('code',)}
            row['progs'] = [{k: v for k, v in p.items() if k != 'code' or a.keep_code} for p in row.get('progs') or []]
            if t.gt:
                for p in row['progs']:
                    p['tests_correct'] = [o is not None and o == g for o, g in zip(p['tests'], t.gt)]
                row['answer_correct'] = [g == t.gt[k] for k, g in enumerate((row.get('answer') or [])[:t.n_test]) if k < len(t.gt)]
            self.samples_f.write(json.dumps(row) + '\n')
            self.samples_f.flush()
        self.write_submission()

    def endgame(self):
        n = 0
        for smp in self.running.values():
            if not smp.cut:
                smp.cut = 'endgame'
                smp.cap.limit = -(1 << 40)
                n += 1
        if self.a.endgame_flag:
            try:
                Path(self.a.endgame_flag).touch()
            except OSError:
                pass
        if self.a.drain_fill == 'on':
            self.log('endgame: cut', n, f'running sessions; no new sessions (drain: {self.DEADLINE - time.time():.0f}s before the deadline)')
        else:
            self.log('endgame: cut', n, 'running sessions; no new sessions')

    def status(self):
        r = self.rate()
        m = self.metrics
        return f'rate {('-' if r is None else f'{r:.0f}')} tok/s, free {self.free_budget() / 1000000.0:.2f}M, running {len(self.running)}, kv {self.kv:.2f}, launched {sum((t.launched for t in self.tasks))}, done {sum((len(t.recs) for t in self.tasks))}, server {('ok' if self.server_ok else 'DOWN')}, tool calls {len(self.tool_stats)}' + (f', preempt {m.get('preempt', 0):.0f}' if m else '') + (f', ctx proj {self.ctx_projection()[1] / 1000.0:.0f}k / kv {self.kv_tokens / 1000.0:.0f}k (typ {self.ctx_projection()[0] / 1000.0:.0f}k)' if self.kv_tokens else '')

    async def run(self):
        import httpx
        a = self.a
        self.load()
        nw = harden_sandbox(a)
        self.gate = asyncio.Semaphore(nw)
        self.load_backend()
        self.write_submission()
        self.log(f'backend {a.backend}: {len(self.tasks)} tasks, {self.n_inputs} test inputs, deadline {a.deadline_s:.0f}s, endgame {a.endgame_s:.0f}s, margin {a.budget_margin_s:.0f}s, concurrency {a.concurrency}, sandbox workers {nw}, sample cap {a.sample_cap}, min cap {a.min_cap}')
        await sandbox_selftest(self.log)
        limits = httpx.Limits(max_connections=a.concurrency + 16, max_keepalive_connections=0)
        async with httpx.AsyncClient(timeout=httpx.Timeout(a.deadline_s), limits=limits) as c:
            bg = [asyncio.create_task(self.metrics_loop(c)), asyncio.create_task(self.health_loop(c))]
            last_status = last_write = 0.0
            while time.time() < self.DEADLINE - 60:
                now = time.time()
                if now >= self.cut_time() and (not self.endgame_done):
                    self.endgame_done = True
                    self.endgame()
                while not self.endgame_done and len(self.running) < a.concurrency and self.server_ok and (self.kv < a.kv_admit or len(self.running) < a.min_running) and self.ctx_room():
                    p = self.pick()
                    if p is None:
                        break
                    t, cap, kind, k = p
                    if kind in ('direct', 'direct2'):
                        if kind == 'direct':
                            t.direct_todo.remove(k)
                        else:
                            t.d2_launched.add(k)
                        res = cap + a.direct_force_tokens
                        smp = Sample(t, -1 - k, cap, res, 'direct', k)
                        smp.att = 2 if kind == 'direct2' else 1
                        t.inflight += 1
                        t.reserved += res
                        task = asyncio.create_task(self.one_direct(c, t, smp))
                    else:
                        res = cap + a.force_tokens * t.n_test
                        smp = Sample(t, t.launched, cap, res)
                        t.launched += 1
                        t.inflight += 1
                        t.reserved += res
                        t.running.add(smp)
                        t._tier = None
                        task = asyncio.create_task(self.one(c, t, smp))
                    self.running[task] = smp
                if not self.running and (self.endgame_done or (all((x.tier(a) is None and (not x.direct_todo) for x in self.tasks)) and self.pick() is None)):
                    if self.server_ok or self.endgame_done:
                        break
                if self.running:
                    done, _ = await asyncio.wait(list(self.running), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                else:
                    done = set()
                    await asyncio.sleep(5)
                for d in done:
                    smp = self.running.pop(d)
                    try:
                        rec = d.result()
                    except Exception as e:
                        rec = {'used': smp.progress(), 'status': 'error', 'progs': list(smp.shadow), 'answer': [], 'error': repr(e)[:300], 'wall_s': time.time() - smp.t0}
                    (self.finish_direct if smp.kind == 'direct' else self.finish)(smp.task, smp, rec)
                if time.time() - last_write > 60:
                    last_write = time.time()
                    self.write_submission()
                if time.time() - last_status > 300:
                    last_status = time.time()
                    self.log(self.status())
            if a.drain_fill == 'on' and self.running:
                self.log('drain: deadline reached with', len(self.running), 'sessions still concluding: their programs are kept')
                for d, smp in list(self.running.items()):
                    if smp.kind != 'repl':
                        continue
                    self.running.pop(d)
                    rec = None
                    if d.done() and not d.cancelled():
                        try:
                            rec = d.result()
                        except Exception as e:
                            rec = {'used': smp.progress(), 'status': 'error', 'progs': list(smp.shadow), 'answer': [], 'error': repr(e)[:300], 'wall_s': time.time() - smp.t0}
                    else:
                        d.cancel()
                        smp.cut = smp.cut or 'deadline'
                        rec = {'used': smp.progress(), 'status': 'cap', 'forced': False, 'progs': list(smp.shadow), 'answer': [], 'cut': 'deadline', 'wall_s': time.time() - smp.t0}
                    self.finish(smp.task, smp, rec)
            for d in list(self.running):
                d.cancel()
            for b in bg:
                b.cancel()
        self.write_submission()
        self.summary()

    def summary(self):
        n_ans = sum((1 for t in self.tasks for k in range(t.n_test) if self.ranked(t)[k]['attempts'])) if self.tasks else 0
        recs = [r for t in self.tasks for r in t.recs]
        self.log(f'final: {n_ans}/{self.n_inputs} test inputs answered, {len(recs)} sessions ({sum((1 for r in recs if any((p.get('verified') for p in r.get('progs') or []))))} verified, {sum((1 for r in recs if r.get('forced')))} forced, {sum((1 for r in recs if r.get('cut')))} cut, {sum((1 for r in recs if r.get('status') == 'error'))} errors)')
        if self.a.demote_flagged == 'on':
            self.log(f"demote: {self.n_demoted} sessions' verified programs demoted by the content flag" + ('' if ARCKIT is not None else ' (arckit.py not found: demotion OFF)'))
        if self.a.fn_fixes == 'on':
            self.log(f'harness fixes: {self.n_timeouts} exec timeouts ({self.a.exec_timeout:.0f} s) in {len(self.tool_stats)} executions; {sum((r.get('n_carry', 0) for r in recs))} cells ran with earlier defs carried; {sum((r.get('n_nudge', 0) for r in recs))} turn nudges')
        if self.a.tolerant_calls == 'on':
            self.log(f'tool-call format: {sum((r.get('fmt_rec', 0) for r in recs))} malformed calls recovered, {sum((r.get('fmt_fail', 0) for r in recs))} unrecoverable, in {sum((1 for r in recs if r.get('fmt_rec', 0) + r.get('fmt_fail', 0)))} sessions; {sum((r.get('n_tool', 0) for r in recs))} tool calls')
        if self.metrics0 and self.metrics:
            m0, m1 = (self.metrics0, self.metrics)
            dt = max(1.0, m1['t'] - m0['t'])
            hit = (m1['hits'] - m0['hits']) / max(1.0, m1['queries'] - m0['queries'])
            self.log(f'server: generated {m1['gen'] - m0['gen']:.0f} tokens in {dt:.0f}s = {(m1['gen'] - m0['gen']) / dt:.0f} tok/s; prompt tokens {m1['prompt'] - m0['prompt']:.0f} (prefix-cache hit rate {hit:.2f}); preemptions {m1['preempt'] - m0['preempt']:.0f}')
        ex = sorted((x[0] for x in self.tool_stats))
        wt = sorted((x[1] for x in self.tool_stats))
        if ex:
            wall = sum((r.get('wall_s', 0) for r in recs))
            exs = sum((r.get('exec_s', 0) for r in recs))
            self.log(f'tools: {len(ex)} executions, mean {sum(ex) / len(ex):.2f}s, p50 {ex[len(ex) // 2]:.2f}s, p90 {ex[int(len(ex) * 0.9)]:.2f}s, max {ex[-1]:.2f}s; slot wait mean {sum(wt) / len(wt):.2f}s p90 {wt[int(len(wt) * 0.9)]:.2f}s; tool time / session wall {exs / max(1e-09, wall):.3f}; gen tokens per session {sum((r.get('used', 0) for r in recs)) / max(1, len(recs)):.0f}')
        from collections import Counter
        rules, rule_ok = (Counter(), Counter())
        sel_rows = {}
        for t in self.tasks:
            sel = self.select(t)
            sel_rows[t.tid] = []
            for k, (att, rule) in enumerate(sel):
                ok = None
                if t.gt and k < len(t.gt):
                    ok = [g == t.gt[k] for g in att[:2]]
                    rule_ok[rule] += any(ok)
                rules[rule] += 1
                sel_rows[t.tid].append({'rule': rule, 'correct': ok, 'direct': t.direct[k] is not None})
        self.log('selection (' + self.a.select + '):', ', '.join((f'{r} {n}' + (f' ({rule_ok[r]} right)' if self.sols else '') for r, n in rules.most_common())))
        if self.a.log:
            json.dump(sel_rows, open(Path(self.a.log).with_name('repl_selection.json'), 'w'))
        if any((t.direct_recs for t in self.tasks)):
            dr = [r for t in self.tasks for r in t.direct_recs]
            self.log(f'direct: {len(dr)} attempts, {sum((r.get('grid') is not None for r in dr))} parsed, {sum((bool(r.get('forced')) for r in dr))} forced, tokens {sum((r.get('used', 0) for r in dr))}')
        sc = self.score()
        if sc is not None:
            self.log(f'local score (pass@2): {sc:.2f}/{len(self.tasks)} = {100 * sc / max(1, len(self.tasks)):.1f}%')

class _Enc:
    __slots__ = ('input_ids',)

    def __init__(self, ids):
        self.input_ids = ids

class TokCache:

    def __init__(self, tok, max_chars=64 << 20):
        import collections
        self._tok, self._max = (tok, max_chars)
        self._ids = collections.OrderedDict()
        self._tpl = collections.OrderedDict()
        self._chars = 0
        self.hits = self.misses = 0

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def _put(self, d, key, val, n):
        d[key] = val
        self._chars += n
        while self._chars > self._max and (self._ids or self._tpl):
            dd = self._ids if len(self._ids) >= len(self._tpl) else self._tpl
            k, v = dd.popitem(last=False)
            self._chars -= len(k[0]) if dd is self._ids else len(v)

    def __call__(self, text, *args, add_special_tokens=True, **kw):
        if args or kw or (not isinstance(text, str)):
            return self._tok(text, *args, add_special_tokens=add_special_tokens, **kw)
        key = (text, bool(add_special_tokens))
        ids = self._ids.get(key)
        if ids is None:
            self.misses += 1
            ids = tuple(self._tok(text, add_special_tokens=add_special_tokens).input_ids)
            self._put(self._ids, key, ids, len(text))
        else:
            self.hits += 1
            self._ids.move_to_end(key)
        return _Enc(list(ids))

    def apply_chat_template(self, conversation, *args, **kw):
        if args or kw.get('tokenize', True):
            return self._tok.apply_chat_template(conversation, *args, **kw)
        try:
            key = json.dumps([conversation, kw], sort_keys=True, default=str)
        except (TypeError, ValueError):
            return self._tok.apply_chat_template(conversation, **kw)
        out = self._tpl.get(key)
        if out is None:
            out = self._tok.apply_chat_template(conversation, **kw)
            self._put(self._tpl, key, out, len(out))
        else:
            self._tpl.move_to_end(key)
        return out

class _SessionEndpoint:

    def __init__(self, S, c, smp):
        self.S, self.c, self.smp = (S, c, smp)
        self.rid, self.lines, self.resp_cm = (None, None, None)
        self.known, self.last_id = ('', None)
        self.n_sess = 0
        self.n_held = 0
        self.t_out = None
        self.exec_out = 0.0
    _FENCE = None

    def _stop_ids(self, body):
        tok = self.S.tok
        ids = set(body.get('stop_token_ids') or [])
        for st in body.get('stop') or []:
            if self._one_token(st) is not None:
                ids.add(self._one_token(st))
        return sorted(ids)

    def _one_token(self, st):
        if st.startswith('<|') and st.endswith('|>'):
            return self.S.tok(st, add_special_tokens=False).input_ids[-1]
        if st.startswith('<') and st.endswith('>'):
            ids = self.S.tok(st, add_special_tokens=False).input_ids
            if len(ids) == 1 and self.S.tok.decode(ids) == st:
                return ids[0]
        return None

    def _turn_log(self, tl, n_prompt, n_out, fin):
        S, smp = (self.S, self.smp)
        t1 = time.time()
        ms = lambda x: None if x is None else round((x - tl['t_in']) * 1000.0, 2)
        row = {'s': f'{smp.task.tid}_{smp.idx}', 't': round(tl['t_in'] - S.T0, 2), 'kind': tl.get('kind'), 'n_in': tl.get('n_in'), 'ctx': n_prompt, 'n_out': n_out, 'fin': fin, 'gap_ms': None if self.t_out is None else round((tl['t_in'] - self.t_out) * 1000.0, 2), 'exec_ms': round((smp.exec_s - self.exec_out) * 1000.0, 2) if self.t_out is not None else None, 'tok_ms': ms(tl.get('t_tok')), 'req_ms': ms(tl.get('t_req')), 'first_ms': ms(tl.get('t_first')), 'end_ms': ms(t1), 'n_ce': tl.get('n_ce'), 'ce_wait_ms': round(tl.get('ce_wait', 0) * 1000.0, 2), 'ce_rtt_ms': round(tl.get('ce_rtt', 0) * 1000.0, 2)}
        self.t_out, self.exec_out = (t1, smp.exec_s)
        try:
            S.turn_f.write(json.dumps(row) + '\n')
        except Exception:
            pass

    async def close(self):
        if self.rid is not None:
            try:
                await self.c.post(self.S.base + '/arc/session/close', json={'rid': self.rid}, timeout=10)
            except Exception:
                pass
        if self.resp_cm is not None:
            try:
                await self.resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self.rid, self.lines, self.resp_cm, self.known, self.last_id, self.n_held = (None, None, None, '', None, 0)

    async def complete(self, body):
        S, smp = (self.S, self.smp)
        cuttable = not smp.cut
        while True:
            await S.wait_server()
            if time.time() > S.DEADLINE - 60:
                raise ServerError('deadline')
            try:
                return await self._turn(body, cuttable)
            except ServerError:
                await self.close()
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self.close()
                smp.n_retry += 1
                smp.live = 0
                try:
                    healthy = (await self.c.get(S.base + '/health', timeout=5)).status_code == 200
                except Exception:
                    healthy = False
                if healthy:
                    smp.req_fail += 1
                    S.log('session error', smp.task.tid, f's{smp.idx}', repr(e)[:160], f'(server healthy: retry {smp.req_fail}/3)')
                    if smp.req_fail >= 3:
                        raise ServerError(f'session failed 3 times with a healthy server: {e!r}'[:300])
                    await asyncio.sleep(2)
                    continue
                S.log('session error', smp.task.tid, f's{smp.idx}', repr(e)[:160], '-> server down: wait, restart session')
                S.server_ok = False
                await asyncio.sleep(5)

    async def _turn(self, body, cuttable):
        S, smp, tok = (self.S, self.smp, self.S.tok)
        prompt, max_tok = (body['prompt'], int(body['max_tokens']))
        samp = {'temperature': body.get('temperature', 1.0), 'top_p': body.get('top_p', 1.0)}
        if body.get('top_k') is not None:
            samp['top_k'] = body['top_k']
        stop = self._stop_ids(body)
        text_stops = [x for x in body.get('stop') or [] if self._one_token(x) is None]
        M = int(S.a.max_model_len)
        tl = {'t_in': time.time()} if S.turn_f is not None else None
        extend = self.rid is not None and prompt.startswith(self.known) and (self.last_id is not None)
        if extend:
            ids = [self.last_id] + tok(prompt[len(self.known):], add_special_tokens=False).input_ids
            n_after = self.n_held + len(ids) - 1
        else:
            pids = tok(prompt, add_special_tokens=False).input_ids
            n_after = len(pids)
        if tl is not None:
            tl.update(t_tok=time.time(), kind='push' if extend else 'start', n_in=len(ids) if extend else len(pids))
        if n_after >= M - 1:
            smp.n_ctx_full = getattr(smp, 'n_ctx_full', 0) + 1
            await self.close()
            return ('', 'length', {'prompt_tokens': max(n_after, M), 'completion_tokens': 0})
        max_tok = min(max_tok, M - n_after)
        chunk = max(1, min(64 if text_stops else S.a.session_chunk, max_tok))
        if extend:
            r = await self.c.post(S.base + '/arc/session/push', json={'rid': self.rid, 'prompt_ids': ids, 'max_tokens': chunk, 'stop_token_ids': stop, 'keep': True, **samp}, timeout=30)
            if r.status_code == 409:
                smp.n_ctx_full = getattr(smp, 'n_ctx_full', 0) + 1
                await self.close()
                return ('', 'length', {'prompt_tokens': max(n_after, M), 'completion_tokens': 0})
            if r.status_code != 200:
                raise RuntimeError(f'push HTTP {r.status_code}')
            self.n_held = n_after
        else:
            await self.close()
            self.n_sess += 1
            self.rid = f'{smp.task.tid}_{smp.idx}_{self.n_sess}_{time.time():.3f}'
            self.n_held = n_after
            b0 = {'rid': self.rid, 'prompt_ids': pids, 'max_tokens': 1 if stop else chunk, 'keep': True, **samp}
            if stop and S.a.start_stops == 'on':
                b0.update(max_tokens=chunk, stop_token_ids=stop)
            if body.get('seed') is not None:
                b0['seed'] = body['seed']
            self.resp_cm = self.c.stream('POST', S.base + '/arc/session', json=b0, timeout=max(30.0, S.DEADLINE - 30 - time.time()))
            resp = await self.resp_cm.__aenter__()
            if resp.status_code != 200:
                txt = (await resp.aread()).decode(errors='replace')[:300]
                raise (ServerError if 400 <= resp.status_code < 500 else RuntimeError)(f'session HTTP {resp.status_code}: {txt}')
            self.lines = resp.aiter_lines().__aiter__()
        if tl is not None:
            tl.update(t_req=time.time(), n_ce=0, ce_wait=0.0, ce_rtt=0.0, t_first=None, t_ce=None)
        parts, ids_out, fin, full = ([], [], None, False)
        n_prompt = self.n_held
        smp.live = 0
        while True:
            line = await self.lines.__anext__()
            if not line:
                continue
            ev = json.loads(line)
            if tl is not None and 'ids' in ev:
                if tl['t_first'] is None:
                    tl['t_first'] = time.time()
                if tl['t_ce'] is not None:
                    tl['ce_wait'] += time.time() - tl['t_ce']
                    tl['t_ce'] = None
            if 'ids' in ev:
                ids_out += ev['ids']
                parts.append(ev['text'])
                smp.live = len(ids_out)
                smp.ctx = n_prompt + len(ids_out)
            elif 'chunk_end' in ev:
                if ev['chunk_end'] != 'length':
                    fin = 'stop'
                    break
                if text_stops and any((x in ''.join(parts) for x in text_stops)):
                    fin = 'stop'
                    break
                if ev.get('ctx_full'):
                    fin, full = ('length', True)
                    break
                if len(ids_out) >= max_tok or (cuttable and smp.cut):
                    fin = 'length'
                    break
                if tl is not None:
                    tl['t_ce'] = t_ce = time.time()
                    tl['n_ce'] += 1
                r = await self.c.post(S.base + '/arc/session/push', json={'rid': self.rid, 'prompt_ids': [ids_out[-1]], 'max_tokens': min(chunk, max_tok - len(ids_out)), 'stop_token_ids': stop, 'keep': True, **samp}, timeout=30)
                if tl is not None:
                    tl['ce_rtt'] += time.time() - t_ce
                if r.status_code == 409:
                    fin, full = ('length', True)
                    break
                if r.status_code != 200:
                    raise RuntimeError(f'push HTTP {r.status_code}')
            elif 'end' in ev:
                raise RuntimeError('session ended unexpectedly')
        text = ''.join(parts)
        if fin == 'stop' and ids_out and (ids_out[-1] in stop) and body.get('include_stop_str_in_output') and (ids_out[-1] not in getattr(self.S, 'eos_ids', ())):
            st = tok.decode(ids_out[-1:])
            if not text.endswith(st):
                text += st
        self.n_held = n_prompt + len(ids_out)
        if tl is not None:
            self._turn_log(tl, n_prompt, len(ids_out), fin)
        self.known = prompt + text
        self.last_id = ids_out[-1] if ids_out else self.last_id
        if not ids_out:
            self.last_id = None
        smp.used_done += len(ids_out)
        smp.live = 0
        for st in body.get('stop') or []:
            i = text.find(st)
            if i >= 0:
                text = text[:i + (len(st) if body.get('include_stop_str_in_output') else 0)]
        if full or (fin == 'length' and n_prompt + len(ids_out) >= M):
            smp.n_ctx_full = getattr(smp, 'n_ctx_full', 0) + 1
            await self.close()
            return (text, 'length', {'prompt_tokens': max(n_prompt, M - len(ids_out)), 'completion_tokens': len(ids_out)})
        return (text, fin, {'prompt_tokens': n_prompt, 'completion_tokens': len(ids_out)})

class _Endpoint:

    def __init__(self, S, c, smp):
        self.S, self.c, self.smp = (S, c, smp)

    async def close(self):
        pass

    async def complete(self, body):
        S, smp = (self.S, self.smp)
        cuttable = not smp.cut
        body = {'model': 'm', 'skip_special_tokens': False, **body, 'stream': True, 'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        while True:
            await S.wait_server()
            if time.time() > S.DEADLINE - 60:
                raise ServerError('deadline')
            try:
                return await self._stream(body, cuttable)
            except ServerError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                smp.n_retry += 1
                smp.live = 0
                try:
                    healthy = (await self.c.get(S.base + '/health', timeout=5)).status_code == 200
                except Exception:
                    healthy = False
                if healthy:
                    smp.req_fail += 1
                    S.log('request error', smp.task.tid, f's{smp.idx}', repr(e)[:160], f'(server healthy: retry {smp.req_fail}/3)')
                    if smp.req_fail >= 3:
                        raise ServerError(f'request failed 3 times with a healthy server: {e!r}'[:300])
                    await asyncio.sleep(2)
                    continue
                S.log('request error', smp.task.tid, f's{smp.idx}', repr(e)[:160], '-> server down: wait, re-send')
                S.server_ok = False
                await asyncio.sleep(5)

    async def _stream(self, body, cuttable):
        S, smp = (self.S, self.smp)
        parts, fin, usage = ([], None, {'prompt_tokens': 0, 'completion_tokens': 0})
        smp.live = 0
        timeout = max(30.0, S.DEADLINE - 30 - time.time())
        async with self.c.stream('POST', S.base + '/v1/completions', json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                txt = (await resp.aread()).decode(errors='replace')[:300]
                if 400 <= resp.status_code < 500:
                    raise ServerError(f'HTTP {resp.status_code}: {txt}')
                raise RuntimeError(f'HTTP {resp.status_code}: {txt}')
            async for line in resp.aiter_lines():
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                ev = json.loads(line[6:])
                if ev.get('error'):
                    raise RuntimeError(f'stream error {str(ev['error'])[:200]}')
                if ev.get('usage'):
                    usage = ev['usage']
                    smp.live = usage.get('completion_tokens', 0)
                    smp.ctx = usage.get('prompt_tokens', 0) + smp.live
                for ch in ev.get('choices') or []:
                    parts.append(ch.get('text') or '')
                    fin = ch.get('finish_reason') or fin
                if cuttable and smp.cut:
                    fin = 'length'
                    break
        smp.used_done += usage.get('completion_tokens', 0)
        smp.live = 0
        return (''.join(parts), fin or 'length', usage)
DIRECT_FORCE = '\n\nTime is up; I will commit to my best hypothesis.\n</think>\n\nFinal answer:\n```json\n[['

def _parse_json(ans):
    m = re.search('\\\\boxed\\{(.*?)\\}', ans, re.DOTALL)
    for cand in ([m.group(1)] if m else []) + [ans]:
        end = cand.rfind(']')
        depth, start = (0, -1)
        for i in range(end, -1, -1):
            depth += cand[i] == ']'
            depth -= cand[i] == '['
            if depth == 0:
                start = i
                break
        if start >= 0:
            try:
                g = json.loads(cand[start:end + 1])
                if isinstance(g, list) and g and all((isinstance(r, list) and r and all((isinstance(v, int) for v in r)) for r in g)):
                    return g
            except Exception:
                pass
    return None

def _parse_partial(txt):
    g = _parse_json(txt)
    if g is not None:
        return g
    rows = re.findall('\\[\\s*(-?\\d+(?:\\s*,\\s*-?\\d+)*)\\s*\\]', txt)
    rows = [[int(v) for v in r.split(',')] for r in rows]
    if not rows:
        return None
    w = len(rows[0])
    out = []
    for r in rows:
        if len(r) != w:
            break
        out.append(r)
    return out if out and len(out) <= 30 and (w <= 30) else None

def main(argv):
    a = parse_args(argv)
    S = Solver(a)
    try:
        asyncio.run(S.run())
    finally:
        try:
            S.write_submission()
        except Exception:
            pass
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
