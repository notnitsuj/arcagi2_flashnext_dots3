import asyncio, json, os, resource, shutil, signal, subprocess, sys, tempfile
PY = sys.executable
UNSHARE = shutil.which('unshare')
if UNSHARE:
    try:
        if subprocess.run([UNSHARE, '-rn', 'true'], capture_output=True, timeout=10).returncode != 0:
            UNSHARE = None
    except Exception:
        UNSHARE = None
OUT_LIMIT = 2500
ARCKIT = os.environ.get('QR_ARCKIT', '')
ARCKIT_PATH = os.environ.get('QR_ARCKIT_PATH') or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'kaggle', 'code_dataset', 'arckit.py'))
if ARCKIT not in ('', 'lib', 'block', 'review'):
    raise SystemExit(f'QR_ARCKIT={ARCKIT!r}: expected lib, block or review')
if ARCKIT and (not os.path.isfile(ARCKIT_PATH)):
    raise SystemExit(f'QR_ARCKIT={ARCKIT} but the helper module is missing: {ARCKIT_PATH}')
H2_FLAGFB = 'flagfb' in os.environ.get('QR_H2', '').split(',')
if H2_FLAGFB and (not ARCKIT):
    raise SystemExit('QR_H2=flagfb needs QR_ARCKIT (the checks use the helper library)')
MAX_CONCURRENT = int(os.environ.get('QR_SANDBOX_PAR', '32'))
_sem = None
RUNNER = '\nimport json, sys, io, contextlib, traceback, copy, socket\ndef _blocked(*a, **k):\n    raise OSError("network disabled in sandbox")\nfor _m in ("connect", "connect_ex", "bind", "sendto", "sendmsg"):\n    setattr(socket.socket, _m, _blocked)\nsocket.create_connection = _blocked; socket.getaddrinfo = _blocked\nimport numpy as np\nD = json.load(open("task.json"))\ntrain = D["train"]; test_inputs = D["test_inputs"]\nns = {"np": np, "numpy": np, "train": copy.deepcopy(train), "test_inputs": copy.deepcopy(test_inputs),\n      "test_input": copy.deepcopy(test_inputs[0]), "__name__": "__main__", "__builtins__": __builtins__}\nbuf = io.StringIO()\nerr = None\ntry:\n    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n        exec(compile(open("code.py").read(), "<tool>", "exec"), ns)\nexcept BaseException:\n    err = "".join(traceback.format_exc().splitlines(True)[-6:])\n\ndef to_grid(x):\n    g = np.asarray(x)\n    if g.ndim != 2 or g.size == 0:\n        raise ValueError(f"transform returned an array of shape {g.shape}, expected a non-empty 2-D grid")\n    out = [[int(v) for v in row] for row in g.tolist()]\n    if len(out) > 30 or len(out[0]) > 30 or any(v < 0 or v > 9 for r in out for v in r):\n        raise ValueError(f"transform returned an invalid ARC grid ({len(out)}x{len(out[0])}, values must be 0-9, size <= 30)")\n    return out\n\ndef call(f, g):\n    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n        try:\n            return to_grid(f(copy.deepcopy(g)))\n        except ValueError:\n            raise\n        except Exception:\n            return to_grid(f(np.array(g)))   # transform written for numpy input\n\nres = {"stdout": buf.getvalue(), "error": err, "has_transform": callable(ns.get("transform"))}\nif res["has_transform"]:\n    f = ns["transform"]\n    pairs = []\n    for p in train:\n        r = {}\n        try:\n            out = call(f, p["input"])\n            r["out"] = out\n            r["ok"] = out == p["output"]\n        except BaseException as e:\n            r["ok"] = False\n            r["exc"] = (type(e).__name__ + ": " + str(e))[:300]\n        pairs.append(r)\n    res["train"] = pairs\n    tests = []\n    for g in test_inputs:\n        try:\n            tests.append({"out": call(f, g)})\n        except BaseException as e:\n            tests.append({"out": None, "exc": (type(e).__name__ + ": " + str(e))[:300]})\n    res["tests"] = tests\njson.dump(res, open("result.json", "w"))\n'
AK_PRE = '\nsys.path.insert(0, ".")\nimport arckit as ak\nfrom collections import Counter, defaultdict, deque\n'
AK_POST = '\nif res.get("has_transform") and all(p.get("ok") for p in res.get("train", [{}])) and res.get("train") \\\n        and all(t.get("out") is not None for t in res.get("tests", [])):\n    try:\n        res["ak_flag"] = ak.flag(train, test_inputs, [t["out"] for t in res["tests"]])\n    except Exception as e:\n        res["ak_flag_err"] = repr(e)[:200]\n'
AK_COV = '\nif res.get("has_transform") and all(p.get("ok") for p in res.get("train", [{}])) and res.get("train") \\\n        and all(t.get("out") is not None for t in res.get("tests", [])):\n    import time as _akt\n    def _akcov(g):\n        seen = set()\n        def tr(fr, ev, arg):\n            if fr.f_code.co_filename == "<tool>":\n                if ev == "line":\n                    seen.add(fr.f_lineno)\n                return tr\n            return None\n        sys.settrace(tr)\n        try:\n            call(f, g)\n        except BaseException:\n            pass\n        finally:\n            sys.settrace(None)\n        return seen\n    try:\n        _t0 = _akt.time()\n        _trl = set()\n        for p in train:\n            _trl |= _akcov(p["input"])\n            if _akt.time() - _t0 > 3.0:\n                raise TimeoutError("coverage skipped (slow program)")\n        res["ak_cov"] = [sorted(_akcov(g) - _trl) for g in test_inputs]\n    except BaseException as e:\n        res["ak_cov_err"] = repr(e)[:200]\n'
H2_EQ = '\nif res.get("has_transform") and all(p.get("ok") for p in res.get("train", [{}])) and res.get("train") \\\n        and all(t.get("out") is not None for t in res.get("tests", [])) and not res.get("ak_flag"):\n    try:\n        import time as _h2t\n        _t0 = _h2t.time()\n        call(f, train[0]["input"])\n        if _h2t.time() - _t0 > 0.1:\n            raise TimeoutError("equivariance skipped (slow program)")\n        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n            res["h2_eq"] = ak.equivariance(f, train, test_inputs, [t["out"] for t in res["tests"]], budget_s=1.5)\n    except BaseException as e:\n        res["h2_eq_err"] = repr(e)[:200]\n'

def _runner():
    if not ARCKIT:
        return RUNNER
    r = RUNNER.replace('D = json.load(open("task.json"))', AK_PRE + 'D = json.load(open("task.json"))', 1)
    r = r.replace('ns = {"np": np,', 'ns = {"ak": ak, "arckit": ak, "Counter": Counter, "defaultdict": defaultdict, "deque": deque, "np": np,', 1)
    post = AK_POST + (AK_COV if ARCKIT == 'review' or H2_FLAGFB else '') + (H2_EQ if H2_FLAGFB else '')
    r = r.replace('json.dump(res, open("result.json", "w"))', post + 'json.dump(res, open("result.json", "w"))', 1)
    assert r.count('import arckit') == 1 and '"ak": ak' in r and ('ak.flag' in r), 'RUNNER layout changed'
    return r

def _limits(timeout):

    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f

async def run_code(code, task, timeout=10.0):
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with _sem:
        d = tempfile.mkdtemp(prefix='qr_')
        try:
            json.dump({'train': task['train'], 'test_inputs': [t['input'] for t in task['test']]}, open(f'{d}/task.json', 'w'))
            open(f'{d}/code.py', 'w').write(code)
            open(f'{d}/runner.py', 'w').write(_runner())
            if ARCKIT:
                shutil.copyfile(ARCKIT_PATH, f'{d}/arckit.py')
            cmd = [PY, '-I', 'runner.py']
            if UNSHARE:
                cmd = [UNSHARE, '-rn'] + cmd
            env = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            p = await asyncio.create_subprocess_exec(*cmd, cwd=d, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, preexec_fn=_limits(timeout))
            to = False
            try:
                raw, _ = await asyncio.wait_for(p.communicate(), timeout + 5)
            except asyncio.TimeoutError:
                to = True
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raw = b''
                await p.wait()
            try:
                res = json.load(open(f'{d}/result.json'))
            except Exception:
                res = {'stdout': '', 'error': raw.decode(errors='replace')[-800:] or 'no result (killed: time or memory limit)', 'has_transform': False}
                if p.returncode is not None and p.returncode < 0 and (not to):
                    to = p.returncode in (-signal.SIGXCPU, -signal.SIGKILL)
            res['timeout'] = to
            if ARCKIT == 'review' and res.get('ak_cov'):
                res['_code'] = code
            return res
        finally:
            shutil.rmtree(d, ignore_errors=True)

def grid_text(g):
    return '\n'.join((''.join((str(v) for v in r)) for r in g))

def masked_diff(exp, got, max_cells=900):
    re_, ce = (len(exp), len(exp[0]))
    rg, cg = (len(got), len(got[0]) if got else 0)
    if (re_, ce) != (rg, cg):
        return f'wrong output size: expected {re_}x{ce}, got {rg}x{cg}.\nYour output ({rg}x{cg}):\n{grid_text(got)}'
    wrong = sum((exp[i][j] != got[i][j] for i in range(re_) for j in range(ce)))
    if re_ * ce > max_cells:
        return f'{wrong} of {re_ * ce} cells wrong.'
    rows = [''.join(('.' if exp[i][j] == got[i][j] else str(got[i][j]) for j in range(ce))) for i in range(re_)]
    return f"{wrong} of {re_ * ce} cells wrong. Your output with correct cells shown as '.' (expected output is in the task statement):\n" + '\n'.join(rows)

def verified(res):
    tr = res.get('train') or []
    ok = bool(tr) and all((p.get('ok') for p in tr)) and all((t.get('out') is not None for t in res.get('tests', [])))
    if ok and ARCKIT in ('block', 'review') and res.get('ak_flag'):
        return False
    return ok

def review_lines(res, code=None):
    if ARCKIT != 'review' or not any(res.get('ak_cov') or []):
        return ''
    src = (code if code is not None else res.get('_code', '')).splitlines()
    parts = []
    for k, ls in enumerate(res['ak_cov']):
        if ls:
            shown = '; '.join((src[i - 1].strip()[:100] for i in ls[:6] if 0 < i <= len(src)))
            parts.append(f"test input {k + 1}: line(s) {', '.join(map(str, ls[:12]))}" + (f' ({shown})' if shown else ''))
    return 'Review before this is accepted: these lines of your code run on a test input but on NONE of the training inputs, so the examples never checked them: ' + ' | '.join(parts) + '. Make sure that branch follows the rule (not a special case or a fallback). If it is right, run the same program again to confirm it.'

def feedback(res, task, timeout=10.0, max_fail_shown=2):
    parts = []
    out = res.get('stdout') or ''
    if len(out) > OUT_LIMIT:
        out = out[:OUT_LIMIT] + f"\n... [output truncated, {len(res['stdout'])} chars]"
    if out.strip():
        parts.append(out.rstrip())
    if res.get('timeout'):
        parts.append(f'[execution timed out after {timeout:.0f} s]')
    if res.get('error'):
        parts.append('Error:\n' + res['error'].rstrip()[-1200:])
    if res.get('has_transform') and 'train' in res:
        tr = res['train']
        n_ok = sum((bool(p.get('ok')) for p in tr))
        lines = [f'transform() check: {n_ok}/{len(tr)} training examples correct.']
        shown = 0
        for k, p in enumerate(tr):
            if p.get('ok'):
                lines.append(f'Example {k + 1}: correct')
            elif 'exc' in p:
                lines.append(f"Example {k + 1}: raised {p['exc']}")
            elif shown < max_fail_shown:
                lines.append(f'Example {k + 1}: ' + masked_diff(task['train'][k]['output'], p['out']))
                shown += 1
            else:
                lines.append(f'Example {k + 1}: wrong')
        for k, t in enumerate(res.get('tests', [])):
            if t.get('exc'):
                lines.append(f"Test input {k + 1}: transform raised {t['exc']}")
        if res.get('ak_flag'):
            lines.append('All training examples pass, BUT your output for the test input(s) breaks a property that EVERY training pair has: ' + '; '.join(res['ak_flag']) + '. The rule probably does not generalise (e.g. a branch that only the test input reaches, or a fallback that returns the input). ' + ('This program is not accepted yet: fix the rule and run it again.' if ARCKIT in ('block', 'review') else 'Check the rule before relying on it.'))
        elif verified(res):
            lines.append('All training examples pass.')
            cov = review_lines(res)
            if cov:
                lines.append(cov)
        parts.append('\n'.join(lines))
    txt = '\n\n'.join(parts) if parts else '(no output)'
    return txt[:7000]
