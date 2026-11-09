import asyncio, json, os, resource, shutil, signal, sys, tempfile
PY = sys.executable
UNSHARE = shutil.which('unshare')
if UNSHARE:
    import subprocess
    try:
        if subprocess.run([UNSHARE, '-rn', 'true'], capture_output=True, timeout=10).returncode != 0:
            UNSHARE = None
    except Exception:
        UNSHARE = None
OUT_LIMIT = 3000
RUNNER = '\nimport json, sys, io, contextlib, traceback, copy, socket\ndef _blocked(*a, **k):\n    raise OSError("network disabled in sandbox")\nfor _m in ("connect", "connect_ex", "bind", "sendto", "sendmsg"):\n    setattr(socket.socket, _m, _blocked)\nsocket.create_connection = _blocked; socket.getaddrinfo = _blocked\nimport numpy as np\nD = json.load(open("task.json"))\ntrain = D["train"]; test_input = D["test_input"]\nns = {"np": np, "numpy": np, "train": copy.deepcopy(train), "test_input": copy.deepcopy(test_input),\n      "__name__": "__main__", "__builtins__": __builtins__}\nbuf = io.StringIO()\nerr = None\ntry:\n    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n        exec(compile(open("code.py").read(), "<tool>", "exec"), ns)\nexcept BaseException:\n    err = "".join(traceback.format_exc().splitlines(True)[-6:])\n\ndef to_grid(x):\n    g = np.asarray(x)\n    if g.ndim != 2 or g.size == 0:\n        raise ValueError(f"transform returned an array of shape {g.shape}, expected 2-D")\n    return [[int(v) for v in row] for row in g.tolist()]\n\nres = {"stdout": buf.getvalue(), "error": err, "has_transform": callable(ns.get("transform"))}\nif res["has_transform"]:\n    f = ns["transform"]\n    pairs = []\n    for p in train:\n        r = {}\n        try:\n            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n                try:\n                    out = to_grid(f(copy.deepcopy(p["input"])))\n                except Exception:\n                    out = to_grid(f(np.array(p["input"])))   # transform written for numpy input\n            exp = p["output"]\n            r["out"] = out\n            r["ok"] = out == exp\n            if len(out) == len(exp) and len(out[0]) == len(exp[0]):\n                r["diff"] = [[i, j, out[i][j], exp[i][j]] for i in range(len(exp)) for j in range(len(exp[0]))\n                             if out[i][j] != exp[i][j]]\n            else:\n                r["shape"] = [len(out), len(out[0]), len(exp), len(exp[0])]\n        except BaseException as e:\n            r["ok"] = False\n            r["exc"] = (type(e).__name__ + ": " + str(e))[:300]\n        pairs.append(r)\n    res["train"] = pairs\n    try:\n        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n            try:\n                res["test_out"] = to_grid(f(copy.deepcopy(test_input)))\n            except Exception:\n                res["test_out"] = to_grid(f(np.array(test_input)))\n    except BaseException as e:\n        res["test_out"] = None\n        res["test_exc"] = (type(e).__name__ + ": " + str(e))[:300]\njson.dump(res, open("result.json", "w"))\n'

def _limits(timeout):

    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f

async def run_code(code, task, test_idx, timeout=10.0, use_unshare=True):
    d = tempfile.mkdtemp(prefix='tool_')
    try:
        json.dump({'train': task['train'], 'test_input': task['test'][test_idx]['input']}, open(f'{d}/task.json', 'w'))
        open(f'{d}/code.py', 'w').write(code)
        open(f'{d}/runner.py', 'w').write(RUNNER)
        cmd = [PY, '-I', 'runner.py']
        if use_unshare and UNSHARE:
            cmd = [UNSHARE, '-rn'] + cmd
        env = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        p = await asyncio.create_subprocess_exec(*cmd, cwd=d, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, preexec_fn=_limits(timeout))
        to = False
        try:
            raw, _ = await asyncio.wait_for(p.communicate(), timeout)
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
            res = {'stdout': '', 'error': raw.decode(errors='replace')[-800:] or 'no result (killed or crashed)', 'has_transform': False}
        res['timeout'] = to
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)

def fmt_grid(g):
    return '\n'.join((json.dumps(r) for r in g))

def feedback(res, timeout=10.0):
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
                lines.append(f'Example {k}: correct')
            elif 'exc' in p:
                lines.append(f"Example {k}: raised {p['exc']}")
            elif 'shape' in p:
                s = p['shape']
                lines.append(f'Example {k}: wrong shape, got {s[0]}x{s[1]}, expected {s[2]}x{s[3]}')
            else:
                df = p.get('diff', [])
                cells = ', '.join((f'({i},{j}) got {a} expected {b}' for i, j, a, b in df[:12]))
                more = f' (+{len(df) - 12} more)' if len(df) > 12 else ''
                lines.append(f'Example {k}: {len(df)} cells differ: {cells}{more}')
                if shown < 1 and 'out' in p:
                    lines.append(f'Your output for example {k}:\n' + fmt_grid(p['out']))
                    shown += 1
        if res.get('test_out') is not None:
            lines.append('transform(test_input) =\n' + fmt_grid(res['test_out']))
        elif res.get('test_exc'):
            lines.append('transform(test_input) raised ' + res['test_exc'])
        parts.append('\n'.join(lines))
    txt = '\n\n'.join(parts) if parts else '(no output)'
    return txt[:9000]
