#!/usr/bin/env python3
import argparse, copy, hashlib, json, os, random, shutil, subprocess, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
RUNTIME = KAGGLE / 'runtime_dataset'
NB_NAME = 'dots3-note-arc-agi-2.ipynb'
MANIFEST = json.load(open(HERE / 'model_parts.json'))
CK, PARTS, SLUG, OWNER = (MANIFEST['files'], MANIFEST['parts'], MANIFEST['slug'], MANIFEST['owner'])
RUNTIME_SLUG = 'dots3-note-arc-runtime'
SAVE_RUN = {'challenges': 'evaluation', 'score': True, 'deadline_total_s': 1200}
RES = {}

def strip(m):
    return {k: v for k, v in m.items() if k != 'note'}

def mode_sha(m):
    return hashlib.sha256(json.dumps(strip(m), sort_keys=True).encode()).hexdigest()

def release_mode():
    return json.load(open(RUNTIME / 'mode_submit.json'))

def preset_mode(preset):
    if preset == 'best_throughput':
        return release_mode()
    return json.load(open(HERE / 'presets' / f'{preset}_mode.json'))

def save_mode(preset):
    m = copy.deepcopy(preset_mode(preset))
    m.pop('save_overrides', None)
    m.update(SAVE_RUN)
    return m

def grid(r):
    return [[r.randint(0, 9) for _ in range(r.randint(2, 5))] for _ in range(r.randint(2, 5))]

def challenges(seed, n):
    r = random.Random(seed)
    return {f'{r.getrandbits(32):08x}': {'train': [{'input': grid(r), 'output': grid(r)}], 'test': [{'input': grid(r)} for _ in range(r.randint(1, 2))]} for _ in range(n)}

def fake_ckpt(d, complete=True, wrong_size=None, names=None):
    d.mkdir(parents=True, exist_ok=True)
    for n in names or CK:
        if not complete and n == 'model-00001.safetensors':
            continue
        with open(d / n, 'wb') as f:
            f.truncate(CK[n] + (1 if n == wrong_size else 0))

def layout(tmp, name, parts=tuple(PARTS), old_dataset=False, runtime_bundles=1, mock_launcher=None, complete=True, wrong_size=None, extra_version=None, runtime_mode=None):
    base = tmp / name
    shutil.rmtree(base, ignore_errors=True)
    inp, work = (base / 'input', base / 'working')
    work.mkdir(parents=True)
    for k in range(runtime_bundles):
        rt = inp / 'datasets' / OWNER / (RUNTIME_SLUG if k == 0 else RUNTIME_SLUG + '-copy')
        rt.mkdir(parents=True)
        if runtime_mode is None:
            shutil.copyfile(RUNTIME / 'mode_submit.json', rt / 'mode_submit.json')
        else:
            json.dump(runtime_mode, open(rt / 'mode_submit.json', 'w'))
        shutil.copyfile(RUNTIME / 'solve.py', rt / 'solve.py')
        if mock_launcher:
            (rt / 'run_l4.py').write_text(mock_launcher)
        else:
            shutil.copyfile(RUNTIME / 'run_l4.py', rt / 'run_l4.py')
        if k == 0:
            (rt / 'arc_py312.tarbin').write_bytes(b'\x00')
    comp = inp / 'competitions/arc-prize-2026-arc-agi-2'
    comp.mkdir(parents=True)
    json.dump(challenges(1, 5), open(comp / 'arc-agi_test_challenges.json', 'w'))
    json.dump(challenges(2, 4), open(comp / 'arc-agi_evaluation_challenges.json', 'w'))
    for pt in parts:
        fake_ckpt(inp / f'models/{OWNER}/{SLUG}/pytorch/{pt}/1', complete, wrong_size=wrong_size, names=PARTS[pt])
    if extra_version:
        fake_ckpt(inp / f'models/{OWNER}/{SLUG}/pytorch/{extra_version}/2', names=PARTS[extra_version])
    if old_dataset:
        fake_ckpt(inp / 'datasets' / OWNER / 'lb144r-checkpoint-copy')
    return (inp, work)

def run_nb(py, nb, inp, work, extra_env=None, timeout=600):
    out = work.parent / 'out.ipynb'
    env = dict(os.environ, ARC_INPUT=str(inp), ARC_WORK=str(work), ARC_CKPT_MERGE_DIR=str(work.parent / 'merged'))
    env.pop('KAGGLE_IS_COMPETITION_RERUN', None)
    env.update(extra_env or {})
    t = time.time()
    r = subprocess.run([py, '-m', 'papermill', str(nb), str(out), '--cwd', str(work), '-k', 'python3', '--log-output'], env=env, capture_output=True, text=True, timeout=timeout)
    text = ''
    if out.exists():
        o = json.load(open(out))
        text = '\n'.join((''.join(x.get('text', '')) if isinstance(x.get('text'), (str, list)) else '' for c in o['cells'] for x in c.get('outputs', [])))
    return (r.returncode, text, r.stderr[-3000:], time.time() - t)

def check(name, cond, info=''):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info and (not cond) else ''), flush=True)
MOCK = '\nimport json, os, sys, threading, time, subprocess\nfrom pathlib import Path\nW = os.environ.get("ARC_WORK", "/kaggle/working")\nLOG = open(os.path.join(W, "run_log.txt"), "a")\nT0 = time.time()\ndef log(*a):\n    s = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)\n    print(s, flush=True); LOG.write(s + "\\n"); LOG.flush()\nif os.environ.get("MOCK_STAGE") != "2":\n    env_ok = {k: os.environ.get(k) for k in ("ARC_MODE_FILE", "ARC_CKPT_DIR")}\n    bad = [k for k in os.environ if k.startswith(("JPY_", "PYDEVD_")) or k == "MPLBACKEND"]\n    code = Path(os.path.abspath(__file__)).parent\n    mode = json.load(open(code / os.environ.get("ARC_MODE_FILE", "mode.json")))   # the launcher\'s own lookup\n    json.dump({"env": env_ok, "kernel_vars": bad, "pid": os.getpid(), "mode": mode}, open(os.path.join(W, "mock_env.json"), "w"))\n    log("stage: always (mount mock)"); log("re-exec under", sys.executable); LOG.close()\n    os.environ["MOCK_STAGE"] = "2"\n    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])\nkill_at = T0 + float(os.environ.get("MOCK_KILL_S", "20"))\nsrv = subprocess.Popen(["sleep", "3600"], start_new_session=True)   # the \'server\' tree: own session, like vLLM\nopen(os.path.join(W, "mock_server.pid"), "w").write(str(srv.pid))\ndef descendants(root):\n    kids = {}\n    for d in os.listdir("/proc"):\n        if d.isdigit():\n            try:\n                kids.setdefault(int(open(f"/proc/{d}/stat").read().rpartition(")")[2].split()[1]), []).append(int(d))\n            except Exception:\n                pass\n    out, todo = [], [root]\n    while todo:\n        for c in kids.get(todo.pop(), []):\n            out.append(c); todo.append(c)\n    return out\ndef fire():\n    while time.time() < kill_at:\n        time.sleep(0.2)\n    n = 0\n    for p in descendants(os.getpid()):\n        try:\n            os.kill(p, 9); n += 1\n        except OSError:\n            pass\n    log("kill-by reached (S0 + 42600 s): SIGKILLed %d processes" % n)\n    os._exit(0)\nif os.environ.get("MOCK_NO_KILLBY") != "1":\n    threading.Thread(target=fire, daemon=True).start()\nwith open(os.path.join(W, "prefetch_log.txt"), "w") as pf:\n    pf.write("[    0.0s] stage(robust) 3 files\\n[    5.0s] staged arc_py312.tarbin 7.66 GiB cum 8.0 GiB at 1600 MB/s\\n"\n             "[   20.0s] staged lowbit-L01.pt 1.13 GiB cum 30.0 GiB at 1500 MB/s\\n"\n             "[   30.0s] staged lowbit-L45.pt 1.13 GiB cum 40.0 GiB at 1300 MB/s\\n[   31.0s] stage finished\\n")\nlog("startup guard: {}"); log("triton cache restored: 0 files"); log("server pid", srv.pid)\ntime.sleep(2); log("server ready after 2s")\n# a partly broken submission: one task missing, one malformed attempt\nch = json.load(open([os.path.join(r, f) for r, _, fs in os.walk(os.environ["ARC_INPUT"]) for f in fs if f == "arc-agi_test_challenges.json"][0]))\nsub = {t: [{"attempt_1": [[1]], "attempt_2": "bad"} for _ in v["test"]] for t, v in list(ch.items())[1:]}\njson.dump(sub, open(os.path.join(W, "submission.json"), "w"))\nlog("solver save-run mock")\ntime.sleep(3600)\n'

def exec_guard(nb_json):
    cell = next((c['source'] for c in nb_json['cells'] if c['cell_type'] == 'code' and 'def submission_mode_guard' in c['source']))
    i = cell.index('def submission_mode_guard')
    k = cell.find('\ndef ', i + 1)
    ns = {'json': json}
    exec(cell[i:k if k > 0 else len(cell)], ns)
    return ns['submission_mode_guard']

def guard_msg(guard, m):
    try:
        guard(m)
        return None
    except AssertionError as ex:
        return str(ex)

def rows_ok(m):
    e = m['env']
    w = int(e.get('ARC_OFFLOAD_WAYS', '1'))
    rows = [int(x.split(':')[1]) for x in e.get('ARC_OFFLOAD_CACHE_ROWS_L', '').split(',') if x] + [int(e.get('ARC_OFFLOAD_CACHE_ROWS', '0'))]
    return (w > 1 and all((r % w == 0 for r in rows)), (w, sorted(set(rows))))

def suite(preset, py, tmp):
    p = lambda s: f'[{preset}] {s}'
    nb = KAGGLE / 'notebook' / preset / NB_NAME
    nb_json = json.load(open(nb))
    tmp = tmp / preset
    tmp.mkdir(parents=True)
    want = preset_mode(preset)
    r = subprocess.run([sys.executable, str(HERE / 'build_notebook.py'), '--preset', preset, '--check'], capture_output=True, text=True)
    check(p('notebook and kernel-metadata.json == build_notebook.py output'), r.returncode == 0, r.stdout + r.stderr)
    inp, work = layout(tmp, 'A')
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('A notebook ran'), rc == 0, err)
    plan = json.load(open(work / 'notebook_plan.json'))
    md = work.parent / 'merged'
    links = {f.name: os.path.realpath(f) for f in md.iterdir()} if md.exists() else {}
    check(p('A 4 Model parts merged into one directory'), plan['ARC_CKPT_DIR'] == str(md) and set(links) == set(CK) and all((f'/models/{OWNER}/{SLUG}/pytorch/lb144r-p' in v for v in links.values())) and all((os.path.getsize(md / n) == CK[n] for n in CK)), plan)
    check(p('A launcher command = [python, <runtime>/run_l4.py]'), plan['cmd'][1].endswith(f'datasets/{OWNER}/{RUNTIME_SLUG}/run_l4.py'), plan)
    got = json.load(open(work / 'mode_diag.json'))
    check(p('A save-run mode = the preset mode without save_overrides + the smoke-test settings'), strip(got) == strip(save_mode(preset)), {k for k in set(got) | set(save_mode(preset)) if got.get(k) != save_mode(preset).get(k)})
    check(p('A ARC_MODE_FILE = WORK/mode_diag.json'), plan['ARC_MODE_FILE'] == str(work / 'mode_diag.json'), plan)
    check(p('A save watchdog 1800/90'), plan['save_watchdog'] == [1800, 90])
    sub = json.load(open(work / 'submission.json'))
    check(p('A dry-run submission check writes a valid echo'), len(sub) == 5)
    if preset != 'best_throughput':
        pf = work / f'mode_{preset}.json'
        check(p(f'A mode_{preset}.json written and == presets/{preset}_mode.json'), pf.exists() and json.load(open(pf)) == want)
    inp, work = layout(tmp, 'B')
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1', 'KAGGLE_IS_COMPETITION_RERUN': '1'})
    plan = json.load(open(work / 'notebook_plan.json')) if rc == 0 else {}
    if preset == 'best_throughput':
        check(p("B rerun ARC_MODE_FILE == mode_submit.json (the runtime's mode as is)"), rc == 0 and plan['ARC_MODE_FILE'] == 'mode_submit.json', err)
    else:
        pf = work / f'mode_{preset}.json'
        check(p(f'B rerun ARC_MODE_FILE == absolute WORK/mode_{preset}.json holding the preset mode'), rc == 0 and plan['ARC_MODE_FILE'] == str(pf) and os.path.isabs(plan['ARC_MODE_FILE']) and (json.load(open(pf)) == want), err or plan)
        check(p("B the preset mode's hash is the notebook's MODE_SHA256"), f'MODE_SHA256 = "{mode_sha(want)}"' in nb_json['cells'][2]['source'])
        ref = json.load(open(HERE / 'presets' / 'v22c_mode.json'))
        skip = set(json.load(open(HERE / 'presets' / f'{preset}.json'))['reference_may_differ'])
        differ = sorted((k for k in set(ref) | set(want) if k not in skip and ref.get(k) != want.get(k)))
        check(p('B preset mode == v22c in every serving and solver key'), not differ, differ)
    check(p('B no save watchdog'), plan.get('save_watchdog', 0) is None, plan)
    ok, info = rows_ok(want)
    check(p('B every row-cache size of the mode is a multiple of ARC_OFFLOAD_WAYS'), ok, info)
    guard = exec_guard(nb_json)
    rel = release_mode()
    check(p('B release guard accepts the runtime mode'), guard_msg(guard, rel) is None, guard_msg(guard, rel))
    for bad_rows in ('408700', '362604'):
        bm = copy.deepcopy(rel)
        bm['env']['ARC_OFFLOAD_CACHE_ROWS_L'] = bm['env']['ARC_OFFLOAD_CACHE_ROWS_L'].replace(':362600', ':' + bad_rows)
        check(p(f'B release guard refuses row cache {bad_rows}'), guard_msg(guard, bm) is not None)
    bm = copy.deepcopy(rel)
    bm['env']['ARC_OFFLOAD_CACHE_ROWS_L'] = bm['env']['ARC_OFFLOAD_CACHE_ROWS_L'].replace('5:362600', '5:362604')
    msg = guard_msg(guard, bm)
    check(p('B divisibility check fires (prefix intact, one later layer off)'), msg is not None and 'multiples of ARC_OFFLOAD_WAYS' in msg, msg)
    bm = copy.deepcopy(rel)
    bm['util'] = 0.9
    inp, work = layout(tmp, 'B2', runtime_mode=bm)
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    expect = 'not the release mode' if preset != 'best_throughput' else "runtime mode differs from this notebook's configuration"
    check(p('B2 a runtime mode_submit.json other than the release mode is refused'), rc != 0 and expect in out + err, (out + err)[-600:])
    if preset != 'best_throughput':
        nb2 = tmp / 'edited.ipynb'
        j = copy.deepcopy(nb_json)
        j['cells'][2]['source'] = j['cells'][2]['source'].replace('"deadline_total_s": 41400,', '"deadline_total_s": 41000,', 1)
        nb2.write_text(json.dumps(j, ensure_ascii=False))
        inp, work = layout(tmp, 'B3')
        rc, out, err, _ = run_nb(py, nb2, inp, work, {'NB_DRY_RUN': '1'})
        check(p('B3 PRESET_CHANGES edited without its pins: refused'), rc != 0 and f"{preset} mode differs from this notebook's configuration" in out + err, (out + err)[-600:])
    inp, work = layout(tmp, 'C', parts=(), old_dataset=True)
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('C no Model attached: refused with a clear message (no dataset fallback)'), rc != 0 and 'checkpoint parts not attached' in out + err and (f'{OWNER}/{SLUG}' in out + err))
    inp, work = layout(tmp, 'C2', old_dataset=True)
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    plan = json.load(open(work / 'notebook_plan.json')) if rc == 0 else {}
    check(p('C2 an extra checkpoint copy is ignored, the parts are used'), rc == 0 and plan['ARC_CKPT_DIR'] == str(work.parent / 'merged'), err)
    inp, work = layout(tmp, 'D', extra_version='lb144r-p2')
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('D two versions of a part refused'), rc != 0 and 'several versions' in out + err)
    inp, work = layout(tmp, 'D2', parts=('lb144r-p1', 'lb144r-p2', 'lb144r-p4'))
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('D2 a missing part refused, named'), rc != 0 and "['lb144r-p3']" in out + err)
    inp, work = layout(tmp, 'E', runtime_bundles=2)
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('E two runtime bundles refused'), rc != 0 and 'exactly one runtime bundle' in out + err)
    inp, work = layout(tmp, 'F', complete=False)
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('F a part missing a file refused'), rc != 0 and 'missing model-00001.safetensors' in out + err)
    inp, work = layout(tmp, '', wrong_size='lowbit-L17.pt')
    rc, out, err, _ = run_nb(py, nb, inp, work, {'NB_DRY_RUN': '1'})
    check(p('F wrong file size refused'), rc != 0 and 'lowbit-L17.pt' in out + err)
    inp, work = layout(tmp, 'G', mock_launcher=MOCK)
    rc, out, err, dt = run_nb(py, nb, inp, work, {'MOCK_KILL_S': '25', 'MPLBACKEND': 'module://matplotlib_inline.backend_inline'})
    check(p("G notebook completed after the launcher's hard stop"), rc == 0, err)
    me = json.load(open(work / 'mock_env.json'))
    check(p('G launcher env: ARC_MODE_FILE + ARC_CKPT_DIR set, kernel vars removed'), me['env']['ARC_MODE_FILE'] == str(work / 'mode_diag.json') and me['env']['ARC_CKPT_DIR'] == str(work.parent / 'merged') and (me['kernel_vars'] == []), me['env'])
    check(p('G the launcher reads the save-run mode'), strip(me['mode']) == strip(save_mode(preset)))
    check(p('G launcher output in the cell log'), 're-exec under' in out and 'kill-by reached' in out, out[-1500:])
    check(p('G stage lines'), 'server ready after' in out and 'kill-by reached' in out)
    spid = int((work / 'mock_server.pid').read_text())
    alive = os.path.exists(f'/proc/{spid}') and open(f'/proc/{spid}/stat').read().rpartition(')')[2].split()[0] != 'Z'
    check(p('G server session killed by the hard stop'), not alive)
    from_nb = json.load(open(work / 'submission.json'))
    ch = json.load(open(inp / 'competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json'))
    check(p('G submission repaired to every test input'), set(from_nb) == set(ch) and all((len(from_nb[t]) == len(v['test']) for t, v in ch.items())))
    check(p('G kept the valid attempts'), all((e['attempt_1'] == [[1]] for t in list(ch)[1:] for e in from_nb[t])))
    check(p('G wall time ~ hard stop'), dt < 120, f'{dt:.0f} s')
    try:
        sd = json.load(open(work / 'storage_diagnostics.json'))
        srcs = {o['source']: o for o in sd['sources']}
        ok = set(PARTS) | {'runtime'} <= set(srcs) and sd['overall']['read_mb_s'] == round(40 * 1024 / 30) and (srcs['runtime']['staged_files'] == 1) and (sum((o['staged_files'] for o in sd['sources'])) == 3) and all((o['fstype'] not in ('', None) for o in sd['sources']))
    except Exception as ex:
        ok, sd = (False, repr(ex))
    check(p('G storage diagnostics: per-part + runtime rows, overall rate, JSON written'), ok, sd)
    check(p('G storage table printed'), 'overall' in out and 'lb144r-p1' in out and ('fstype' in out))
    inp, work = layout(tmp, 'G_rerun', mock_launcher=MOCK)
    rc, out, err, dt = run_nb(py, nb, inp, work, {'MOCK_KILL_S': '15', 'KAGGLE_IS_COMPETITION_RERUN': '1'})
    me = json.load(open(work / 'mock_env.json')) if (work / 'mock_env.json').exists() else {'mode': {}, 'env': {}}
    check(p('G scored rerun: notebook completed, the launcher read the preset mode'), rc == 0 and me['mode'] == want and (mode_sha(me['mode']) == mode_sha(want)), (err, me['env']))
    check(p('G scored rerun: no save-run overrides, submission repaired'), json.load(open(work / 'notebook_plan.json'))['save_watchdog'] is None and set(json.load(open(work / 'submission.json'))) == set(json.load(open(inp / 'competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json'))))
    ns = {}
    os.environ.update(ARC_INPUT=str(tmp / 'A/input'), ARC_WORK=str(tmp / 'H'))
    (tmp / 'H').mkdir()
    for c in nb_json['cells']:
        if c['cell_type'] == 'code' and ('def walk_inputs' in c['source'] or 'class Runner' in c['source'] or 'MODE_SHA256 =' in c['source']):
            exec(c['source'], ns)
    pr = subprocess.Popen(['sh', '-c', 'sleep 600 & sleep 600'])
    time.sleep(0.5)
    kids = ns['descendants'](pr.pid)
    r = ns['Runner'](pr, tmp / 'H/run_log.txt', 0, time.time() - 50000, time.time() - ns['BACKSTOP_GRACE_S'] - 1)
    r.poll()
    time.sleep(1.0)
    gone = pr.poll() is not None and (not any((os.path.exists(f'/proc/{k}') and open(f'/proc/{k}/stat').read().rpartition(')')[2].split()[0] != 'Z' for k in kids)))
    check(p('H backstop kills a launcher past its hard stop (and its children)'), gone and r.backstop)
    r2 = ns['Runner'](subprocess.Popen(['sleep', '5']), tmp / 'H/run_log.txt', 0, time.time(), time.time() + 1000)
    r2.poll()
    check(p('H no backstop before the hard stop'), r2.alive() and (not r2.backstop))
    r2.proc.kill()
    diag = next((c['source'] for c in nb_json['cells'] if c['cell_type'] == 'code' and 'def storage_report' in c['source']))
    for name, bad in (('missing checkpoint dir', {'checkpoint': str(tmp / 'nope'), 'code': str(tmp / 'nope')}), ('empty inputs', {}), ('not a dict', None)):
        ns3 = dict(ns)
        ns3['INPUTS'] = bad
        (tmp / 'H/prefetch_log.txt').write_text('[ garbage\n[   x s] staged\n')
        try:
            exec(diag, ns3)
            raised = None
        except BaseException as ex:
            raised = repr(ex)
        check(p(f'J diagnostics cell survives: {name}'), raised is None, raised)
    g_in = {'checkpoint': str(tmp / 'G/merged'), 'code': str(tmp / 'G/input/datasets' / OWNER / RUNTIME_SLUG)}
    for n in ('cuda_toolkit.tarbin', 'triton_cache_lv5.tarbin', 'p1s_delta.safetensors'):
        (Path(g_in['code']) / n).touch()
    shutil.copy(tmp / 'G/working/prefetch_log.txt', tmp / 'H/prefetch_log.txt')
    ns3 = dict(ns)
    ns3['INPUTS'] = g_in
    ns3['open'] = lambda *a, **k: (_ for _ in ()).throw(PermissionError('no /proc')) if 'mountinfo' in str(a[0]) else open(*a, **k)
    try:
        exec(diag, ns3)
        raised = None
    except BaseException as ex:
        raised = repr(ex)
    sd = json.load(open(tmp / 'H/storage_diagnostics.json'))
    check(p('J diagnostics cell survives: unreadable mountinfo (speeds kept)'), raised is None and all((o['fstype'] == '?' for o in sd['sources'])) and sd['overall']['read_mb_s'], (raised, sd))
    for k in ('ARC_INPUT', 'ARC_WORK'):
        os.environ.pop(k, None)

def launcher_checks(tmp):
    import glob as g
    src = (RUNTIME / 'run_l4.py').read_text()
    fn = src[src.index('def _ckpt_dir():'):src.index('def _mount_fstype')]
    base = tmp / 'launcher'
    for which, extra in (('model', False), ('model + extra copy', True)):
        inp, work = layout(base, which.replace(' ', '_').replace('+', 'and'), old_dataset=extra)
        ns2 = {'os': os, 'Path': Path, 'MODE': {'ckpt_marker': MANIFEST['marker']}, 'IN': str(inp), 'find': lambda pat: sorted(g.glob(pat, recursive=True))[0]}
        exec(fn, ns2)
        os.environ.pop('ARC_CKPT_DIR', None)
        first = str(ns2['_ckpt_dir']())
        ref = os.path.dirname(sorted(g.glob(str(inp) + '/**/' + MANIFEST['marker'], recursive=True))[0])
        check(f'[launcher] I {which}: ARC_CKPT_DIR unset == first sorted marker match', first == ref)
        merged = base / 'merged'
        merged.mkdir(exist_ok=True)
        (merged / MANIFEST['marker']).touch()
        os.environ['ARC_CKPT_DIR'] = str(merged)
        check(f'[launcher] I {which}: ARC_CKPT_DIR -> the merged parts', str(ns2['_ckpt_dir']()) == str(merged))
        os.environ['ARC_CKPT_DIR'] = str(base / 'nonexistent')
        try:
            ns2['_ckpt_dir']()
            raised = False
        except FileNotFoundError:
            raised = True
        check(f'[launcher] I {which}: unusable ARC_CKPT_DIR is an error (no marker search)', raised)
        os.environ.pop('ARC_CKPT_DIR', None)
    line = 'MODE = json.load(open(code / os.environ.get("ARC_MODE_FILE", "mode.json")))'
    check('[launcher] K run_l4.py reads code / ARC_MODE_FILE (an absolute path wins over the code dir)', line in src and Path('/runtime') / '/work/mode_best_score.json' == Path('/work/mode_best_score.json'))

def main():
    ap = argparse.ArgumentParser(description='Mocked tests of the notebook presets (papermill, synthetic inputs, no GPU). PY = a Python with papermill and ipykernel (default: this one).')
    ap.add_argument('workdir', nargs='?', help='scratch folder (default: a new temporary folder)')
    ap.add_argument('--preset', choices=('best_throughput', 'best_score', 'all'), default='all')
    a = ap.parse_args()
    py = os.environ.get('PY', sys.executable)
    tmp = Path(a.workdir or tempfile.mkdtemp(prefix='nbtest_'))
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for preset in ('best_throughput', 'best_score') if a.preset == 'all' else (a.preset,):
        suite(preset, py, tmp)
    launcher_checks(tmp)
    n_ok = sum(RES.values())
    print(f'\n{n_ok}/{len(RES)} PASS' + ('' if n_ok == len(RES) else '  FAILED: ' + ', '.join((k for k, v in RES.items() if not v))))
    json.dump(RES, open(tmp / 'results.json', 'w'), indent=1)
    sys.exit(0 if n_ok == len(RES) else 1)
if __name__ == '__main__':
    main()
