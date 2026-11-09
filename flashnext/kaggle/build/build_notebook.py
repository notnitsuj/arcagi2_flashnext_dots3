#!/usr/bin/env python3
import argparse, ast, json, os, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / 'notebook' / 'rc8' / 'flash-next-arc-eval.ipynb'
_int = lambda lo, hi: lambda v: v.isdigit() and lo <= int(v) <= hi
_one = lambda *ok: lambda v: v in ok
ALLOWED = {'CONC': _int(8, 96), 'BT': _int(512, 8192), 'MINCAP': _int(4096, 131072), 'MARGIN': _int(60, 3600), 'ENDG': _int(60, 3600), 'KV_AGG': _one('max', 'mean'), 'IMPORT_FIX': _one('off', 'carry', 'alias'), 'TB_FIX': _one('off', 'on'), 'ANSWER_ORDER': _one('off', 'shape'), 'LATE_PF': _one('0', '1'), 'FN_HOST_DP_FUSE': _one('0', '1'), 'FN_HOST_LOGITS': _one('0', '1'), 'FN_MOE_ROUTER_FIRST': _one('0', '1'), 'FN_MOE_CHUNK': _int(256, 8192), 'FN_EP_Q8': _one('0', '1'), 'FN_EP_Q8_RS': _one('0', '1'), 'FN_TP_Q8_RS': _one('0', '1'), 'FN_MOE_HYB_FUSE': _one('0', '1', '2'), 'FN_GDN_BA_OVERLAP': _one('0', '1'), 'FN_LOAD_EC': _one('0', '1'), 'NCCL_CUMEM_ENABLE': _one('0', '1'), 'DRAIN': _one('0', '1'), 'FN_QSA_SHARE_TOPK': _one('0', '1'), 'CACHE_EXPORT': _one('0', '1')}
RESERVED = {'ARC_HOST_AR_BLOCKS', 'T0', 'PY', 'CODE', 'MODEL', 'EXPERTS', 'CHAL', 'OUT', 'WORK'}
COMMENT_REBUILT = '   # name -> bytes of the rebuilt build (None: no delta)'
COMMENT_LAUNCHER = '   # one script runs server + solver (None = separate stages)'
COMMENT_EVAL = '   # save run = full-length public-eval run (scored reruns use "rerun")'

def load_config(path):
    cfg = json.load(open(path))
    base = Path(path).resolve().parent
    if 'model' in cfg:
        cfg['models'] = [dict({'role': 'core', 'framework': 'pyTorch'}, **cfg.pop('model'))]
    for M in cfg['models']:
        M.setdefault('framework', 'pyTorch')
        if 'manifest' in M:
            man = json.load(open(base / M.pop('manifest')))
            M.setdefault('files', man['files'])
            M.setdefault('parts', list(man['parts']))
    for k, v in cfg.get('knobs', {}).items():
        assert k not in RESERVED, f"knob {k} collides with the notebook's own launcher environment"
        assert k in ALLOWED, f'unknown knob {k} (it would be silently ignored); allowed: {sorted(ALLOWED)}'
        assert ALLOWED[k](v), f'invalid value {k}={v}'
    if cfg.get('save_dt') is not None:
        assert cfg.get('eval_dt') is None, 'save_dt applies only without eval mode'
        assert int(cfg['save_dt']) >= 2100, 'save_dt < 2100: the launcher never starts the server with < 1800 s left'
    return cfg

def render_config_block(cfg):
    models = cfg['models']
    out = {'MODELS': 'MODELS = ' + json.dumps([{k: M[k] for k in ('role', 'owner', 'slug', 'framework', 'parts')} for M in models], indent=1), 'MODEL_FILES': 'MODEL_FILES = ' + json.dumps({M['role']: M['files'] for M in models}, indent=1, sort_keys=True), 'REBUILT_FILES': f"REBUILT_FILES = {cfg.get('experts_rebuilt_files')!r}", 'RUNTIME_DATASET': f"RUNTIME_DATASET = {cfg['runtime_dataset']!r}"}
    launcher = cfg.get('launcher')
    tail = f'LAUNCHER = {launcher!r}{COMMENT_LAUNCHER}\n'
    knobs = cfg.get('knobs') or {}
    if launcher and knobs:
        tail += ''.join((f'LAUNCHER["{m}"].update({json.dumps(knobs)})\n' for m in ('rerun', 'save')))
    if launcher and cfg.get('eval_dt') is not None:
        dt = int(cfg['eval_dt'])
        tail += f'LAUNCHER["eval"] = dict(LAUNCHER["rerun"]){COMMENT_EVAL}\n'
        tail += f'LAUNCHER["eval"].update({{"DT": {dt}, "KILLBY": {dt + 450}, "MARGIN": 300, "ENDG": 300}})\n'
    if launcher and cfg.get('save_dt') is not None:
        dt = int(cfg['save_dt'])
        tail += f'LAUNCHER["save"].update({{"DT": {dt}, "KILLBY": {dt + 200}}})\n'
    return (out, tail)

def splice_config_cell(src, cfg):
    lines = src.splitlines(True)
    tree = ast.parse(src)
    spans = {}
    launcher_line = None
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            name = st.targets[0].id
            if name in ('MODELS', 'MODEL_FILES', 'REBUILT_FILES', 'RUNTIME_DATASET'):
                spans[name] = (st.lineno, st.col_offset, st.end_lineno, st.end_col_offset)
            elif name == 'LAUNCHER':
                launcher_line = st.lineno
    assert set(spans) == {'MODELS', 'MODEL_FILES', 'REBUILT_FILES', 'RUNTIME_DATASET'} and launcher_line, 'template configuration cell: anchors not found'
    new, tail = render_config_block(cfg)
    for name, (l0, c0, l1, c1) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        assert c0 == 0, name
        rest = lines[l1 - 1][c1:]
        if name == 'REBUILT_FILES':
            rest = COMMENT_REBUILT + '\n'
        lines[l0 - 1:l1] = [new[name] + rest]
    text = ''.join(lines)
    lines = text.splitlines(True)
    tree = ast.parse(text)
    launcher_line = next((st.lineno for st in tree.body if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and (st.targets[0].id == 'LAUNCHER')))
    return ''.join(lines[:launcher_line - 1]) + tail

def build(cfg, template):
    nb = json.load(open(template))
    cells = nb['cells']
    for c in cells:
        c['source'] = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    nparts = sum((len(M['parts']) for M in cfg['models']))
    head = cells[0]
    assert head['cell_type'] == 'markdown' and head['source'].startswith('# '), 'template heading cell'
    head['source'] = re.sub('^# .*', '# ' + cfg['title'], head['source'], count=1)
    head['source'], n = re.subn('Kaggle Models, \\d+ parts', f'Kaggle Models, {nparts} parts', head['source'])
    assert n == 1, 'template heading: part count not found'
    cfg_cells = [c for c in cells if c['cell_type'] == 'code' and '\nMODELS = [' in '\n' + c['source']]
    assert len(cfg_cells) == 1, 'template: exactly one configuration cell expected'
    cfg_cells[0]['source'] = splice_config_cell(cfg_cells[0]['source'], cfg)
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            compile(c['source'], f'cell{i}', 'exec')
    kernel = {'id': cfg['kernel_id'], 'title': cfg.get('kernel_title', cfg['title']), 'code_file': cfg['notebook_name'], 'language': 'python', 'kernel_type': 'notebook', 'is_private': cfg.get('is_private', True), 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaL4', 'dataset_sources': [cfg['runtime_dataset']], 'competition_sources': [cfg.get('competition', 'arc-prize-2026-arc-agi-2')], 'kernel_sources': [], 'model_sources': [f"{M['owner']}/{M['slug']}/{M['framework']}/{p}/{M.get('version', 1)}" for M in cfg['models'] for p in M['parts']]}
    if cfg.get('docker_image'):
        kernel['docker_image'] = cfg['docker_image']
    return (nb, kernel)

def main():
    ap = argparse.ArgumentParser(description='render the submission notebook + kernel-metadata.json from CONFIG')
    ap.add_argument('config')
    ap.add_argument('out')
    ap.add_argument('--template', default=str(TEMPLATE))
    ap.add_argument('--forbid', action='append', default=[], help='regex that must not occur in the output (default: your home directory)')
    ap.add_argument('--check', action='store_true', help='report which cells differ from the template')
    a = ap.parse_args()
    cfg = load_config(a.config)
    nb, kernel = build(cfg, a.template)
    home = os.path.expanduser('~')
    forbid = a.forbid + ([re.escape(home)] if home not in ('', '/') else [])
    text = json.dumps(nb, indent=1) + json.dumps(kernel)
    hits = sorted({m.group(0) for f in forbid for m in re.finditer(f, text)})
    assert not hits, f'forbidden strings in the notebook or metadata: {hits}'
    if a.check:
        tpl = json.load(open(a.template))['cells']
        J = lambda s: ''.join(s) if isinstance(s, list) else s
        diff = [i for i, (x, y) in enumerate(zip(nb['cells'], tpl)) if x['cell_type'] != y['cell_type'] or J(x['source']) != J(y['source'])]
        diff += list(range(min(len(tpl), len(nb['cells'])), max(len(tpl), len(nb['cells']))))
        print(f"{len(nb['cells'])} cells; differ from the template: {diff or 'none'}")
    for c in nb['cells']:
        c['source'] = c['source'].splitlines(True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / cfg['notebook_name'], 'w'), indent=1)
    json.dump(kernel, open(out / 'kernel-metadata.json', 'w'), indent=2)
    print(f"wrote {out / cfg['notebook_name']} ({len(nb['cells'])} cells, {sum((len(M['parts']) for M in cfg['models']))} Model parts) and kernel-metadata.json ({kernel['id']})")
if __name__ == '__main__':
    main()
