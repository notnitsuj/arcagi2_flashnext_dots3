#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
REPO = KAGGLE.parent
BINARIES = {'env_tar': 'arc_py312.tarbin', 'cuda_tar': 'cuda_toolkit.tarbin', 'triton_cache': 'triton_cache_lv5.tarbin', 'delta': 'p1s_delta.safetensors'}
KERNELS = ('host_arc', 'offload_attn', 'offload_attn2', 'swa_attn', 'topk_select')

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()

def place(src, dst, copy):
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)

def text_files(runtime, engine):
    out = {p.name: p for p in sorted(runtime.iterdir()) if p.is_file() and p.name != 'MANIFEST.md5'}
    pkg = engine / 'arcstats_pkg' / 'arcstats'
    for p in sorted(pkg.glob('*.py')):
        out['arcstats__' + p.name] = p
    patched = engine / 'vllm_patch'
    for p in sorted((patched / 'vllm').rglob('*.py')):
        out['__'.join(p.relative_to(patched).parts)] = p
    return out

def main():
    ap = argparse.ArgumentParser(description='Assemble the Dots3-Note runtime dataset folder (upload it with `kaggle datasets create|version -p OUT`) and compare every file with MANIFEST.md5 (dataset version 5). Text files come from runtime_dataset/ and engine/; binaries from --binaries and the per-file options.')
    ap.add_argument('out', help='output folder (must be empty or absent)')
    ap.add_argument('--runtime', default=str(KAGGLE / 'runtime_dataset'))
    ap.add_argument('--engine', default=str(REPO / 'engine'))
    ap.add_argument('--binaries', help='folder holding binaries under their published names (for example a download of dataset version 5): ' + ', '.join(list(BINARIES.values()) + [f'arcstats__{k}.so' for k in KERNELS]))
    ap.add_argument('--env-tar', help='arc_py312.tarbin (build_nb_env.py)')
    ap.add_argument('--cuda-tar', help='cuda_toolkit.tarbin (build_bundle.sh)')
    ap.add_argument('--triton-cache', help='triton_cache_lv5.tarbin (a Triton cache saved by a run with triton_cache_save)')
    ap.add_argument('--delta', help='p1s_delta.safetensors (the fine-tuned delta)')
    ap.add_argument('--kernels', help='folder with the compiled modules host_arc.so, offload_attn.so, offload_attn2.so, swa_attn.so, topk_select.so (engine/build_kernels.sh)')
    ap.add_argument('--dataset-id', help='OWNER/SLUG: also write dataset-metadata.json for the Kaggle CLI')
    ap.add_argument('--title', default='Dots3-Note ARC runtime')
    ap.add_argument('--copy', action='store_true', help='copy binaries instead of hard-linking them')
    ap.add_argument('--strict', action='store_true', help='exit 1 also when a binary is missing or differs')
    a = ap.parse_args()
    runtime, engine, out = (Path(a.runtime).resolve(), Path(a.engine).resolve(), Path(a.out).resolve())
    want = {}
    for line in open(runtime / 'MANIFEST.md5'):
        if line.strip():
            h, n = line.split(None, 1)
            want[n.strip()] = h
    assert not out.exists() or not any(out.iterdir()), f'{out} exists and is not empty'
    out.mkdir(parents=True, exist_ok=True)
    texts = text_files(runtime, engine)
    for n, p in texts.items():
        shutil.copyfile(p, out / n)
    bins = {}
    if a.binaries:
        b = Path(a.binaries)
        for n in list(BINARIES.values()) + [f'arcstats__{k}.so' for k in KERNELS]:
            if (b / n).is_file():
                bins[n] = b / n
    for opt, n in BINARIES.items():
        if getattr(a, opt):
            bins[n] = Path(getattr(a, opt))
    if a.kernels:
        for k in KERNELS:
            if (Path(a.kernels) / f'{k}.so').is_file():
                bins[f'arcstats__{k}.so'] = Path(a.kernels) / f'{k}.so'
    for n, p in bins.items():
        place(p, out / n, a.copy)
    if a.dataset_id:
        json.dump({'id': a.dataset_id, 'title': a.title, 'licenses': [{'name': 'other'}]}, open(out / 'dataset-metadata.json', 'w'), indent=1)
    bad_text, bad_bin = (0, 0)
    for n in sorted(set(want) | {p.name for p in out.iterdir() if p.name != 'dataset-metadata.json'}):
        p = out / n
        is_text = n in texts or (n not in bins and (not n.endswith(('.tarbin', '.safetensors', '.so'))))
        if not p.exists():
            print(f"  {n:72s} {('MISSING' if is_text else 'not included')}")
            bad_text += is_text
            bad_bin += not is_text
            continue
        got = md5(p)
        state = 'matches dataset version 5' if want.get(n) == got else 'NOT in MANIFEST.md5' if n not in want else 'DIFFERS from dataset version 5' + ('' if is_text else ' (a rebuilt binary)')
        if want.get(n) != got:
            bad_text += is_text
            bad_bin += not is_text
        print(f'  {n:72s} {got}  {state}')
    n_files = sum((1 for p in out.iterdir() if p.name != 'dataset-metadata.json'))
    print(f"wrote {out}: {n_files} of {len(want)} published files; text files: {('all byte-identical' if not bad_text else f'{bad_text} differ or are missing')}; binaries: {('all identical' if not bad_bin else f'{bad_bin} missing or rebuilt')}")
    sys.exit(1 if bad_text or (a.strict and bad_bin) else 0)
if __name__ == '__main__':
    main()
