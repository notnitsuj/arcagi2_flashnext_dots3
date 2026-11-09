import argparse, os, re, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_fn as S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trained')
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--no-ovr', action='store_true')
    a = ap.parse_args()
    src, out = (Path(a.src), Path(a.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ovr').mkdir(exist_ok=True)
    tr = torch.load(a.trained, map_location='cpu')
    layers = {}
    for k, v in tr.items():
        m = re.match('layers\\.(\\d+)\\.mlp\\.(tgu|td|gate)$', k)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = v
    other = [k for k in tr if not re.match('layers\\.(\\d+)\\.mlp\\.(tgu|td|gate)$', k)]
    if other:
        torch.save({k: tr[k] for k in other}, out / 'kd_extra.pt')
        print('non-expert trained tensors saved to kd_extra.pt (need a loader overlay):', len(other))
    done = set()
    for l, d in sorted(layers.items()):
        lb = torch.load(src / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
        ovr = {}
        pre = f'model.language_model.layers.{l}'
        for i, e in enumerate(lb['experts']):
            ex = lb[e]
            for nm, tk in (('gu', 'tgu'), ('d', 'td')):
                s0 = ex[nm + '_s'].float()
                if tk in d:
                    ex[nm + '_s'] = (s0 * d[tk][i].float().exp()).half()
            if not a.no_ovr:
                for nm in ('gu', 'd'):
                    b = ex['bits']
                    pl = (ex[nm]['p4'],) if b == 4 else (ex[nm]['p2'], ex[nm].get('p1'))
                    W = S.deq(pl, b, ex[nm + '_z'], ex[nm + '_s'].float())
                    if nm == 'gu':
                        g, u = W.chunk(2, 0)
                        ovr[f'{pre}.mlp.experts.{i}.gate_proj.weight'] = g.contiguous()
                        ovr[f'{pre}.mlp.experts.{i}.up_proj.weight'] = u.contiguous()
                    else:
                        ovr[f'{pre}.mlp.experts.{i}.down_proj.weight'] = W.contiguous()
        torch.save(lb, out / f'lowbit-L{l:02d}.pt')
        r = torch.load(src / f'router-L{l:02d}.pt', map_location='cpu', weights_only=False)
        if 'gate' in d:
            assert d['gate'].shape == r['weight'].shape
            r['weight'] = d['gate'].to(torch.bfloat16).contiguous()
        torch.save(r, out / f'router-L{l:02d}.pt')
        if not a.no_ovr:
            ovr[f'{pre}.mlp.gate.weight'] = r['weight']
            save_file(ovr, str(out / 'ovr' / f'ovr-L{l:02d}.safetensors'), metadata={'format': 'pt'})
        done.add(f'lowbit-L{l:02d}.pt')
        done.add(f'router-L{l:02d}.pt')
        done.add(f'ovr/ovr-L{l:02d}.safetensors')
        print('layer', l, flush=True)
    for f in sorted(src.rglob('*')):
        rel = str(f.relative_to(src))
        if f.is_dir() or rel in done or rel in ('DONE', 'README.md'):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(f, dst)
    (out / 'README.md').write_text(f'# {out.name} (trained scales and router rows on top of {src.name})\n- Same files/format as {src}; trunk lowbit-L00..47 scales (fp16) and router-L00..47 rows replaced by the trained values of {a.trained}. Codes/zeros/packing/keep set unchanged; L48 (MTP) and keep.* hard-linked from the source.\n- ovr/: BF16 dequant + trained router rows (`.mlp.gate.weight`) for evaluation serving through serve/serve_build.sh.\n')
    (out / 'DONE').write_text('ok\n')
if __name__ == '__main__':
    main()
