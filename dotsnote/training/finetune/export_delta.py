import argparse, json, os, re, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.join(os.environ.get('D3_ENGINE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine')), 'arcstats_pkg'))
from arcstats import delta_overlay as DO
NAMES = {'ln1': 'input_layernorm.weight', 'ln2': 'post_attention_layernorm.weight', 'attn.q_ln': 'self_attn.q_a_layernorm.weight', 'attn.kv_ln': 'self_attn.kv_a_layernorm.weight', 'attn.kr_ln': 'self_attn.k_rope_only_layernorm.weight', 'mlp.gate': 'mlp.gate.weight'}

def student_scales(base, t):
    return torch.stack([(base[i].float() * t[i].float().exp()).half() for i in range(base.shape[0])])

def dense_targets(sd, base_ckpt):
    from safetensors import safe_open
    wm = json.load(open(base_ckpt / 'model.safetensors.index.json'))['weight_map']
    out = {}
    for k, v in sd.items():
        if k.endswith('.tgu') or k.endswith('.td'):
            continue
        if k == 'norm':
            name = 'model.norm.weight'
        else:
            m = re.match('layers\\.(\\d+)\\.(.+)$', k)
            assert m and m.group(2) in NAMES, k
            name = f'model.layers.{m.group(1)}.{NAMES[m.group(2)]}'
        with safe_open(str(base_ckpt / wm[name]), 'pt') as f:
            b = f.get_tensor(name)
        out[name] = v.to(b.dtype).contiguous()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--base', default=os.environ.get('D3_LB144R', os.path.join(os.environ.get('D3_WORK', 'work'), 'lb144r')))
    ap.add_argument('--out', default=None)
    ap.add_argument('--check', nargs='?', const='', default=None, help='verify DELTA (optionally vs a student ckpt)')
    a = ap.parse_args()
    base = Path(a.base)
    lbdir = base / json.load(open(base / 'config.json')).get('arc_lowbit', 'lowbit')
    out = Path(a.out or Path(a.run) / 'delta.safetensors')
    sd = torch.load(Path(a.run) / 'trained.pt', map_location='cpu')
    if a.check is None:
        T = {}
        nchg = ntot = 0
        for l in range(1, 46):
            d = torch.load(lbdir / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
            k = f'lowbit.L{l:02d}.'
            T[k + 'experts'] = torch.tensor(list(d['experts']), dtype=torch.int64)
            cks = []
            for nm, tk in (('gu_s', f'layers.{l}.mlp.tgu'), ('d_s', f'layers.{l}.mlp.td')):
                b = torch.stack([d[e][nm] for e in d['experts']])
                cks.append(DO.checksum(b))
                new = student_scales(b, sd[tk])
                diff = DO._bits(new) - DO._bits(b)
                assert (new > 0).all() and (b > 0).all()
                dt = torch.int8 if diff.abs().max() <= 127 else torch.int16
                T[k + nm] = diff.to(dt).contiguous()
                nchg += int((diff != 0).sum())
                ntot += diff.numel()
            T[k + 'base_ck'] = torch.tensor(cks, dtype=torch.int64)
        for name, v in dense_targets(sd, base).items():
            T['dense.' + name] = v
        save_file(T, str(out), metadata={'base': str(base.name), 'run': str(a.run), 'format': 'arc_delta_v1'})
        print(f'wrote {out}: {out.stat().st_size / 2 ** 30:.2f} GiB; scales changed {nchg}/{ntot} ({nchg / ntot:.1%}); int16 layers: {sum((v.dtype == torch.int16 for v in T.values()))}')
        return
    os.environ['ARC_DELTA_FILE'] = str(out)
    stud = Path(a.check) if a.check else None
    bad = 0
    for l in range(1, 46):
        d = torch.load(lbdir / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
        ref = {nm: student_scales(torch.stack([d[e][nm] for e in d['experts']]), sd[f'layers.{l}.mlp.{tk}']) for nm, tk in (('gu_s', 'tgu'), ('d_s', 'td'))}
        assert DO.apply_lowbit(l, d) == 1
        S = torch.load(stud / 'lowbit' / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False) if stud else None
        for nm in ('gu_s', 'd_s'):
            got = torch.stack([d[e][nm] for e in d['experts']])
            bad += not torch.equal(got.view(torch.int16), ref[nm].view(torch.int16))
            if S is not None:
                bad += not torch.equal(got.view(torch.int16), torch.stack([S[e][nm] for e in S['experts']]).view(torch.int16))
                for e in d['experts']:
                    for f in ('gu_z', 'd_z', 'bits'):
                        x, y = (d[e][f], S[e][f])
                        bad += not (torch.equal(x, y) if torch.is_tensor(x) else x == y)
    from safetensors import safe_open
    tg = dense_targets(sd, base)
    wm = json.load(open(base / 'model.safetensors.index.json'))['weight_map']

    def stream(ck):
        for fn in sorted(set(wm.values())):
            with safe_open(str(ck / fn), 'pt') as f:
                for k in f.keys():
                    yield (k, f.get_tensor(k))
    n = 0
    sstream = dict(stream(stud)) if stud else None
    for name, t in DO.wrap_weights(stream(base)):
        if name in tg:
            n += 1
            bv = lambda x: x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x
            bad += not (t.dtype == tg[name].dtype and torch.equal(bv(t), bv(tg[name])))
        if sstream is not None:
            u = sstream[name]
            bad += not (t.dtype == u.dtype and t.shape == u.shape and torch.equal(t.view(torch.uint8), u.view(torch.uint8)))
    print(f"check: 45 lowbit layers, {n}/{len(tg)} dense tensors, student ckpt {('compared' if stud else 'not given')}: {('OK (bit-identical)' if bad == 0 else f'{bad} MISMATCHES')}")
if __name__ == '__main__':
    main()
