import hashlib, json, os, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
sys.path.insert(0, str(Path(__file__).resolve().parent))

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()

def trained_tensors(ckpt, base, model_dir):
    from mtp_torch import MTP
    m = MTP(model_dir, base, torch.device('cuda'))
    orig = {k: v.detach().cpu().clone() for k, v in m.state_delta().items()}
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('params', sd)
    miss = m.load_trained(sd)
    assert not miss, miss
    new = m.state_delta()
    dense = {k: v for k, v in new.items() if not torch.equal(v, orig[k])}
    scales = router = None
    if 'moe.tgu' in sd or 'moe.td' in sd:
        d = torch.load(f'{base}/lowbit-L48.pt', map_location='cpu', weights_only=False)
        scales = {}
        for i, e in enumerate(d['experts']):
            sgu, sd_ = m.moe.scales(i)
            scales[e] = {'gu_s': sgu.detach().half().cpu(), 'd_s': sd_.detach().half().cpu()}
    if 'moe.gate' in sd:
        router = torch.load(f'{base}/router-L48.pt', map_location='cpu', weights_only=False)
        router['weight'] = m.moe.gate.detach().to(torch.bfloat16).cpu()
    return (dense, scales, router)

def make(ckpt, base, model_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dense, scales, router = trained_tensors(ckpt, base, model_dir)
    if dense:
        save_file(dense, str(out / 'mtp_ovr.safetensors'))
    if scales is not None:
        torch.save(scales, out / 'scales-L48.pt')
    if router is not None:
        torch.save(router, out / 'router-L48.pt')
    man = {f.name: {'md5': md5(f), 'bytes': f.stat().st_size} for f in sorted(out.iterdir()) if f.name != 'manifest.json'}
    man['_dense_tensors'] = sorted(dense)
    json.dump(man, open(out / 'manifest.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in man.items() if not k.startswith('_')}, indent=1))
    print('dense tensors:', len(dense), 'total MB', sum((v['bytes'] for k, v in man.items() if not k.startswith('_'))) / 1000000.0)

def build(base, delta, out):
    base, delta, out = (Path(base), Path(delta), Path(out))
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name in ('lowbit-L48.pt', 'router-L48.pt') or not f.is_file():
            continue
        dst = out / f.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(f.resolve(), dst)
        except OSError:
            shutil.copy2(f, dst)
    b = torch.load(base / 'lowbit-L48.pt', map_location='cpu', weights_only=False)
    if (delta / 'scales-L48.pt').exists():
        sc = torch.load(delta / 'scales-L48.pt', map_location='cpu', weights_only=False)
        assert list(sc) == list(b['experts']), 'L48: expert ids differ'
        for e, v in sc.items():
            assert v['gu_s'].shape == b[e]['gu_s'].shape and v['d_s'].shape == b[e]['d_s'].shape
            b[e]['gu_s'], b[e]['d_s'] = (v['gu_s'], v['d_s'])
    torch.save(b, out / 'lowbit-L48.pt')
    shutil.copy2(delta / 'router-L48.pt' if (delta / 'router-L48.pt').exists() else base / 'router-L48.pt', out / 'router-L48.pt')
    if (delta / 'mtp_ovr.safetensors').exists():
        shutil.copy2(delta / 'mtp_ovr.safetensors', out / 'mtp_ovr.safetensors')
    print('built', out)

def check(built, ckpt, base, model_dir):
    dense, scales, router = trained_tensors(ckpt, base, model_dir)
    built = Path(built)
    ok = True
    if dense:
        o = load_file(str(built / 'mtp_ovr.safetensors'))
        ok &= set(o) == set(dense) and all((torch.equal(o[k], dense[k]) for k in dense))
    b = torch.load(built / 'lowbit-L48.pt', map_location='cpu', weights_only=False)
    if scales is not None:
        ok &= all((torch.equal(b[e]['gu_s'], v['gu_s']) and torch.equal(b[e]['d_s'], v['d_s']) for e, v in scales.items()))
    r = torch.load(built / 'router-L48.pt', map_location='cpu', weights_only=False)
    if router is not None:
        ok &= torch.equal(r['weight'], router['weight'])
    for l in range(48):
        for nm in (f'lowbit-L{l:02d}.pt', f'router-L{l:02d}.pt'):
            ok &= os.path.samefile(built / nm, Path(base) / nm) or md5(built / nm) == md5(Path(base) / nm)
    print('CHECK', 'PASS' if ok else 'FAIL')
    return ok
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make':
        make(*sys.argv[2:6])
    elif cmd == 'build':
        build(*sys.argv[2:5])
    elif cmd == 'check':
        sys.exit(0 if check(*sys.argv[2:6]) else 1)
