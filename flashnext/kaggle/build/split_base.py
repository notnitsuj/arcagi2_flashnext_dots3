#!/usr/bin/env python3
import argparse, json, os, re, struct, sys, shutil
from pathlib import Path
EXP = re.compile('^(model\\.language_model\\.layers\\.\\d+|mtp\\.layers\\.\\d+)\\.mlp\\.experts\\.\\d+\\.')
NGRAM = re.compile('(^|\\.)(ple|ngram|n_gram|engram)[._]|ple_embedding|ngram_embedding', re.I)
SMALL_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.jinja', 'preprocessor_config.json', 'processor_config.json', 'special_tokens_map.json', 'added_tokens.json', 'LICENSE']

def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    return (h, 8 + n)

def classify(name):
    if EXP.match(name):
        return 'experts'
    if NGRAM.search(name):
        return 'ngram'
    return 'base'

def scan(src):
    idx = json.load(open(src / 'model.safetensors.index.json'))
    files = sorted(set(idx['weight_map'].values()))
    tensors = []
    for fn in files:
        h, off = header(src / fn)
        for name, t in h.items():
            if name == '__metadata__':
                continue
            s, e = t['data_offsets']
            tensors.append((classify(name), name, fn, off + s, off + e, t['dtype'], t['shape']))
    return tensors

def write_shards(tensors, src, out, prefix, shard_bytes):
    out.mkdir(parents=True, exist_ok=True)
    groups, cur, cur_b = ([], [], 0)
    for t in sorted(tensors, key=lambda t: t[1]):
        sz = t[4] - t[3]
        if cur and cur_b + sz > shard_bytes:
            groups.append(cur)
            cur, cur_b = ([], 0)
        cur.append(t)
        cur_b += sz
    if cur:
        groups.append(cur)
    wmap, total = ({}, 0)
    for i, g in enumerate(groups, 1):
        fn = f'{prefix}-{i:05d}-of-{len(groups):05d}.safetensors'
        hdr, o = ({}, 0)
        for _, name, _, s, e, dt, shape in g:
            hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [o, o + (e - s)]}
            o += e - s
        hdr['__metadata__'] = {'format': 'pt'}
        hb = json.dumps(hdr, separators=(',', ':')).encode()
        hb += b' ' * ((8 - len(hb) % 8) % 8)
        tmp = out / (fn + '.tmp')
        with open(tmp, 'wb') as w:
            w.write(struct.pack('<Q', len(hb)))
            w.write(hb)
            for _, name, sf, s, e, _, _ in g:
                with open(src / sf, 'rb') as r:
                    r.seek(s)
                    left = e - s
                    while left:
                        b = r.read(min(left, 64 << 20))
                        w.write(b)
                        left -= len(b)
                wmap[name] = fn
        os.replace(tmp, out / fn)
        total += o
        print(f'  {fn}: {len(g)} tensors {o / 1000000000.0:.2f} GB', flush=True)
    json.dump({'metadata': {'total_size': total}, 'weight_map': wmap}, open(out / f'{prefix}.index.json', 'w'), indent=1)
    return total

def merge_index(dirs, out_path):
    wmap, total = ({}, 0)
    for d in dirs:
        for ix in Path(d).glob('*.index.json'):
            j = json.load(open(ix))
            wmap.update(j['weight_map'])
            total += j['metadata']['total_size']
    json.dump({'metadata': {'total_size': total}, 'weight_map': wmap}, open(out_path, 'w'), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--shard-gb', type=float, default=4.0)
    a = ap.parse_args()
    src = Path(a.src)
    ts = scan(src)
    by = {}
    for t in ts:
        c = by.setdefault(t[0], [0, 0, set()])
        c[0] += 1
        c[1] += t[4] - t[3]
        c[2].add(t[5])
    for k, (n, b, dts) in sorted(by.items()):
        print(f'{k:8s} {n:7d} tensors {b / 2 ** 30:8.2f} GiB  dtypes {sorted(dts)}')
    ng = sorted({t[1].rsplit('.', 1)[0] for t in ts if t[0] == 'ngram'})
    print('ngram tensor groups:', ng[:12], '...' if len(ng) > 12 else '')
    base_names = [t[1] for t in ts if t[0] == 'base']
    print('base sample:', [n for n in base_names if 'layers.0.' in n][:30])
    print('base non-layer:', [n for n in base_names if '.layers.' not in n][:40])
    if a.analyze or not a.out:
        return
    out = Path(a.out)
    sb = int(a.shard_gb * 1000000000.0)
    print('writing base ...', flush=True)
    write_shards([t for t in ts if t[0] == 'base'], src, out, 'base', sb)
    print('writing ngram ...', flush=True)
    write_shards([t for t in ts if t[0] == 'ngram'], src, out, 'ngram', sb)
    for f in SMALL_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    print('done')
if __name__ == '__main__':
    sys.exit(main())
