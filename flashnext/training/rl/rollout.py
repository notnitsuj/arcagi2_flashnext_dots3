import asyncio, json, os, sys, tempfile, zlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'harness'))
import pool
EVAL_DIRS = [os.path.join(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'), 'evaluation')]

class Recorder:

    def __init__(self, ep):
        self.ep, self.log = (ep, [])

    def __getattr__(self, k):
        return getattr(self.ep, k)

    async def complete(self, body):
        text, finish, u = await self.ep.complete(body)
        self.log.append((body['prompt'], text, body.get('temperature')))
        return (text, finish, u)

async def sample_repl_rec(a, ep, tok, task, seed):
    rp = Recorder(ep)
    rec, text = await pool.run_sample(pool.SimpleNamespace(effort=a.effort, temp=a.temp, top_p=a.top_p, top_k=a.top_k, max_model_len=a.max_model_len, force_tokens=a.force_tokens, turn_tokens=a.turn_tokens, max_tool_calls=a.max_tool_calls, exec_timeout=a.exec_timeout, stop_on_verify=not a.no_stop_on_verify, exec_on_cut=True, keep_code=True, prompt=a.prompt), rp, tok, task, a.repl_cap, seed=seed)
    if not rp.log:
        return (rec, text)
    prompt = rp.log[0][0]
    full = prompt + text
    spans = []
    for p, t, temp in rp.log:
        s = len(p)
        if not full.startswith(p):
            continue
        n = 0
        while n < len(t) and s + n < len(full) and (full[s + n] == t[n]):
            n += 1
        if n:
            spans.append([s - len(prompt), s - len(prompt) + n, temp])
    rec['rl'] = {'prompt': prompt, 'spans': spans}
    return (rec, text)

def augment(d, seed):
    import random
    rng = random.Random(seed)
    t = rng.randrange(1, 8)
    perm = list(range(1, 10))
    rng.shuffle(perm)
    cmap = [0] + perm

    def g(x):
        x = [list(r) for r in x]
        if t & 1:
            x = [r[::-1] for r in x]
        if t & 2:
            x = x[::-1]
        if t & 4:
            x = [list(r) for r in zip(*x)]
        return [[cmap[v] for v in r] for r in x]
    return {k: [{kk: g(vv) for kk, vv in p.items()} for p in d[k]] for k in ('train', 'test')}

def main():
    argv = sys.argv[1:]
    excl, loo = ([], False)
    if '--exclude' in argv:
        i = argv.index('--exclude')
        excl = argv[i + 1].split(',')
        del argv[i:i + 2]
    naug, aoff = (0, 0)
    if '--aug-offset' in argv:
        i_ = argv.index('--aug-offset')
        aoff = int(argv[i_ + 1])
        del argv[i_:i_ + 2]
    if '--aug' in argv:
        i_ = argv.index('--aug')
        naug = int(argv[i_ + 1])
        del argv[i_:i_ + 2]
    loo_one = '--loo-one' in argv
    if loo_one:
        argv.remove('--loo-one')
    if '--loo' in argv:
        argv.remove('--loo')
        loo = True
    ev = set()
    for d in EVAL_DIRS:
        if Path(d).is_dir():
            ev |= {p.stem for p in Path(d).glob('*.json')}
    assert ev, 'evaluation task list not found (needed for the contamination guard)'
    drop = set()
    for f in excl:
        drop |= set(open(f).read().split())
    i = argv.index('--split')
    split = Path(argv[i + 1])
    assert 'evaluation' not in str(split), split
    j = argv.index('--tasks') if '--tasks' in argv else None
    if j is not None:
        t = argv[j + 1]
        sel = open(t[1:]).read().split() if t.startswith('@') else t.split(',')
    else:
        sel = [p.stem for p in split.glob('*.json')]
    sel = [x for x in sel if x.split('~')[0].split('@')[0] not in drop]
    bad = [x for x in sel if x.split('~')[0].split('@')[0] in ev]
    assert not bad, f'evaluation tasks in the rollout set: {bad[:5]}'
    tmp = Path(tempfile.mkdtemp(prefix='rlsplit_'))
    out_ids = []
    for x in sel:
        base = x.split('~')[0].split('@')[0]
        d0 = json.load(open(split / f'{base}.json'))
        variants = [(base, d0)] if not naug else [(f'{base}@a{a}', augment(d0, zlib.crc32(base.encode()) + 7919 * a)) for a in range(aoff, aoff + naug)]
        for vid, d in variants:
            if loo:
                if len(d['train']) < 3:
                    continue
                ks = range(len(d['train']))
                if loo_one:
                    ks = [zlib.crc32(base.encode()) % len(d['train'])]
                for k in ks:
                    v = {'train': [p for q, p in enumerate(d['train']) if q != k], 'test': [d['train'][k]]}
                    json.dump(v, open(tmp / f'{vid}~{k}.json', 'w'))
                    out_ids.append(f'{vid}~{k}')
            elif naug:
                json.dump(d, open(tmp / f'{vid}.json', 'w'))
                out_ids.append(vid)
            else:
                os.symlink(split / f'{base}.json', tmp / f'{base}.json')
                out_ids.append(base)
    (tmp / '_tasks.txt').write_text('\n'.join(out_ids) + '\n')
    argv[i + 1] = str(tmp)
    if j is not None:
        argv[j + 1] = '@' + str(tmp / '_tasks.txt')
    else:
        argv += ['--tasks', '@' + str(tmp / '_tasks.txt')]
    print(f'rollout: {len(out_ids)} tasks ({len(drop)} excluded ids, loo={loo}) split={tmp}', flush=True)
    pool.sample_repl = sample_repl_rec
    sys.argv = [sys.argv[0]] + argv
    asyncio.run(pool.main())
if __name__ == '__main__':
    main()
