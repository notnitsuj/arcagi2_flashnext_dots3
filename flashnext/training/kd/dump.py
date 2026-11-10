import argparse, asyncio, json, random, time
from pathlib import Path
import httpx
import numpy as np
K = 20

def parse(pl, plen, T):
    ti = np.full((T - plen, K), -1, np.int32)
    tl = np.full((T - plen, K), -10000.0, np.float16)
    for i in range(plen, T):
        d = pl[i]
        if d is None:
            continue
        items = sorted(((int(k), v['logprob'], v.get('rank') or 99) for k, v in d.items()), key=lambda x: x[2])
        tops = [x for x in items if x[2] <= K][:K]
        ti[i - plen, :len(tops)] = [x[0] for x in tops]
        tl[i - plen, :len(tops)] = [max(x[1], -10000.0) for x in tops]
    return (ti, tl)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recs')
    ap.add_argument('out')
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--conc', type=int, default=3, help='per endpoint')
    ap.add_argument('--max-len', type=int, default=65536)
    ap.add_argument('--split', default='train,val')
    ap.add_argument('--mode', default='repl,direct')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(a.recs)]
    recs = [r for r in recs if r['split'] in a.split.split(',') and r['mode'] in a.mode.split(',')]
    random.Random(0).shuffle(recs)
    recs.sort(key=lambda r: (r['mode'] != 'repl', r['split'] != 'val'))
    eps = a.endpoints.split(',')
    q = asyncio.Queue()
    for r in recs:
        if not (out / f"{r['key']}.npz").exists():
            q.put_nowait(r)
    print(f'{q.qsize()} to dump on {len(eps)} endpoints', flush=True)
    cl = httpx.AsyncClient(timeout=httpx.Timeout(7200.0), limits=httpx.Limits(max_connections=256))
    st = {'n': 0, 'tok': 0, 't0': time.time(), 'err': 0}

    async def worker(ep):
        while not q.empty():
            r = q.get_nowait()
            ids = r['ids'][:a.max_len]
            T, plen = (len(ids), r['plen'])
            if T <= plen + 16:
                continue
            try:
                resp = await cl.post(f'http://{ep}/v1/completions', json={'model': 'm', 'prompt': ids, 'max_tokens': 1, 'prompt_logprobs': K, 'temperature': 0})
                js = resp.json()
                pl = js['choices'][0]['prompt_logprobs']
            except Exception as e:
                st['err'] += 1
                print('ERR', ep, r['key'], T, repr(e)[:200], flush=True)
                continue
            ti, tl = await asyncio.to_thread(parse, pl, plen, T)
            gen = np.asarray([(x, min(y, T)) for x, y in r['gen'] if x < T], np.int32).reshape(-1, 2)
            np.savez(out / f"{r['key']}.npz", ids=np.asarray(ids, np.int32), plen=np.asarray(plen), gen=gen, top_ids=ti, top_lp=tl)
            st['n'] += 1
            st['tok'] += T
            el = time.time() - st['t0']
            print(f"{st['n']} {r['key']} T={T} {st['tok'] / el:.0f} tok/s err={st['err']}", flush=True)
    await asyncio.gather(*[worker(ep) for ep in eps for _ in range(a.conc)])
    print('DONE', st, flush=True)
if __name__ == '__main__':
    asyncio.run(main())
