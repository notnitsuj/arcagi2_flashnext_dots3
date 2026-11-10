import argparse, asyncio, json, os
from pathlib import Path
import httpx
import numpy as np
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
EVAL = {p.stem for p in (DATA / 'evaluation').glob('*.json')}
K = 20

def parse(pl, ids):
    L = len(ids)
    top_ids = np.full((L, K), -1, np.int32)
    top_lp = np.full((L, K), -1000000000.0, np.float32)
    for i, d in enumerate(pl):
        if d is None:
            continue
        items = sorted(((int(k), v['logprob'], v.get('rank', 99)) for k, v in d.items()), key=lambda x: x[2])
        tops = [x for x in items if x[2] <= K][:K]
        top_ids[i, :len(tops)] = [x[0] for x in tops]
        top_lp[i, :len(tops)] = [x[1] for x in tops]
    return (top_ids, top_lp)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('port', type=int)
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--len', type=int, default=12288)
    ap.add_argument('--conc', type=int, default=4)
    ap.add_argument('--tag', default='t')
    ap.add_argument('--tasks', default=None, help='@file: only these task ids')
    ap.add_argument('--tokenizer', default=os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
    a = ap.parse_args()
    assert EVAL, f'no evaluation tasks under {DATA}: set ARC_DATA'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sel = set(open(a.tasks[1:]).read().split()) if a.tasks else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(a.src)]
    recs = [r for r in recs if sel is None or r['task'] in sel]
    assert not any((r['task'] in EVAL for r in recs)), 'evaluation-split task in distillation data'
    sem = asyncio.Semaphore(a.conc)
    cl = httpx.AsyncClient(timeout=httpx.Timeout(7200.0))
    done = [0]

    async def one(r):
        fn = out / f"{r['task']}_{r['test']}_{r['attempt']}_{a.tag}.npz"
        if fn.exists():
            return
        plen = len(tok(r['prompt'], add_special_tokens=False).input_ids)
        ids = tok(r['prompt'] + r['completion'], add_special_tokens=False).input_ids[:a.len]
        if len(ids) <= plen + 16:
            return
        async with sem:
            js = (await cl.post(f'http://127.0.0.1:{a.port}/v1/completions', json={'model': 'm', 'prompt': ids, 'max_tokens': 1, 'prompt_logprobs': K, 'temperature': 0})).json()
        ti, tl = parse(js['choices'][0]['prompt_logprobs'], ids)
        np.savez_compressed(fn, ids=np.asarray(ids, np.int32), plen=np.asarray(plen), top_ids=ti, top_lp=tl)
        done[0] += 1
        print(done[0], fn.name, len(ids), flush=True)
    await asyncio.gather(*[one(r) for r in recs])
if __name__ == '__main__':
    asyncio.run(main())
