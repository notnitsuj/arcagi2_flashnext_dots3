import argparse, asyncio, gzip, json, os, time
import httpx
ap = argparse.ArgumentParser()
ap.add_argument('--traces', required=True, nargs='+')
ap.add_argument('--endpoint', required=True)
ap.add_argument('--model-dir', required=True)
ap.add_argument('--cap', required=True)
ap.add_argument('--holdout-file', required=True)
ap.add_argument('--max-len', type=int, default=81000)
ap.add_argument('--conc', type=int, default=16)
ap.add_argument('--tag', default='', help='suffix of the capture subdirs (tr<tag>, ho<tag>)')
ap.add_argument('--split-pos', type=int, default=0, help='if > 0: traces longer than this are replayed twice, prefix [:P] into <ph>_pre and full into <ph>; late-position routing stats = <ph> - <ph>_pre (prefix tokens of the full replay = <ph>_pre)')
ap.add_argument('--done-file', default=None, help='keys already replayed (skipped) / appended after this run')
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model_dir)
ho = set(open(a.holdout_file).read().split())
key = lambda r: f"{r['mode']}|{r['task']}|{r['test']}|{r['attempt']}"
done = set(open(a.done_file).read().split()) if a.done_file and os.path.exists(a.done_file) else set()
rows = []
for f in a.traces:
    try:
        for l in gzip.open(f, 'rt'):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                break
            if key(r) not in done:
                rows.append(r)
    except (EOFError, OSError) as e:
        print('trace read stopped:', repr(e)[:100], flush=True)
phases = {'tr' + a.tag: [r for r in rows if r['task'] not in ho], 'ho' + a.tag: [r for r in rows if r['task'] in ho]}
for r in rows:
    r['ids'] = tok(r['prompt'] + r['completion'], add_special_tokens=False).input_ids[:a.max_len]
if a.split_pos:
    for ph in list(phases):
        longs = [dict(r, ids=r['ids'][:a.split_pos]) for r in phases[ph] if len(r['ids']) > a.split_pos]
        phases[ph + '_pre'] = longs
    print('split', {k: len(v) for k, v in phases.items()}, flush=True)
print({k: len(v) for k, v in phases.items()}, flush=True)

async def send(client, ids):
    body = {'model': 'm', 'prompt': ids, 'max_tokens': 1, 'temperature': 0.0}
    for i in range(5):
        try:
            r = await client.post(f'http://{a.endpoint}/v1/completions', json=body)
            r.raise_for_status()
            return r.json()['usage']['prompt_tokens']
        except Exception as e:
            print('retry', repr(e)[:200], flush=True)
            await asyncio.sleep(5)
    return 0

async def main():
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=None))
    await send(client, tok('warmup', add_special_tokens=False).input_ids)
    for ph, rs in phases.items():
        with open(os.path.join(a.cap, 'on'), 'w') as f:
            f.write(ph)
        q = asyncio.Queue()
        for r in rs:
            q.put_nowait(r)
        tot = [0]
        t0 = time.time()

        async def worker():
            while True:
                try:
                    r = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                tot[0] += await send(client, r['ids'])
                if q.qsize() % 50 == 0:
                    print(f'{ph}: {len(rs) - q.qsize()}/{len(rs)} {tot[0]} tok {tot[0] / (time.time() - t0):.0f} tok/s', flush=True)
        await asyncio.gather(*[worker() for _ in range(a.conc)])
        print(f'phase {ph} done: {tot[0]} tokens in {time.time() - t0:.0f}s', flush=True)
    os.remove(os.path.join(a.cap, 'on'))
    if a.done_file:
        with open(a.done_file, 'a') as f:
            f.write('\n'.join((key(r) for r in rows)) + '\n')
    import collections
    print('length hist (k tokens):', sorted(collections.Counter((len(r['ids']) // 10000 * 10 for r in rows)).items()), flush=True)
    await send(client, tok('flush', add_special_tokens=False).input_ids)
    await asyncio.sleep(5)
    print('REPLAY_DONE', flush=True)
asyncio.run(main())
