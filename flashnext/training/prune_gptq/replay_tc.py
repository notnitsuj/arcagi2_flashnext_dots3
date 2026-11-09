import argparse, asyncio, gzip, json, os, time
import httpx
ap = argparse.ArgumentParser()
ap.add_argument('--traces', required=True, nargs='+')
ap.add_argument('--endpoint', required=True)
ap.add_argument('--model-dir', required=True)
ap.add_argument('--cap', required=True)
ap.add_argument('--max-calls', type=int, default=3, help='tool calls per trace (first N)')
ap.add_argument('--max-len', type=int, default=81000)
ap.add_argument('--conc', type=int, default=16)
ap.add_argument('--exclude-tasks', default=None)
ap.add_argument('--header', action='store_true', help="phases hd_pre (before <tool_call>) / hd_post (through '<parameter=code>' + newline) only")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model_dir)
ex = set(open(a.exclude_tasks).read().split()) if a.exclude_tasks else set()
reqs = {'hd_pre': [], 'hd_post': []} if a.header else {'tc_pre': [], 'tc_call': [], 'tc_resp': []}
n_tr = 0
for f in a.traces:
    try:
        for l in gzip.open(f, 'rt'):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                break
            if r['mode'] != 'repl' or r['task'] in ex:
                continue
            text = r['prompt'] + r['completion']
            p0 = len(r['prompt'])
            k = 0
            n_tr += 1
            while k < a.max_calls:
                s = text.find('<tool_call>', p0)
                if s < 0:
                    break
                e = text.find('</tool_call>', s)
                if e < 0:
                    break
                e += len('</tool_call>')
                q = text.find('</tool_response>', e)
                q = q + len('</tool_response>') if q >= 0 else None
                if a.header:
                    h = text.find('<parameter=code>', s)
                    if h < 0 or h > e:
                        p0 = e
                        continue
                    h += len('<parameter=code>') + (1 if text[h + len('<parameter=code>'):][:1] == '\n' else 0)
                    ids = [tok(text[:x], add_special_tokens=False).input_ids for x in (s, h)]
                    if len(ids[1]) > a.max_len:
                        break
                else:
                    ids = [tok(text[:x], add_special_tokens=False).input_ids for x in (s, e, q) if x is not None]
                    if len(ids) < 3 or len(ids[2]) > a.max_len:
                        break
                for ph, x in zip(reqs, ids):
                    reqs[ph].append(x)
                p0 = q if q is not None else e
                k += 1
    except (EOFError, OSError) as e:
        print('trace read stopped:', repr(e)[:100], flush=True)
print(f'{n_tr} repl traces, {len(next(iter(reqs.values())))} tool calls', flush=True)

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
    for ph, rs in reqs.items():
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
                    ids = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                tot[0] += await send(client, ids)
        await asyncio.gather(*[worker() for _ in range(a.conc)])
        print(f'phase {ph}: {len(rs)} requests, {tot[0]} tokens, {time.time() - t0:.0f}s', flush=True)
    os.remove(os.path.join(a.cap, 'on'))
    await send(client, tok('flush', add_special_tokens=False).input_ids)
    await asyncio.sleep(5)
    print('REPLAY_DONE', flush=True)
asyncio.run(main())
