import argparse, asyncio, gzip, hashlib, json, random, re, sys, time
from pathlib import Path
import httpx
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from direct import make_prompt, parse
from sandbox import run_code, feedback
FORCE_PREFIX = '\n\nTime is up; I will commit to my best hypothesis.\n</think>\n\nFinal answer:\n```json\n[['
TOOL = {'type': 'function', 'function': {'name': 'python', 'description': "Run Python 3 code (numpy available as np) in a fresh sandbox; state does not persist between calls. Pre-loaded: `train`, a list of {'input': grid, 'output': grid} training examples, and `test_input`, the test grid (grids are lists of lists of ints). Printed output is returned. If the code defines `transform(grid)`, it is run on every training input and the result is compared with the expected output (cells that differ are listed), and transform(test_input) is shown.", 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string', 'description': 'Python code'}}, 'required': ['code']}}}
SYS_REPL = 'You are an expert at solving ARC abstract-reasoning puzzles, and you have a python tool. Work out the transformation rule from the training examples and express it as a Python function `transform(grid)` (grid: list of lists of ints; return the output grid). Test it with the python tool: the tool runs transform on every training input, reports which examples match and which cells differ, and shows transform(test_input). Fix the program until it reproduces every training example exactly. Then give the final answer: the test output grid as JSON in a ```json block.'

def parse_partial(txt):
    try:
        g = json.loads(txt.split('```')[0].strip())
        if isinstance(g, list) and g and all((isinstance(r, list) and r and all((isinstance(v, int) for v in r)) for r in g)):
            return g
    except Exception:
        pass
    rows = re.findall('\\[\\s*(-?\\d+(?:\\s*,\\s*-?\\d+)*)\\s*\\]', txt)
    rows = [[int(v) for v in r.split(',')] for r in rows]
    if not rows:
        return None
    w = len(rows[0])
    out = []
    for r in rows:
        if len(r) != w:
            break
        out.append(r)
    return out if out and len(out) <= 30 and (w <= 30) else None

def extract_code(call):
    i = call.rfind('<tool_call>')
    t = call[i:] if i >= 0 else call
    m = re.search('<parameter=code>\\n?', t)
    if not m:
        m2 = re.search('```python\\n(.*?)```', t, re.S)
        return m2.group(1) if m2 else None
    t = t[m.end():]
    cut = [t.find(x) for x in ('</parameter>', '</function>', '</tool_call>')]
    cut = [c for c in cut if c >= 0]
    if cut:
        t = t[:min(cut)]
    t = t.rstrip('\n')
    t = re.sub('^\\s*```(?:python)?\\s*\\n', '', t)
    t = re.sub('\\n```\\s*$', '', t)
    return t if t.strip() else None

async def post(client, ep, body):
    for retry in range(4):
        try:
            r = (await client.post(f'http://{ep}/v1/completions', json=body)).json()
            return (r, r['choices'][0])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print('error', ep, repr(e)[:200], repr(r)[:300] if 'r' in dir() else '', flush=True)
            await asyncio.sleep(20)
    raise RuntimeError('request failed')

def sampling(a):
    return {'temperature': a.temp, 'top_p': a.top_p, 'top_k': a.top_k, 'skip_special_tokens': False}

async def force(a, client, ep, ctx):
    b2 = {'model': 'm', 'prompt': ctx + FORCE_PREFIX, 'max_tokens': a.force_tokens, 'temperature': 0.0, 'top_p': 1.0, 'stop': ['```'], 'skip_special_tokens': False}
    r2, c2 = await post(client, ep, b2)
    return (parse_partial('[[' + c2['text']), FORCE_PREFIX + c2['text'], r2['usage'])

async def attempt_direct(a, client, ep, tok, task, ti):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': make_prompt(task, ti)}], tokenize=False, add_generation_prompt=True, reasoning_effort=a.effort)
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    body = {'model': 'm', 'prompt': prompt, **sampling(a), 'max_tokens': min(a.max_tokens, a.max_model_len - n_prompt - a.force_tokens - 64)}
    r, c = await post(client, ep, body)
    text = c['text']
    rec = {'ct': r['usage']['completion_tokens'], 'pt_sum': r['usage']['prompt_tokens'], 'pt0': n_prompt, 'n_calls': 1, 'n_tool': 0, 'finish': c['finish_reason'], 'forced': False, 'progs': []}
    if '</think>' in text:
        grid = parse(text.rpartition('</think>')[2])
    else:
        grid, add, u = await force(a, client, ep, prompt + text)
        text += add
        rec.update(forced=True, n_calls=2, forced_ct=u['completion_tokens'])
        rec['ct'] += u['completion_tokens']
        rec['pt_sum'] += u['prompt_tokens']
    rec['grid'] = grid
    return (rec, prompt, text)

async def attempt_repl(a, client, ep, tok, task, ti):
    msgs = [{'role': 'system', 'content': SYS_REPL}, {'role': 'user', 'content': make_prompt(task, ti)}]
    prompt = tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, reasoning_effort=a.effort)
    n_ctx = pt0 = len(tok(prompt, add_special_tokens=False).input_ids)
    ctx = prompt
    turn_start = len(ctx)
    used = pt_sum = n_calls = n_tool = resp_tok = exec_err = 0
    progs = []
    finish = None
    while True:
        remaining = a.max_tokens - used
        if remaining <= 0:
            finish = 'length'
            break
        mt = max(1, min(remaining, a.max_model_len - n_ctx - a.force_tokens - 64))
        body = {'model': 'm', 'prompt': ctx, **sampling(a), 'max_tokens': mt, 'stop': ['</tool_call>'], 'include_stop_str_in_output': True}
        r, c = await post(client, ep, body)
        n_calls += 1
        g, p = (r['usage']['completion_tokens'], r['usage']['prompt_tokens'])
        used += g
        pt_sum += p
        n_ctx = p + g
        text = c['text']
        ctx += text
        finish = c['finish_reason']
        if finish == 'length':
            if mt < remaining:
                finish = 'ctx'
            break
        seg = ctx[turn_start:]
        if seg.rstrip().endswith('</tool_call>') or (finish == 'stop' and '<tool_call>' in seg.rpartition('</think>')[2] and ('</tool_call>' not in seg[seg.rfind('<tool_call>'):])):
            if not ctx.endswith('</tool_call>'):
                ctx = ctx.removesuffix('<|im_end|>') + '\n</tool_call>'
            code = extract_code(ctx[turn_start:])
            n_tool += 1
            res = None
            if code is None:
                resp = 'Error: no `code` parameter found in the call.'
            elif n_tool > a.max_tool_calls:
                resp = 'Tool budget exhausted. Give your final answer now (the test output grid in a ```json block).'
            else:
                res = await run_code(code, task, ti, a.exec_timeout)
                resp = feedback(res, a.exec_timeout)
            if res is not None:
                exec_err += bool(res.get('error') or res.get('timeout'))
                if res.get('has_transform') and 'train' in res:
                    progs.append({'k': n_tool, 'train_ok': [bool(x.get('ok')) for x in res['train']], 'test_out': res.get('test_out'), 'used': used})
            add = '<|im_end|>\n<|im_start|>user\n<tool_response>\n' + resp + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'
            ntok = len(tok(add, add_special_tokens=False).input_ids)
            resp_tok += ntok
            n_ctx += ntok
            ctx += add
            turn_start = len(ctx)
            continue
        break
    seg = ctx[turn_start:]
    forced = False
    if '</think>' in seg and finish == 'stop':
        ans = seg.rpartition('</think>')[2].split('<tool_call>')[0]
        grid = parse(ans) if ans.strip() else None
    else:
        grid, add, u = await force(a, client, ep, ctx)
        ctx += add
        n_calls += 1
        used += u['completion_tokens']
        pt_sum += u['prompt_tokens']
        forced = True
    rec = {'ct': used, 'pt_sum': pt_sum, 'pt0': pt0, 'n_calls': n_calls, 'n_tool': n_tool, 'tool_resp_tok': resp_tok, 'finish': finish, 'forced': forced, 'grid': grid, 'progs': progs, 'exec_err': exec_err}
    return (rec, prompt, ctx[len(prompt):])

def finalize(rec, gt):
    grid = rec.pop('grid')
    rec['pred'] = grid
    rec['parsed'] = grid is not None
    rec['grid_correct'] = grid == gt
    progs = rec.get('progs') or []
    for pg in progs:
        pg['test_correct'] = pg['test_out'] == gt
        pg['verified'] = bool(pg['train_ok']) and all(pg['train_ok']) and (pg['test_out'] is not None)
        pg['agree_grid'] = pg['test_out'] is not None and pg['test_out'] == grid
    ver = [pg for pg in progs if pg['verified']]
    sub = ver[-1]['test_out'] if ver else grid
    rec['sub_from'] = 'prog' if ver else 'grid'
    rec['sub'] = sub
    rec['sub_correct'] = sub == gt and sub is not None
    rec['any_verified'] = bool(ver)
    rec['n_prog'] = len(progs)
    for pg in progs:
        pg['test_out_h'] = None if pg['test_out'] is None else hashlib.md5(json.dumps(pg['test_out']).encode()).hexdigest()[:12]
        pg.pop('test_out', None)
    return rec

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--model-dir', required=True, help='tokenizer / chat template')
    ap.add_argument('--split', required=True)
    ap.add_argument('--tasks', default=None, help='comma list or @file')
    ap.add_argument('--modes', default='direct,repl')
    ap.add_argument('--attempts', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=98304)
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--force-tokens', type=int, default=3000)
    ap.add_argument('--max-tool-calls', type=int, default=12)
    ap.add_argument('--exec-timeout', type=float, default=10.0)
    ap.add_argument('--effort', default='xhigh')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    eps = a.endpoints.split(',')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    split = Path(a.split)
    ids = sorted((p.stem for p in split.glob('*.json')))
    if a.tasks:
        sel = set(open(a.tasks[1:]).read().split() if a.tasks.startswith('@') else a.tasks.split(','))
        ids = [i for i in ids if i in sel]
    tasks = {i: json.load(open(split / f'{i}.json')) for i in ids}
    modes = a.modes.split(',')
    done = set()
    if (out / 'results.jsonl').exists():
        for l in open(out / 'results.jsonl'):
            r = json.loads(l)
            done.add((r['mode'], r['task'], r['test'], r['attempt']))
    jobs = [(m, t, ti, at) for t in ids for at in range(a.attempts) for ti in range(len(tasks[t]['test'])) for m in modes if (m, t, ti, at) not in done]
    random.Random(a.seed).shuffle(jobs)
    jobs.sort(key=lambda j: j[3])
    print(f'{len(jobs)} attempts over {len(eps)} servers x {a.concurrency}', flush=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    client = httpx.AsyncClient(timeout=httpx.Timeout(6 * 3600.0), limits=httpx.Limits(max_connections=None))
    res = open(out / 'results.jsonl', 'a')
    txt = gzip.open(out / 'traces.jsonl.gz', 'at')
    q = asyncio.Queue()
    for j in jobs:
        q.put_nowait(j)
    t_start = time.time()
    n_done = [0]

    async def worker(ep):
        while True:
            try:
                mode, tid, ti, att = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.time()
            try:
                fn = attempt_direct if mode == 'direct' else attempt_repl
                rec, prompt, completion = await fn(a, client, ep, tok, tasks[tid], ti)
            except Exception as e:
                print('attempt failed', mode, tid, ti, att, repr(e)[:300], flush=True)
                continue
            gt = tasks[tid]['test'][ti]['output']
            rec = {'mode': mode, 'task': tid, 'test': ti, 'attempt': att, 't0': t0, 't1': time.time(), **finalize(rec, gt)}
            res.write(json.dumps(rec) + '\n')
            res.flush()
            txt.write(json.dumps({k: rec[k] for k in ('mode', 'task', 'test', 'attempt')} | {'prompt': prompt, 'completion': completion}) + '\n')
            txt.flush()
            n_done[0] += 1
            if n_done[0] % 10 == 0:
                print(f'{n_done[0]} done, {time.time() - t_start:.0f}s', flush=True)
    await asyncio.gather(*[worker(e) for e in eps for _ in range(a.concurrency)])
    txt.close()
    print(f'done in {time.time() - t_start:.0f}s', flush=True)
if __name__ == '__main__':
    asyncio.run(main())
