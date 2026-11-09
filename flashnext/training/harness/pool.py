import argparse, asyncio, gzip, json, random, sys, time, zlib
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'qrepl'))
from session import Endpoint, run_sample, _state
from prompt import parse_grids, _valid
TEMPLATE = 'You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:'
FORCE = '\n\nTime is up; I will commit to my best hypothesis.\n</think>\n\nFinal answer:\n```json\n'
FORCE_AFTER = '\n\nFinal answer:\n```json\n'

def direct_prompt(task, ti):
    ex = ''
    for i, p in enumerate(task['train']):
        ex += f'--Example {i}-- \n\n INPUT: \n\n' + json.dumps(p['input']) + '\n\n'
        ex += 'OUTPUT: \n\n' + json.dumps(p['output']) + '\n\n'
    return TEMPLATE.format(training_examples=ex, test_input=json.dumps(task['test'][ti]['input']))

def last_grid(text):
    gs = parse_grids(text)
    return gs[-1] if gs else None

async def forced(ep, ctx, a):
    st = _state(ctx.rpartition('<|im_start|>assistant\n')[2]) if '</think>' in ctx else 'think'
    add = FORCE if st == 'think' else FORCE_AFTER
    text, _, u = await ep.complete({'prompt': ctx + add, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': a.force_tokens, 'stop': ['```']})
    return (last_grid('```json\n' + text + '\n```'), u['completion_tokens'])

async def sample_direct(a, ep, tok, task, ti, seed):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': direct_prompt(task, ti)}], tokenize=False, add_generation_prompt=True, reasoning_effort=a.effort)
    pt0 = len(tok(prompt, add_special_tokens=False).input_ids)
    ctx, used, ckpts = (prompt, 0, [])
    marks = sorted((int(x) for x in a.direct_ckpts.split(',') if x)) + [a.direct_cap]
    samp = {'temperature': a.temp, 'top_p': a.top_p, 'top_k': a.top_k, 'seed': seed}
    status, answer, n_think = ('cap', None, None)
    for m in marks:
        mt = min(m - used, a.max_model_len - pt0 - used - a.force_tokens - 64)
        if mt <= 0:
            break
        text, finish, u = await ep.complete({'prompt': ctx, 'max_tokens': mt, **samp})
        used += u['completion_tokens']
        ctx += text
        gen = ctx[len(prompt):]
        if n_think is None and '</think>' in gen:
            n_think = len(tok(gen.partition('</think>')[0], add_special_tokens=False).input_ids)
        if finish != 'length':
            if '</think>' in gen:
                status = 'answer'
                answer = last_grid(gen.rpartition('</think>')[2])
            else:
                status = 'eos_think'
                ctx = ctx.removesuffix('<|im_end|>')
                answer, n = await forced(ep, ctx, a)
                used += n
            break
        g, n = await forced(ep, ctx, a)
        ckpts.append({'at': used, 'grid': g, 'tokens': n})
    rec = {'used': used, 'pt0': pt0, 'status': status, 'n_think': n_think, 'answer': answer, 'ckpts': ckpts}
    return (rec, ctx[len(prompt):])

async def sample_repl(a, ep, tok, task, seed):
    cfg = SimpleNamespace(effort=a.effort, temp=a.temp, top_p=a.top_p, top_k=a.top_k, max_model_len=a.max_model_len, force_tokens=a.force_tokens, turn_tokens=a.turn_tokens, max_tool_calls=a.max_tool_calls, exec_timeout=a.exec_timeout, stop_on_verify=not a.no_stop_on_verify, exec_on_cut=True, keep_code=True, prompt=a.prompt)
    return await run_sample(cfg, ep, tok, task, a.repl_cap, seed=seed)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--tasks', default=None, help='comma list or @file')
    ap.add_argument('--arms', default='direct:4,repl:8', help='arm:k,...')
    ap.add_argument('--direct-cap', type=int, default=98304)
    ap.add_argument('--direct-ckpts', default='32768,65536')
    ap.add_argument('--repl-cap', type=int, default=98304)
    ap.add_argument('--turn-tokens', type=int, default=49152)
    ap.add_argument('--max-tool-calls', type=int, default=24)
    ap.add_argument('--exec-timeout', type=float, default=10.0)
    ap.add_argument('--no-stop-on-verify', action='store_true')
    ap.add_argument('--prompt', default='v1')
    ap.add_argument('--force-tokens', type=int, default=3072)
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--effort', default='xhigh')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--concurrency', type=int, default=64, help='per endpoint')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    import httpx
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    split = Path(a.split)
    ids = sorted((p.stem for p in split.glob('*.json')))
    if a.tasks:
        sel = set(open(a.tasks[1:]).read().split() if a.tasks.startswith('@') else a.tasks.split(','))
        ids = [i for i in ids if i in sel]
    tasks, gts = ({}, {})
    for t in ids:
        d = json.load(open(split / f'{t}.json'))
        gts[t] = [x['output'] for x in d['test']]
        tasks[t] = {'train': d['train'], 'test': [{'input': x['input']} for x in d['test']]}
    arms = dict(((x.split(':')[0], int(x.split(':')[1])) for x in a.arms.split(',')))
    done = set()
    if (out / 'samples.jsonl').exists():
        for l in open(out / 'samples.jsonl'):
            r = json.loads(l)
            if r['status'] != 'error':
                done.add((r['arm'], r['task'], r['test'], r['j']))
    jobs = []
    for j in range(max(arms.values())):
        lot = []
        for t in ids:
            if j < arms.get('repl', 0):
                lot.append(('repl', t, None, j))
            for ti in range(len(tasks[t]['test'])):
                if j < arms.get('direct', 0):
                    lot.append(('direct', t, ti, j))
        random.Random(a.seed * 1000 + j).shuffle(lot)
        jobs += [x for x in lot if x not in done]
    (out / 'args.json').write_text(json.dumps(vars(a) | {'n_tasks': len(ids), 'n_inputs': sum((len(g) for g in gts.values()))}, indent=1))
    eps_s = a.endpoints.split(',')
    print(f'{len(ids)} tasks, {len(jobs)} samples to run on {len(eps_s)} endpoints x {a.concurrency}', flush=True)
    client = httpx.AsyncClient(timeout=httpx.Timeout(6 * 3600.0), limits=httpx.Limits(max_connections=None))
    eps = [Endpoint(client, e) for e in eps_s]
    fs = open(out / 'samples.jsonl', 'a')
    ft = gzip.open(out / 'traces.jsonl.gz', 'at')
    q = asyncio.Queue()
    for x in jobs:
        q.put_nowait(x)
    t_start, n_done, gen = (time.time(), [0], [0])

    def seed_of(arm, t, ti, j):
        return zlib.crc32(json.dumps([a.seed, arm, t, ti, j]).encode()) & 2147483647

    async def worker(ep):
        while True:
            try:
                arm, t, ti, j = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.time()
            gt = gts[t]
            try:
                if arm == 'direct':
                    rec, text = await sample_direct(a, ep, tok, tasks[t], ti, seed_of(arm, t, ti, j))
                    rec['answer_correct'] = rec['answer'] is not None and rec['answer'] == gt[ti]
                    for c in rec['ckpts']:
                        c['correct'] = c['grid'] is not None and c['grid'] == gt[ti]
                else:
                    rec, text = await sample_repl(a, ep, tok, tasks[t], seed_of(arm, t, ti, j))
                    for p in rec['progs']:
                        p['tests_correct'] = [o is not None and o == g for o, g in zip(p['tests'], gt)]
                    rec['answer_correct'] = [g == gt[k] for k, g in enumerate(rec['answer'][:len(gt)])]
            except Exception as e:
                print('sample failed', arm, t, ti, j, repr(e)[:300], flush=True)
                rec, text = ({'used': 0, 'status': 'error', 'error': repr(e)[:300]}, '')
            rec.update(arm=arm, task=t, test=ti, j=j, n_test=len(gt), t0=t0, t1=time.time(), ep=ep.ep)
            fs.write(json.dumps(rec) + '\n')
            fs.flush()
            ft.write(json.dumps({'arm': arm, 'task': t, 'test': ti, 'j': j, 'text': text}) + '\n')
            ft.flush()
            n_done[0] += 1
            gen[0] += rec['used']
            if n_done[0] % 20 == 0:
                el = time.time() - t_start
                print(f'[{el:.0f}s] {n_done[0]}/{len(jobs)} done, {gen[0] / 1000000.0:.2f}M gen, {gen[0] / max(1, el):.0f} tok/s (completed samples)', flush=True)
    await asyncio.gather(*[worker(e) for e in eps for _ in range(a.concurrency)])
    fs.close()
    ft.close()
    print(f'done in {time.time() - t_start:.0f}s', flush=True)
if __name__ == '__main__':
    asyncio.run(main())
