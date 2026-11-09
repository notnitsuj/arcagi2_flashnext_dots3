import asyncio, os, re, time
from prompt import SYSTEM, TOOL, task_prompt, parse_grids, extract_code
from sandbox import run_code, feedback, verified
import sandbox as _sandbox
H2 = {x for x in os.environ.get('QR_H2', '').split(',') if x}
if H2 - {'novelty', 'flagfb', 'softnudge'}:
    raise SystemExit(f"QR_H2={os.environ.get('QR_H2')!r}: expected novelty, flagfb, softnudge")
H2_REVIEW_TOKENS = int(os.environ.get('QR_H2_REVIEW_TOKENS', '8192'))
H2_FLAG_TOKENS = int(os.environ.get('QR_H2_FLAG_TOKENS', '12288'))
H2_REVIEW_CALLS = int(os.environ.get('QR_H2_REVIEW_CALLS', '2'))
H2_REVIEW_END = os.environ.get('QR_H2_REVIEW_END', 'transform')
NOVELTY = 'transform() reproduces all training examples. Before finishing, look at the test input(s) only:\n1. List any structure, colour role, orientation, object shape or configuration in a test input that NO training example shows (e.g. a different orientation, a bent/diagonal element, an object touching the border, repeated colours).\n2. For each, say what your transform does there and whether an equally consistent alternative reading exists.\nIf there is a plausible alternative, define BOTH `transform` (your preferred reading) and `transform_alt` (the alternative) in one program and call the tool once more; otherwise reply DONE.'
SOFT_NUDGE = "\n\nWait. I have been reasoning for a while without running code. If I have a candidate rule, even a partial one, I should put it in transform() soon and let the checker's diff show what is left; if I already have a script that predicts outputs, I should wrap it in transform(grid) (each call starts fresh, so include all definitions). Continuing from where I was:\n\n"
ALT_RE = re.compile('^\\s*def\\s+transform_alt\\s*\\(', re.M)

def _flag_text(res, code):
    src = (code or '').splitlines()
    parts = []
    for k, ls in enumerate(res.get('ak_cov') or []):
        if ls:
            shown = '\n'.join(('    ' + src[i - 1].rstrip()[:120] for i in ls[:8] if 0 < i <= len(src)))
            parts.append(f"- Lines {', '.join(map(str, ls[:12]))} of transform run on test input {k + 1} but on no training input:\n{shown}\n  The examples never exercise this branch; make sure it implements the rule, not a fallback.")
    for k, ms in enumerate(res.get('h2_eq') or []):
        if ms:
            parts.append(f"- Your transform is consistent under {', '.join(ms[:4])} on every training example but not on test input {k + 1}.")
    if not parts:
        return ''
    return 'All training examples pass, but these checks suggest the rule may not transfer:\n' + '\n'.join(parts) + '\nIf you are confident, reply DONE. Otherwise fix transform (or add transform_alt for the other reading) and run it again.'
NUDGE = '\n\nI have thought about this long enough; let me test my current best hypothesis in code.\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n'
CLOSE_CALL = '\n</parameter>\n</function>\n</tool_call>'
FORCE = '\n\nTime is up; I will commit to my best prediction now.\n</think>\n\nFinal answer (one grid per test input):\n```json\n'
FORCE_AFTER_THINK = '\n\nFinal answer (one grid per test input):\n```json\n'

class Endpoint:

    def __init__(self, client, ep, model='m'):
        self.client, self.ep, self.model = (client, ep, model)

    async def complete(self, body):
        body = {'model': self.model, 'skip_special_tokens': False, **body}
        err = None
        for retry in range(5):
            try:
                r = await self.client.post(f'http://{self.ep}/v1/completions', json=body)
                j = r.json()
                return (j['choices'][0]['text'], j['choices'][0]['finish_reason'], j['usage'])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = e
                print('request error', self.ep, repr(e)[:200], flush=True)
                await asyncio.sleep(10 * (retry + 1))
        raise RuntimeError(f'request failed: {err!r}')

def _review(res, progs):
    if _sandbox.ARCKIT != 'review' or not any(res.get('ak_cov') or []) or any((p.get('review') for p in progs[:-1])):
        return False
    progs[-1]['review'] = True
    return True

def _state(seg):
    if '</think>' not in seg:
        return 'think'
    after = seg.rpartition('</think>')[2]
    i = after.rfind('<tool_call>')
    if i >= 0 and '</tool_call>' not in after[i:]:
        return 'call'
    return 'answer'

async def run_sample(cfg, endpoint, tok, task, cap, seed=None):
    t0 = time.time()
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task_prompt(task, getattr(cfg, 'prompt', 'v1'))}]
    prompt = tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, reasoning_effort=cfg.effort)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    pt0 = n_ctx = ntok(prompt)
    ctx, turn_start = (prompt, len(prompt))
    used = pre_new = n_req = n_tool = n_nudge = n_exec_err = 0
    progs, answer, status, forced = ([], [], None, False)
    samp = {'temperature': cfg.temp, 'top_p': cfg.top_p, 'top_k': cfg.top_k}
    if seed is not None:
        samp['seed'] = seed
    pending_exec = False
    rv, alts = (None, [])
    while True:
        if not pending_exec:
            remaining = cap - used
            if remaining <= 0:
                status = 'cap'
                break
            room = cfg.max_model_len - n_ctx - cfg.force_tokens * len(task['test']) - 64
            mt = min(remaining, cfg.turn_tokens, room)
            if rv is not None:
                mt = min(mt, rv['left'])
                if mt < 64:
                    status = 'reviewed'
                    break
            if mt < 128:
                status = 'ctx'
                break
            text, finish, u = await endpoint.complete({'prompt': ctx, 'max_tokens': mt, **samp, 'stop': ['</tool_call>'], 'include_stop_str_in_output': True})
            n_req += 1
            used += u['completion_tokens']
            n_ctx = u['prompt_tokens'] + u['completion_tokens']
            ctx += text
            seg = ctx[turn_start:]
            st = _state(seg)
            if rv is not None and finish == 'length' and (mt == rv['left']) and (mt < min(remaining, room)):
                status = 'reviewed'
                break
            if rv is not None:
                rv['left'] -= u['completion_tokens']
            if finish == 'length':
                if mt == remaining or mt == room:
                    if st == 'call' and cfg.exec_on_cut:
                        ctx += CLOSE_CALL
                        pending_exec = True
                        status = 'cap'
                    else:
                        status = 'cap' if mt == remaining else 'ctx'
                        break
                elif st == 'think' and rv is None:
                    nd = SOFT_NUDGE if 'softnudge' in H2 else NUDGE
                    ctx += nd
                    n_nudge += 1
                    pre_new += ntok(nd)
                    continue
                else:
                    continue
            if not pending_exec:
                if not seg.rstrip().endswith('</tool_call>'):
                    if st == 'call':
                        ctx = ctx.removesuffix('<|im_end|>') + CLOSE_CALL
                    elif st == 'think' and 'softnudge' in H2 and n_nudge and seg.endswith('<|im_end|>'):
                        ctx = ctx.removesuffix('<|im_end|>') + NUDGE
                        n_nudge += 1
                        pre_new += ntok(NUDGE)
                        continue
                    else:
                        answer = parse_grids(seg.rpartition('</think>')[2]) if '</think>' in seg else []
                        status = 'answer' if rv is None else 'reviewed'
                        if rv is not None:
                            rv['end'] = 'reply'
                        break
        pending_exec_was = pending_exec
        pending_exec = False
        code = extract_code(ctx[turn_start:])
        n_tool += 1
        res = None
        if code is None:
            resp = 'Error: no `code` parameter found in the call.'
        elif n_tool > cfg.max_tool_calls:
            resp = 'Tool budget exhausted. Give your final answer now: for each test input, the output grid in a ```json block.'
        else:
            res = await run_code(code, task, cfg.exec_timeout)
            resp = feedback(res, task, cfg.exec_timeout)
        if res is not None:
            n_exec_err += bool(res.get('error') or res.get('timeout'))
            if res.get('has_transform') and 'train' in res:
                ok = [bool(x.get('ok')) for x in res['train']]
                progs.append({'k': n_tool, 'train_ok': ok, 'n_ok': sum(ok), 'verified': verified(res), 'tests': [t.get('out') for t in res.get('tests', [])], 'used': used, 'code': code if cfg.keep_code else None, **({'ak_flag': res['ak_flag']} if 'ak_flag' in res else {})})
                if rv is not None:
                    progs[-1]['after_review'] = True
                if progs[-1]['verified'] and cfg.stop_on_verify and (not _review(res, progs)):
                    if rv is None and H2 & {'novelty', 'flagfb'} and (n_tool < cfg.max_tool_calls):
                        txt = ''
                        if 'flagfb' in H2:
                            txt = _flag_text(res, code)
                        if 'novelty' in H2:
                            txt = (txt + '\n\n' if txt else '') + NOVELTY
                        if txt:
                            rv = {'left': H2_REVIEW_TOKENS if 'novelty' in H2 else H2_FLAG_TOKENS, 'calls': 0, 'at': used, 'k': n_tool, 'end': None, 'flags': {'cov': res.get('ak_cov'), 'eq': res.get('h2_eq')}}
                            resp = resp + '\n\n' + txt
                    if rv is None:
                        status = 'verified'
                        break
            if rv is not None and rv['k'] != n_tool and code and ALT_RE.search(code):
                ra = await run_code(code + '\n\ntransform = transform_alt\n', task, cfg.exec_timeout)
                if ra.get('has_transform') and 'train' in ra:
                    ok = [bool(x.get('ok')) for x in ra['train']]
                    alts.append({'k': n_tool, 'train_ok': ok, 'n_ok': sum(ok), 'verified': verified(ra), 'tests': [t.get('out') for t in ra.get('tests', [])], 'used': used, 'code': code if cfg.keep_code else None})
        if rv is not None and rv['k'] != n_tool:
            rv['calls'] += 1
        if rv is not None and rv['k'] != n_tool and (rv['calls'] >= H2_REVIEW_CALLS or (code and (ALT_RE.search(code) if H2_REVIEW_END == 'alt' else 'def transform' in code))):
            status = 'reviewed'
            rv['end'] = 'call'
            break
        if pending_exec_was:
            break
        add = '<|im_end|>\n<|im_start|>user\n<tool_response>\n' + resp + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'
        k = ntok(add)
        pre_new += k
        n_ctx += k
        ctx += add
        turn_start = len(ctx)
    ver = [p for p in progs if p['verified']]
    if not ver and (not answer) and (cfg.force_tokens > 0) and (status in ('cap', 'ctx')):
        seg = ctx[turn_start:]
        st = _state(seg)
        if st != 'call':
            add = FORCE if st == 'think' else FORCE_AFTER_THINK
            text, _, u = await endpoint.complete({'prompt': ctx + add, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': cfg.force_tokens * len(task['test'])})
            n_req += 1
            used += u['completion_tokens']
            pre_new += ntok(add)
            ctx += add + text
            answer = parse_grids('```json\n' + text)
            forced = True
    rec = {'used': used, 'pt0': pt0, 'pre_new': pre_new, 'n_ctx': n_ctx, 'n_req': n_req, 'n_tool': n_tool, 'n_nudge': n_nudge, 'n_exec_err': n_exec_err, 'status': status, 'forced': forced, 'progs': progs, 'answer': answer, 't0': t0, 't1': time.time()}
    if rv is not None:
        rec['h2_review'] = {k: rv[k] for k in ('at', 'k', 'calls', 'end', 'flags')}
    if alts:
        rec['alts'] = alts
    return (rec, ctx[len(prompt):])

class ChatEndpoint(Endpoint):

    async def chat(self, body):
        body = {'model': self.model, **body}
        err = None
        for retry in range(5):
            try:
                r = await self.client.post(f'http://{self.ep}/v1/chat/completions', json=body)
                j = r.json()
                return (j['choices'][0]['message'], j['choices'][0]['finish_reason'], j['usage'])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = e
                print('request error', self.ep, repr(e)[:200], repr(j)[:300] if 'j' in dir() else '', flush=True)
                await asyncio.sleep(10 * (retry + 1))
        raise RuntimeError(f'request failed: {err!r}')
TURN_NUDGE = 'Your last turn ran out of tokens. Test your current best hypothesis now by calling the python tool.'
FORCE_MSG = 'Time is up. Without calling any tool, give your final answer now: for each test input in order, the output grid as JSON in its own ```json block.'

async def run_sample_chat(cfg, endpoint, tok, task, cap, seed=None):
    import json as _json
    t0 = time.time()
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task_prompt(task, getattr(cfg, 'prompt', 'v1'))}]
    ctk = _json.loads(cfg.chat_template_kwargs) if cfg.chat_template_kwargs else {}
    samp = {'temperature': cfg.temp, 'top_p': cfg.top_p}
    if cfg.top_k and cfg.top_k > 0:
        samp['top_k'] = cfg.top_k
    if seed is not None:
        samp['seed'] = seed
    used = n_req = n_tool = n_nudge = n_exec_err = 0
    pt0 = pre_last = n_ctx = 0
    pre_new = 0
    progs, answer, status, forced = ([], [], None, False)
    transcript = []
    while True:
        remaining = cap - used
        if remaining <= 0:
            status = 'cap'
            break
        mt = min(remaining, cfg.turn_tokens, max(128, cfg.max_model_len - n_ctx - 2048))
        msg, finish, u = await endpoint.chat({'messages': msgs, 'tools': [TOOL], 'max_tokens': mt, **samp, 'chat_template_kwargs': ctk})
        n_req += 1
        used += u['completion_tokens']
        if not pt0:
            pt0 = u['prompt_tokens']
        else:
            pre_new += max(0, u['prompt_tokens'] - pre_last)
        pre_last = u['prompt_tokens'] + u['completion_tokens']
        n_ctx = pre_last
        reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
        content = msg.get('content') or ''
        calls = msg.get('tool_calls') or []
        transcript.append({'reasoning': reasoning, 'content': content, 'tool_calls': calls})
        am = {'role': 'assistant', 'content': content}
        if cfg.keep_reasoning and reasoning:
            am['reasoning_content'] = reasoning
        if calls:
            am['tool_calls'] = calls
        msgs.append(am)
        if not calls:
            if finish == 'length':
                if mt == remaining:
                    status = 'cap'
                    break
                msgs.append({'role': 'user', 'content': TURN_NUDGE})
                n_nudge += 1
                continue
            answer = parse_grids(content)
            status = 'answer'
            break
        stop = False
        for c in calls:
            n_tool += 1
            try:
                code = _json.loads(c['function']['arguments']).get('code')
            except Exception:
                code = None
            res = None
            if c['function'].get('name') != 'python' or not code:
                resp = 'Error: call the `python` tool with a `code` argument.'
            elif n_tool > cfg.max_tool_calls:
                resp = 'Tool budget exhausted. Give your final answer now (one ```json grid per test input).'
            else:
                res = await run_code(code, task, cfg.exec_timeout)
                resp = feedback(res, task, cfg.exec_timeout)
            if res is not None:
                n_exec_err += bool(res.get('error') or res.get('timeout'))
                if res.get('has_transform') and 'train' in res:
                    ok = [bool(x.get('ok')) for x in res['train']]
                    progs.append({'k': n_tool, 'train_ok': ok, 'n_ok': sum(ok), 'verified': verified(res), 'tests': [t.get('out') for t in res.get('tests', [])], 'used': used, 'code': code if cfg.keep_code else None, **({'ak_flag': res['ak_flag']} if 'ak_flag' in res else {})})
                    if progs[-1]['verified'] and cfg.stop_on_verify and (not _review(res, progs)):
                        stop = True
            msgs.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'content': resp})
            transcript.append({'tool': resp})
            if stop:
                break
        if stop:
            status = 'verified'
            break
    if not any((p['verified'] for p in progs)) and (not answer) and (cfg.force_tokens > 0) and (status in ('cap', 'ctx')):
        if msgs[-1]['role'] == 'assistant' and msgs[-1].get('tool_calls'):
            msgs.append({'role': 'tool', 'tool_call_id': msgs[-1]['tool_calls'][-1].get('id', ''), 'content': '(not run: time is up)'})
        msgs.append({'role': 'user', 'content': FORCE_MSG})
        msg, _, u = await endpoint.chat({'messages': msgs, 'max_tokens': cfg.force_tokens * len(task['test']) + 1024, 'temperature': 0.0, 'chat_template_kwargs': ctk})
        n_req += 1
        used += u['completion_tokens']
        answer = parse_grids((msg.get('content') or '') + '\n' + (msg.get('reasoning_content') or ''))
        forced = True
        transcript.append({'forced': msg.get('content')})
    rec = {'used': used, 'pt0': pt0, 'pre_new': pre_new, 'n_ctx': n_ctx, 'n_req': n_req, 'n_tool': n_tool, 'n_nudge': n_nudge, 'n_exec_err': n_exec_err, 'status': status, 'forced': forced, 'progs': progs, 'answer': answer, 't0': t0, 't1': time.time()}
    return (rec, _json.dumps(transcript))
