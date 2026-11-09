import asyncio
import json
import os
_SESSIONS = {}
_STATS = {'sessions': 0, 'open': 0, 'pushes': 0, 'pushed_tokens': 0, 'start_tokens': 0, 'generated': 0, 'evicted': 0, 'ctx_full': 0, 'push_refused': 0}

def _guard():
    return os.environ.get('FN_SESSION_CTXGUARD', '1') == '1'

def _max_len(req):
    m = getattr(req.app.state, 'fn_max_len', None)
    return int(m) if m else 1 << 30

def _sp(d, n):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    kw = dict(max_tokens=int(n), temperature=float(d.get('temperature', 1.0)), top_p=float(d.get('top_p', 1.0)), seed=d.get('seed'), stop_token_ids=d.get('stop_token_ids') or None, output_kind=RequestOutputKind.DELTA, skip_special_tokens=False)
    if d.get('top_k') is not None:
        kw['top_k'] = int(d['top_k'])
    if d.get('logprobs') is not None:
        kw['logprobs'] = int(d['logprobs'])
    if d.get('ignore_eos'):
        kw['ignore_eos'] = True
    return SamplingParams(**kw)
_KEYS = ('temperature', 'top_p', 'top_k', 'seed', 'stop_token_ids', 'logprobs', 'ignore_eos')

class SessionPlugin:
    name = 'fnsession'
    required_tasks = None

    def attach_router(self, app):
        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse

        @app.post('/arc/session')
        async def session(req: Request):
            from vllm.engine.protocol import StreamingInput
            b = await req.json()
            eng = req.app.state.fn_engine
            rid = str(b['rid'])
            q = asyncio.Queue()
            _SESSIONS[rid] = q
            base = {k: b[k] for k in _KEYS if b.get(k) is not None}
            _SESSIONS[rid + '#base'] = base
            _SESSIONS[rid + '#keep'] = bool(b.get('keep'))
            _SESSIONS[rid + '#held'] = len(b['prompt_ids'])
            _STATS['sessions'] += 1
            _STATS['open'] += 1
            _STATS['start_tokens'] += len(b['prompt_ids'])
            await q.put(StreamingInput(prompt={'prompt_token_ids': list(b['prompt_ids'])}, sampling_params=_sp(base, b['max_tokens'])))

            async def inputs():
                while True:
                    x = await q.get()
                    if x is None:
                        return
                    yield x
            want_lp = base.get('logprobs') is not None

            async def body():
                n = 0
                n_chunk = 0
                end = 'done'
                try:
                    async for out in eng.generate(inputs(), _sp(base, b['max_tokens']), rid):
                        o = out.outputs[0]
                        if o.token_ids:
                            n += len(o.token_ids)
                            n_chunk += len(o.token_ids)
                            _SESSIONS[rid + '#held'] = _SESSIONS.get(rid + '#held', 0) + len(o.token_ids)
                            _STATS['generated'] += len(o.token_ids)
                            ev = {'ids': list(o.token_ids), 'text': o.text}
                            if want_lp and o.logprobs:
                                ev['lp'] = [float(lp[t].logprob) if t in lp else None for t, lp in zip(o.token_ids, o.logprobs[-len(o.token_ids):])]
                            yield (json.dumps(ev) + '\n')
                        if o.finish_reason is not None:
                            if n_chunk:
                                _SESSIONS[rid + '#held'] = _SESSIONS.get(rid + '#held', 0) - 1
                            n_chunk = 0
                            full = o.stop_reason == 'ctx_full' or (_guard() and o.finish_reason == 'length' and (_SESSIONS.get(rid + '#held', 0) >= _max_len(req) - 1))
                            ev = {'chunk_end': o.finish_reason, 'n': n}
                            if full:
                                ev['ctx_full'] = True
                            yield (json.dumps(ev) + '\n')
                            if full:
                                _STATS['ctx_full'] += 1
                                end = 'ctx_full'
                                _SESSIONS[rid + '#full'] = True
                                await q.put(None)
                            elif o.finish_reason != 'length' and (not _SESSIONS.get(rid + '#keep')):
                                await q.put(None)
                        if out.finished:
                            break
                    yield (json.dumps({'end': 'ctx_full' if _SESSIONS.get(rid + '#full') else end, 'n': n}) + '\n')
                finally:
                    _STATS['open'] -= 1
                    for k in (rid, rid + '#base', rid + '#keep', rid + '#held', rid + '#full'):
                        _SESSIONS.pop(k, None)
            return StreamingResponse(body(), media_type='application/x-ndjson')

        @app.post('/arc/session/push')
        async def push(req: Request):
            from vllm.engine.protocol import StreamingInput
            b = await req.json()
            rid = str(b['rid'])
            q = _SESSIONS.get(rid)
            if q is None:
                return JSONResponse({'ok': False, 'error': 'no such session'}, status_code=404)
            ids = list(b['prompt_ids'])
            if _SESSIONS.get(rid + '#full') or (_guard() and _SESSIONS.get(rid + '#held', 0) + len(ids) > _max_len(req) - 1):
                _STATS['push_refused'] += 1
                if not _SESSIONS.get(rid + '#full'):
                    _SESSIONS[rid + '#full'] = True
                    await q.put(None)
                return JSONResponse({'ok': False, 'error': 'ctx_full', 'held': _SESSIONS.get(rid + '#held'), 'max_model_len': _max_len(req)}, status_code=409)
            base = _SESSIONS[rid + '#base']
            d = dict(base, **{k: b[k] for k in _KEYS if k in b and k != 'seed'})
            _SESSIONS[rid + '#keep'] = bool(b.get('keep'))
            _STATS['pushes'] += 1
            _STATS['pushed_tokens'] += len(ids)
            _SESSIONS[rid + '#held'] = _SESSIONS.get(rid + '#held', 0) + len(ids)
            await q.put(StreamingInput(prompt={'prompt_token_ids': ids}, sampling_params=_sp(d, b['max_tokens'])))
            return JSONResponse({'ok': True})

        @app.post('/arc/session/close')
        async def close(req: Request):
            b = await req.json()
            q = _SESSIONS.get(str(b['rid']))
            if q is not None:
                await q.put(None)
            return JSONResponse({'ok': q is not None})

        @app.get('/arc/session/stats')
        async def stats():
            return JSONResponse(dict(_STATS))

    async def init_state(self, engine_client, state, args):
        state.fn_engine = engine_client
        mc = getattr(engine_client, 'model_config', None) or getattr(getattr(engine_client, 'vllm_config', None), 'model_config', None)
        state.fn_max_len = getattr(mc, 'max_model_len', None)
        n = getattr(args, 'api_server_count', 1) or 1
        if n > 1:
            print(f'[fnsession] WARNING api_server_count={n}: pushes can land on another API server (404 -> the client restarts the session with a full prefill). Serve sessions with --api-server-count 1 (DP engines keep session affinity inside one API server).', flush=True)

def _evict_on_deadlock():
    from vllm.v1.core.sched import scheduler as sch
    sc = sch.Scheduler.schedule

    def schedule(self, *a, **kw):
        out = sc(self, *a, **kw)
        if out.total_num_scheduled_tokens == 0 and (not self.running) and (self.waiting or getattr(self, 'kv_holding_waiting', None)):
            pools = [self.waiting] + [x for x in (getattr(self, 'kv_holding_waiting', None), getattr(self, 'skipped_waiting', None)) if x is not None]
            victims = [r for p in pools for r in p if r.resumable and r.num_computed_tokens > 0 and (not r.is_finished())]
            victims.sort(key=lambda r: (r.status == sch.RequestStatus.WAITING_FOR_STREAMING_REQ, -r.arrival_time))
            if victims:
                r = victims[0]
                self._free_request_blocks(r)
                r.num_computed_tokens = 0
                r.spec_token_ids = []
                if r.num_in_flight_tokens > 0:
                    r.num_stale_output_tokens = r.num_in_flight_tokens
                    r.drop_stale_output = True
                r.num_output_placeholders = 0
                r.num_preemptions += 1
                _STATS['evicted'] += 1
                print(f'[fnsession] KV deadlock: evicted waiting session {r.request_id}', flush=True)
        return out
    sch.Scheduler.schedule = schedule

def _ctx_guard():
    from vllm.v1.core.sched import scheduler as sch
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    S = sch.Scheduler
    RS = sch.RequestStatus
    upd0, ufo0, hr0 = (S._update_request_as_session, S.update_from_output, S.has_requests)

    def _update_request_as_session(self, session, update):
        n_new = len(update.prompt_token_ids or ()) if update is not None else 0
        if update is None or session.num_computed_tokens + n_new < self.max_model_len:
            return upd0(self, session, update)
        upd0(self, session, sch.replace(update, prompt_token_ids=[], mm_features=None))
        session.status = RS.WAITING_FOR_STREAMING_REQ
        self.num_waiting_for_streaming_input += 1
        pend = self.__dict__.setdefault('_fn_ctx_full', [])
        pend.append(session)
        _STATS['ctx_full'] += 1
        print(f'[fnsession] ctx_full: session {session.request_id} at {session.num_computed_tokens} tokens + {n_new} pushed >= max_model_len {self.max_model_len}: push not applied, chunk ended', flush=True)

    def update_from_output(self, *a, **kw):
        out = ufo0(self, *a, **kw)
        pend = self.__dict__.get('_fn_ctx_full')
        while pend:
            r = pend.pop(0)
            if r.is_finished() or r.request_id not in self.requests:
                continue
            eco = out.get(r.client_index)
            if eco is None:
                eco = out[r.client_index] = EngineCoreOutputs(outputs=[])
            eco.outputs.append(EngineCoreOutput(request_id=r.request_id, new_token_ids=[], finish_reason=FinishReason.LENGTH, stop_reason='ctx_full', events=r.take_events(), trace_headers=r.trace_headers))
            if r.streaming_queue:
                nxt = r.streaming_queue.popleft()
                if nxt is None:
                    self.finish_requests(r.request_id, RS.FINISHED_ABORTED)
                elif r.status == RS.WAITING_FOR_STREAMING_REQ:
                    self._update_request_as_session(r, nxt)
        return out

    def has_requests(self):
        return bool(self.__dict__.get('_fn_ctx_full')) or hr0(self)
    S._update_request_as_session = _update_request_as_session
    S.update_from_output = update_from_output
    S.has_requests = has_requests

def install():
    import vllm.plugins as vp
    if not getattr(vp.load_endpoint_plugins, '_fnsession', False):
        orig = vp.load_endpoint_plugins

        def load_endpoint_plugins(supported_tasks=None):
            return list(orig(supported_tasks)) + [SessionPlugin()]
        load_endpoint_plugins._fnsession = True
        vp.load_endpoint_plugins = load_endpoint_plugins
    if os.environ.get('FN_SESSION_EVICT') == '1':
        from vllm.v1.core.sched import scheduler as sch
        if not getattr(sch.Scheduler, '_fnsession_evict', False):
            _evict_on_deadlock()
            sch.Scheduler._fnsession_evict = True
    from . import resume_fix
    resume_fix.install()
    if os.environ.get('FN_SESSION_CTXGUARD', '1') in ('1', '2'):
        from vllm.v1.core.sched import scheduler as sch
        if not getattr(sch.Scheduler, '_fnsession_ctxguard', False):
            _ctx_guard()
            sch.Scheduler._fnsession_ctxguard = True
