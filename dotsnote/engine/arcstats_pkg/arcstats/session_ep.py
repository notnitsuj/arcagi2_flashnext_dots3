"""ARC_SESSION_EP=1: HTTP routes for vLLM streaming-input sessions, so one attempt keeps its KV across input chunks.

A session generates in chunks (max_tokens each). At a chunk end the engine keeps the request's KV blocks and waits for
the next input; the client then either pushes the chunk's last sampled token back (plain continuation: vLLM drops that
token from the context when a chunk ends) or pushes [last token] + a forcing text (forced conclusion with the whole
reasoning in context and only the forcing text to prefill), or closes the session. Without this, a forced conclusion
is a new request whose prompt is not a prefix-cache hit under load (hybrid SWA cache), i.e. a ~14k-token re-prefill.

  POST /arc/session        {"rid", "prompt_ids", "max_tokens", "temperature", "top_p", "seed"}
       -> NDJSON stream: {"ids": [...], "text": "..."} per delta, {"chunk_end": reason, "n": tokens} per chunk,
          {"end": reason} once the session is over
  POST /arc/session/push   {"rid", "prompt_ids", "max_tokens"[, "temperature", "top_p", "stop_token_ids"]}
  "keep": true (on the start request or a push; applies until the next push) keeps the session open after that chunk
  stops on EOS / a stop token instead of ending it, so the client can push one more chunk (solve.py --alt session);
  the client must then push or close. Without "keep" a stop ends the session (unchanged).
  POST /arc/session/close  {"rid"}                                          -> end after the current chunk
Disconnecting the stream aborts the session (KV freed).
Scheduler fixes for sessions under async scheduling + MTP are always installed with the endpoint; ARC_SESSION_FIX=0
leaves out _fix_stale_inflight (stale in-flight output, preemption, KV deadlock) for A/B repros only.
"""
import asyncio
import json
import os

_SESSIONS = {}   # rid -> asyncio.Queue of StreamingInput | None


class SessionPlugin:
    name = "arcsession"
    required_tasks = None

    def attach_router(self, app):
        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse

        @app.post("/arc/session")
        async def session(req: Request):
            from vllm import SamplingParams
            from vllm.engine.protocol import StreamingInput
            from vllm.sampling_params import RequestOutputKind
            b = await req.json()
            eng = req.app.state.arc_engine
            rid = str(b["rid"])
            q = asyncio.Queue()
            _SESSIONS[rid] = q

            def sp(d, n):
                return SamplingParams(max_tokens=int(n), temperature=float(d.get("temperature", 1.0)),
                                      top_p=float(d.get("top_p", 1.0)), seed=d.get("seed"),
                                      stop_token_ids=d.get("stop_token_ids"), output_kind=RequestOutputKind.DELTA,
                                      skip_special_tokens=False)
            base = dict(temperature=b.get("temperature", 1.0), top_p=b.get("top_p", 1.0), seed=b.get("seed"))
            await q.put(StreamingInput(prompt={"prompt_token_ids": list(b["prompt_ids"])},
                                       sampling_params=sp(base, b["max_tokens"])))
            _SESSIONS[rid + "#sp"] = (sp, base)
            _SESSIONS[rid + "#keep"] = bool(b.get("keep"))

            async def inputs():
                while True:
                    x = await q.get()
                    if x is None:
                        return
                    yield x

            async def body():
                n = 0
                try:
                    async for out in eng.generate(inputs(), sp(base, b["max_tokens"]), rid):
                        o = out.outputs[0]
                        if o.token_ids:
                            n += len(o.token_ids)
                            yield json.dumps({"ids": list(o.token_ids), "text": o.text}) + "\n"
                        if o.finish_reason is not None:
                            yield json.dumps({"chunk_end": o.finish_reason, "n": n}) + "\n"
                            if o.finish_reason != "length" and not _SESSIONS.get(rid + "#keep"):
                                await q.put(None)              # EOS: nothing to continue (unless "keep")
                        if out.finished:
                            break
                    yield json.dumps({"end": "done", "n": n}) + "\n"
                finally:
                    _SESSIONS.pop(rid, None); _SESSIONS.pop(rid + "#sp", None); _SESSIONS.pop(rid + "#keep", None)
            return StreamingResponse(body(), media_type="application/x-ndjson")

        @app.post("/arc/session/push")
        async def push(req: Request):
            from vllm.engine.protocol import StreamingInput
            b = await req.json()
            rid = str(b["rid"])
            q = _SESSIONS.get(rid)
            if q is None:
                return JSONResponse({"ok": False, "error": "no such session"}, status_code=404)
            sp, base = _SESSIONS[rid + "#sp"]
            d = dict(base, **{k: b[k] for k in ("temperature", "top_p", "stop_token_ids") if k in b})
            _SESSIONS[rid + "#keep"] = bool(b.get("keep"))
            await q.put(StreamingInput(prompt={"prompt_token_ids": list(b["prompt_ids"])},
                                       sampling_params=sp(d, b["max_tokens"])))
            return JSONResponse({"ok": True})

        @app.post("/arc/session/close")
        async def close(req: Request):
            b = await req.json()
            q = _SESSIONS.get(str(b["rid"]))
            if q is not None:
                await q.put(None)
            return JSONResponse({"ok": q is not None})

    async def init_state(self, engine_client, state, args):
        state.arc_engine = engine_client


def _fix_scheduler():
    """vLLM's session update assumes num_computed_tokens counts only real tokens. With MTP drafts and async scheduling it
    also counts the stopped step's draft slots (placeholders), so the next chunk had <= 0 new tokens (scheduler assert).
    Clamp to the tokens whose KV is really written (all but the last sampled token) and drop the stale drafts.
    Also apply the pushed chunk's max_tokens (vLLM keeps the first chunk's)."""
    from vllm.v1.core.sched import scheduler as sch
    orig = sch.Scheduler._update_request_as_session

    def _update_request_as_session(self, session, update):
        real = len(session._all_token_ids) - 1
        if os.environ.get("ARC_SESSION_DEBUG"):
            print(f"[arcsession] {session.request_id} status={session.status} computed={session.num_computed_tokens} "
                  f"all={len(session._all_token_ids)} prompt={session.num_prompt_tokens} "
                  f"ph={session.num_output_placeholders} inflight={getattr(session, 'num_in_flight_tokens', None)} "
                  f"spec={len(session.spec_token_ids)} new={len(update.prompt_token_ids or ())} "
                  f"max_tokens={session.max_tokens}/{update.max_tokens} out={session.num_output_tokens}", flush=True)
        session.num_computed_tokens = min(session.num_computed_tokens - session.num_output_placeholders, real)
        session.num_output_placeholders = 0
        session.spec_token_ids = []
        r = orig(self, session, update)
        session.max_tokens = update.max_tokens   # this chunk's budget (vLLM keeps the first chunk's)
        return r

    sch.Scheduler._update_request_as_session = _update_request_as_session

    # A step scheduled ahead (async scheduling) can still deliver tokens after a chunk ended. For a finished request
    # vLLM drops them; a parked session is not finished, so they were appended and stopped the session a second time.
    ou = sch.Scheduler._update_request_with_output

    def _update_request_with_output(self, request, new_token_ids, *a, **kw):
        if request.resumable and request.status == sch.RequestStatus.WAITING_FOR_STREAMING_REQ:
            return [], False
        return ou(self, request, new_token_ids, *a, **kw)

    sch.Scheduler._update_request_with_output = _update_request_with_output
    if os.environ.get("ARC_SESSION_FIX", "1") == "1":
        _fix_stale_inflight(sch)
    if os.environ.get("ARC_SESSION_DEBUG"):
        oc = sch.check_stop

        def check_stop(request, max_model_len):
            r = oc(request, max_model_len)
            if r and request.resumable:
                print(f"[arcsession] stop {request.request_id} status={request.status} out={request.num_output_tokens} "
                      f"max={request.max_tokens} ntok={request.num_tokens} computed={request.num_computed_tokens}",
                      flush=True)
            return r
        sch.check_stop = check_stop


_STATS = {"stop_inflight": 0, "update_inflight": 0, "preempt": 0, "clamped": 0, "evicted": 0, "t": 0.0}


def _stat(key):
    import time
    _STATS[key] += 1
    if time.time() - _STATS["t"] > 300:
        _STATS["t"] = time.time()
        print("[arcsession] stale-drop stats " + " ".join(f"{k}={v}" for k, v in _STATS.items() if k != "t"),
              flush=True)


def _fix_stale_inflight(sch):
    """Async scheduling + MTP: when a session chunk stops at step N, step N+1 is usually already in flight. Its output
    used to be dropped only if it arrived while the session was still parked (WAITING_FOR_STREAMING_REQ). If the next
    input was applied first -- a push that lands before N+1's output is processed (run 67, 3.2 h in), or one queued
    while the chunk was still running, which vLLM applies inside the stop handling (deterministic local repro:
    dots3/session_stress.py --p-early) -- the session update zeroed num_output_placeholders and the stale output was
    appended to the new chunk and decremented the placeholders below zero (AsyncScheduler assert -> EngineDeadError). A preempted session could likewise be stopped by its stale output,
    after which vLLM removes it from every queue (the next push then leaves it unscheduled forever).
    Fix: mark every in-flight step of a session as stale drop-mode output (vLLM's own mechanism, used for same-step
    resume after reset_prefix_cache) when the chunk stops, when a push is applied, and when the session is preempted;
    update_from_output then discards those outputs before touching any counter. Last line of defence: a resumable
    request's placeholder underflow is clamped and logged instead of killing the engine."""
    from vllm.v1.core.sched import async_scheduler as asch

    def _mark_stale(request, key):
        if request.num_in_flight_tokens > 0:
            request.num_stale_output_tokens = request.num_in_flight_tokens
            request.drop_stale_output = True
            _stat(key)

    hs = sch.Scheduler._handle_stopped_request

    def _handle_stopped_request(self, request):
        if request.resumable:
            _mark_stale(request, "stop_inflight")
        return hs(self, request)

    sch.Scheduler._handle_stopped_request = _handle_stopped_request

    us = sch.Scheduler._update_request_as_session

    def _update_request_as_session(self, session, update):
        _mark_stale(session, "update_inflight")
        return us(self, session, update)

    sch.Scheduler._update_request_as_session = _update_request_as_session

    pr = sch.Scheduler._preempt_request

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if request.resumable:
            drop_stale_output = True
            _stat("preempt")
        return pr(self, request, timestamp, drop_stale_output=drop_stale_output)

    sch.Scheduler._preempt_request = _preempt_request

    ao = asch.AsyncScheduler._update_request_with_output

    def _async_update(self, request, new_token_ids, is_stale=False):
        if not request.resumable:
            return ao(self, request, new_token_ids, is_stale)
        status_before_update = request.status
        new_token_ids, stopped = sch.Scheduler._update_request_with_output(self, request, new_token_ids)
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            if request.num_output_placeholders < 0:
                print(f"[arcsession] WARNING placeholder underflow {request.request_id} "
                      f"ph={request.num_output_placeholders} status={status_before_update} -> clamped", flush=True)
                request.num_output_placeholders = 0
                _stat("clamped")
        if status_before_update == sch.RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders)
        return new_token_ids, stopped

    asch.AsyncScheduler._update_request_with_output = _async_update

    # KV deadlock: a session keeps its KV blocks while it waits for its next chunk (parked or queued after a push),
    # and vLLM only preempts RUNNING requests. With the pool full and every session waiting, the waiting head cannot
    # allocate (the waiting loop just breaks) and nothing runs again (local stress: 0 running / 64 waiting at KV 0.94
    # forever). When a step schedules nothing while nothing runs, evict the KV of the lowest-priority waiting session
    # (newest first, queued sessions before parked ones): it recomputes from its token ids like a preempted request.
    sc = sch.Scheduler.schedule

    def schedule(self, *a, **kw):
        out = sc(self, *a, **kw)
        if out.total_num_scheduled_tokens == 0 and not self.running and (self.waiting or self.skipped_waiting):
            victims = [r for q in (self.waiting, self.skipped_waiting) for r in q
                       if r.resumable and r.num_computed_tokens > 0 and not r.is_finished()]
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
                print(f"[arcsession] KV deadlock: evicted waiting session {r.request_id} ({r.status.name}, "
                      f"{len(r._all_token_ids)} tokens) for recompute", flush=True)
                _stat("evicted")
        return out

    sch.Scheduler.schedule = schedule


def install():
    """Add the plugin to vLLM's endpoint-plugin list without a packaging entry point (no reinstall of the env)."""
    import vllm.plugins as vp
    orig = vp.load_endpoint_plugins

    def load_endpoint_plugins(supported_tasks=None):
        return list(orig(supported_tasks)) + [SessionPlugin()]

    vp.load_endpoint_plugins = load_endpoint_plugins
    _fix_scheduler()
