"""ARC_SHARD_SAMPLE=1: speculative-decoding verification on the vocab-parallel logits shard, without the all-gather.

Stock vLLM (v2 model runner, TP4) all-gathers the target logits of every verify step ([4*64, 152064] bf16: each L4
receives ~58 MB over PCIe, NCCL AllGather ~3.6 ms), then re-lays them out (0.6 ms), copies them to fp32 (1.0 ms),
applies the temperature in place (1.4 ms), reduces per-block max/sum-exp (0.6 ms) and Gumbel-resamples (0.2 ms), the
same on every rank: ~7.4 ms per decode step on L4 (L4 profile).

Here each rank reduces its own vocab shard in one pass ([256, 38016] bf16) to per-row statistics:
  m, s   max and sum(exp(x - m)) of x = logit / T (the probability of the draft token needs the global logsumexp),
  gv, gi Gumbel-max over x with the row's draft token excluded (the residual sampled on rejection / at the bonus
         position): vLLM's own noise (`gumbel_noised_argmax`, Philox keyed by seed, position and global token id), so
         every element gets the identical noised value as on the full vocab,
  av, ai plain argmax (greedy rows),
  dl     x of the draft token (one rank owns it),
all-gathers those 8 floats per row (8 KB per rank) and finishes the stock rejection algorithm (Leviathan et al. with a
one-hot greedy draft, `_rejection_kernel` + `_resample_kernel` + `_insert_resampled_kernel`) on every rank.

Exactness: resampled / bonus / greedy tokens are bit-identical to stock (same values, same lowest-index tie break). The
acceptance test `logit_d/T - logsumexp > log(u)` uses a logsumexp summed in a different order (per shard, then across
shards, instead of over 8192-wide blocks of the full vocab), so it can flip only when log(u) falls within ~1 ulp of
the threshold.

Taken only for verify steps whose requests use plain temperature sampling (no logprobs, penalties, logit bias, bad
words, min_p / top-k / top-p, thinking budget, grammar, watermark, probabilistic drafts, block verification); any
other step runs the stock path. ARC_SHARD_SAMPLE_CHECK=N: the first N sharded steps also run the stock path (with the
all-gather) and compare tokens / counts (log only).  ARC_SHARD_SAMPLE_AG=nccl|host (default host when ARC_HOST_AR).
"""
import os

import numpy as np
import torch
import triton
import triton.language as tl

from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32

CHECK = int(os.environ.get("ARC_SHARD_SAMPLE_CHECK", "0"))
AG = os.environ.get("ARC_SHARD_SAMPLE_AG", "auto")
BLOCK = int(os.environ.get("ARC_SHARD_SAMPLE_BLOCK", "2048"))
WARPS = int(os.environ.get("ARC_SHARD_SAMPLE_WARPS", "8"))
NST = 8   # floats per row: m, s, gv, gi, av, ai, dl, pad
# ARC_SHARD_SAMPLE_NOSYNC=1: excl_tokens marks the bonus rows with index_fill_ instead of `nxt[last] = -1`. The latter
# is index_put_ with a Python scalar: torch stages the scalar as a 4-byte pageable H2D copy, i.e. cudaMemcpyAsync +
# cudaStreamSynchronize, which blocks the worker's main thread once per decode step until the whole target forward has
# run and starves the GPU while the CPU launches sampling, drafts and the next step's input prep (Kaggle nsys: ~14 ms
# of GPU idle per 256 ms step). Same values, so bit-identical outputs.
# ARC_NOSYNC_AB=P (A/B diagnostics only): alternate P-second windows of time since ARC_T0, even = nosync, odd = stock.
NOSYNC = os.environ.get("ARC_SHARD_SAMPLE_NOSYNC", "0") == "1"
_NOSYNC_AB = float(os.environ.get("ARC_NOSYNC_AB", "0"))
_NOSYNC_T0 = float(os.environ.get("ARC_T0", "0"))
_NOSYNC_PH = [None]


def _nosync():
    if _NOSYNC_AB <= 0:
        return NOSYNC
    import time
    t = time.time() - _NOSYNC_T0
    ph = int(t // _NOSYNC_AB) % 2
    if ph != _NOSYNC_PH[0]:
        _NOSYNC_PH[0] = ph
        print(f"[nosync_ab] phase {ph} {'sync' if ph else 'nosync'} t={t:.1f}", flush=True)
    return ph == 0


@triton.jit
def _row_stats_kernel(logits_ptr, logits_stride, out_ptr, excl_ptr, eidx_ptr, temp_ptr, seed_ptr, pos_ptr, n_local,
                      v0, BLOCK: tl.constexpr):
    """One program per logits row: statistics of this rank's vocab shard (global ids v0 .. v0 + n_local - 1)."""
    row = tl.program_id(0).to(tl.int64)
    rs = tl.load(eidx_ptr + row).to(tl.int64)
    valid_req = rs >= 0
    temp = tl.load(temp_ptr + rs, mask=valid_req, other=0.0).to(tl.float32)
    seed = tl.load(seed_ptr + rs, mask=valid_req, other=0)
    pos = tl.load(pos_ptr + row)
    excl = tl.load(excl_ptr + row)            # global id of the row's draft token, -1 = none (bonus / invalid draft)
    offs0 = tl.arange(0, BLOCK)
    m = -float("inf")
    s = 0.0
    gv = -float("inf")
    gi = 0
    dl = -float("inf")
    for start in range(0, n_local, BLOCK):
        offs = start + offs0
        msk = offs < n_local
        gid = v0 + offs
        lg = tl.load(logits_ptr + row * logits_stride + offs, mask=msk, other=float("-inf")).to(tl.float32)
        if temp == 0.0:
            # greedy: plain argmax of the raw logits (first maximum), also the bonus-position "resample"
            bv, bi = tl.max(lg, axis=0, return_indices=True)
            if bv > gv:
                gv = bv
                gi = v0 + start + bi
        else:
            if temp == 1.0:
                x = lg
            else:
                x = lg / temp                    # same fp32 division as vLLM's _temperature_kernel
            bm = tl.max(x, axis=0)
            m_new = tl.maximum(m, bm)
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m = m_new
            dl = tl.maximum(dl, tl.max(tl.where(gid == excl, x, -float("inf")), axis=0))
            resid = tl.where(gid != excl, x, -float("inf"))
            bv, bi = gumbel_noised_argmax(resid, gid, msk, seed, pos, temp, IS_DRAFTING=False, USE_FP64=False,
                                          APPLY_TEMPERATURE=False)
            if bv > gv:
                gv = bv
                gi = v0 + start + bi
    o = out_ptr + row * 8
    tl.store(o + 0, m)
    tl.store(o + 1, s)
    tl.store(o + 2, gv)
    tl.store(o + 3, gi.to(tl.int32).to(tl.float32, bitcast=True))
    tl.store(o + 6, dl)


@triton.jit
def _finalize_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, st_ptr, n_logits, draft_ptr, cu_ptr, idx_ptr,
                     eidx_ptr, temp_ptr, seed_ptr, pos_ptr, W: tl.constexpr):
    """stock _rejection_kernel (one-hot draft, no block verification) + _insert_resampled_kernel on gathered stats"""
    req = tl.program_id(0)
    rs = tl.load(idx_ptr + req).to(tl.int64)
    start = tl.load(cu_ptr + req).to(tl.int64)
    end = tl.load(cu_ptr + req + 1)
    nd = end - start - 1
    seed = tl.load(seed_ptr + rs)
    temp = tl.load(temp_ptr + rs).to(tl.float32)
    greedy = temp == 0.0
    ranks = tl.arange(0, W)
    acc = tl.zeros((), tl.int64)
    verifying = True
    for i in range(nd):
        li = start + i
        d = tl.load(draft_ptr + li + 1).to(tl.int64)
        valid = d >= 0
        d = tl.maximum(0, d)
        if not greedy:
            verifying &= valid
        if verifying:
            pos = tl.load(pos_ptr + li)
            u = tl_rand32(seed, pos, includes_zero=False)
            base = st_ptr + (ranks * n_logits + li) * 8
            if greedy:
                gv = tl.load(base + 2)
                r = tl.argmax(gv, axis=0)
                ta = tl.sum(tl.where(ranks == r, tl.load(base + 3).to(tl.int32, bitcast=True), 0)).to(tl.int64)
                accepted = (ta == d) & valid
                verifying = accepted
                acc += accepted
                tl.store(sampled_ptr + req * sampled_stride + i, d if accepted else ta)
            else:
                ms = tl.load(base + 0)
                ss = tl.load(base + 1)
                gmax = tl.max(ms, axis=0)
                lse = gmax + tl.log(tl.sum(ss * tl.exp(ms - gmax)))
                x_d = tl.max(tl.load(base + 6), axis=0)
                accepted = (x_d - lse) > tl.log(u) + 0.0
                verifying = accepted
                acc += accepted
                tl.store(sampled_ptr + req * sampled_stride + i, d)
    tl.store(num_sampled_ptr + req, (acc + 1).to(tl.int32))
    j = start + acc
    rs2 = tl.load(eidx_ptr + j).to(tl.int64)
    temp2 = tl.load(temp_ptr + rs2).to(tl.float32)
    is_bonus = j == end - 1
    if temp2 == 0.0 and not is_bonus:
        return
    base = st_ptr + (ranks * n_logits + j) * 8
    gv = tl.load(base + 2)
    gv = tl.where(gv != gv, -float("inf"), gv)
    r = tl.argmax(gv, axis=0)
    tok = tl.sum(tl.where(ranks == r, tl.load(base + 3).to(tl.int32, bitcast=True), 0)).to(tl.int64)
    tl.store(sampled_ptr + req * sampled_stride + acc, tok)


def row_stats(local_logits, excl, eidx, temp, seed, pos, v0, n_local):
    n = local_logits.shape[0]
    out = torch.zeros((n, NST), dtype=torch.float32, device=local_logits.device)
    if n:
        _row_stats_kernel[(n,)](local_logits, local_logits.stride(0), out, excl, eidx, temp, seed, pos, n_local, v0,
                                BLOCK=BLOCK, num_warps=WARPS)
    return out


def excl_tokens(draft_sampled, cu_num_logits):
    """per logits row: the draft token tested there (draft_sampled[i + 1]) unless the row is its request's last
    (bonus) row; -1 = none.  Global vocab ids (int32)."""
    n = draft_sampled.shape[0]
    nxt = torch.empty(n, dtype=torch.int32, device=draft_sampled.device)
    nxt[:-1] = draft_sampled[1:]
    nxt[-1:] = -1
    last = cu_num_logits[1:].long() - 1
    if _nosync():
        nxt.index_fill_(0, last, -1)
    else:
        nxt[last] = -1
    return nxt


def finalize(gathered, world, n_logits, draft_sampled, cu_num_logits, idx_mapping, eidx, temp, seed, pos, nspec):
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(num_reqs, nspec + 1, dtype=torch.int64)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=draft_sampled.device)
    _finalize_kernel[(num_reqs,)](sampled, sampled.stride(0), num_sampled, gathered, n_logits, draft_sampled,
                                  cu_num_logits, idx_mapping, eidx, temp, seed, pos, W=world, num_warps=1)
    return sampled, num_sampled


def _lm(model):
    return getattr(model, "language_model", model)


def install():
    if os.environ.get("ARC_LOGIT_BIAS"):
        print("[shard_sample] not installed: ARC_LOGIT_BIAS needs the full logits", flush=True)
        return
    from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample import sampler as smp
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

    orig_add = smp.Sampler.add_request

    def add_request(self, req_idx, prompt_len, sampling_params):
        orig_add(self, req_idx, prompt_len, sampling_params)
        if not hasattr(self, "_arc_plain"):
            self._arc_plain = np.zeros(len(self.needs_logits_processing), dtype=bool)
        st = self.sampling_states
        self._arc_plain[req_idx] = not (
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (self.thinking_budget_state.enabled and self.thinking_budget_state.use_thinking_budget[req_idx])
            or st.min_p.np[req_idx] != 0.0
            or st.top_k.np[req_idx] != st.vocab_size
            or st.top_p.np[req_idx] != 1.0)
    smp.Sampler.add_request = add_request

    orig_sample = mr.GPUModelRunner.sample
    state = {"n": 0, "checked": 0, "bad": 0, "logged": False}

    def eligible(self, input_batch, grammar_output):
        rsamp = self.rejection_sampler
        smpl = self.sampler
        if (grammar_output is not None or self.batch_sharder is not None or rsamp is None or smpl is None
                or input_batch.num_draft_tokens == 0 or input_batch.num_reqs == 0
                or getattr(self.speculator, "draft_logits", None) is not None
                or rsamp.watermark_key is not None or rsamp.use_block_verification
                or rsamp.synthetic_conditional_rates is not None or rsamp.enable_adaptive_verification
                or smpl.compute_nans or smpl.use_fp64_gumbel or getattr(smpl, "trace_replay_state", None) is not None
                or not hasattr(smpl, "_arc_plain")):
            return False
        idx = input_batch.idx_mapping_np
        if not smpl._arc_plain[idx].all():
            return False
        return smpl.sampling_states.max_num_logprobs(idx) == NO_LOGPROBS

    def gather(self, st):
        mode = AG
        if mode in ("auto", "host"):
            from . import host_ar
            h = host_ar._INSTANCES.get("tp")
            if h is not None:
                out = h.all_gather0(st)
                if out is not None:
                    return out
            if mode == "host" and not state.get("warned"):
                state["warned"] = True
                print("[shard_sample] host all-gather unavailable, using NCCL", flush=True)
        return get_tp_group().all_gather(st, dim=0)

    def sample(self, hidden_states, input_batch, grammar_output):
        world = get_tensor_model_parallel_world_size()
        if world == 1 or not eligible(self, input_batch, grammar_output):
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        lm = _lm(self.model)
        head = lm.lm_head
        si = head.shard_indices
        v0, v1 = si.org_vocab_start_index, si.org_vocab_end_index
        h = hidden_states[input_batch.logits_indices]
        local = lm.logits_processor(head, h, skip_gather=True)          # [n, vocab_per_rank (+ padding)]
        n = local.shape[0]
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        ss = self.sampler.sampling_states
        temp, seed = ss.temperature.gpu, ss.seeds.gpu
        eidx = input_batch.expanded_idx_mapping
        excl = excl_tokens(draft_sampled, input_batch.cu_num_logits)
        st = row_stats(local, excl, eidx, temp, seed, pos, v0, v1 - v0)
        gathered = gather(self, st)                                   # [world * n, 8], rank-major
        sampled, num_sampled = finalize(gathered, world, n, draft_sampled, input_batch.cu_num_logits,
                                        input_batch.idx_mapping, eidx, temp, seed, pos,
                                        self.rejection_sampler.num_speculative_steps)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled, input_batch.seq_lens, input_batch.cu_num_logits, input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu)
        out = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                            num_sampled=num_sampled, num_rejected=num_rejected)
        state["n"] += 1
        if not state["logged"]:
            state["logged"] = True
            print(f"[shard_sample] active: {n} rows, local vocab {v0}..{v1} of {local.shape[1]} cols, world {world}",
                  flush=True)
        if state["checked"] < CHECK:
            state["checked"] += 1
            ref, rn, rr = orig_sample(self, hidden_states, input_batch, grammar_output)
            k = num_sampled.shape[0]
            same_n = torch.equal(rn, num_sampled)
            mask = torch.arange(sampled.shape[1], device=sampled.device)[None, :] < num_sampled[:, None]
            same_t = same_n and torch.equal(torch.where(mask, sampled, 0), torch.where(mask, ref.sampled_token_ids, 0))
            if not (same_n and same_t):
                state["bad"] += 1
            if state["checked"] in (1, 10, 100, 1000) or state["checked"] == CHECK or not (same_n and same_t):
                print(f"[shard_sample] check {state['checked']}: reqs {k} counts {'same' if same_n else 'DIFF'} "
                      f"tokens {'same' if same_t else 'DIFF'} (bad steps {state['bad']})", flush=True)
            return ref, rn, rr
        return out, num_sampled, num_rejected

    mr.GPUModelRunner.sample = sample
    print(f"[shard_sample] installed (AG={AG}, check={CHECK}, nosync={int(NOSYNC)}, nosync_ab={_NOSYNC_AB:g})", flush=True)
