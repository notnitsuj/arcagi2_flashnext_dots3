import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ST = {'on': False, 'draft': False, 'chunks': [], 'dchunks': [], 'pos': None, 'outer': None}

def _first_rows(pos):
    return pos[0] if pos.dim() == 2 else pos

def install(worker):
    import torch.nn as nn
    runner = worker.model_runner
    tgt = runner.model if hasattr(runner, 'model') else runner.get_model()
    mixers = [(n, m) for n, m in tgt.named_modules() if n.endswith('hyper_connection_mixer')]
    assert len(mixers) == 1, [n for n, _ in mixers]
    mname, mixer = mixers[0]
    inner = [m for n, m in tgt.named_modules() if type(m).__name__ == 'Qwen4ExpModel']
    assert len(inner) == 1, [type(m).__name__ for m in tgt.modules()][:20]
    inner = inner[0]
    ST['outer'] = tgt
    of = inner.forward

    def fwd(input_ids, positions, *a, **k):
        ST['pos'] = _first_rows(positions)
        return of(input_ids, positions, *a, **k)
    inner.forward = fwd
    oc = mixer.combine_and_mix

    def cam(*a, **k):
        multi, sample, inj = oc(*a, **k)
        if ST['on']:
            pos = ST['pos']
            n = pos.shape[0]
            tid, tlp = ([], [])
            for i in range(0, n, 2048):
                lg = ST['outer'].compute_logits(sample[i:i + 2048]).float()
                lse = torch.logsumexp(lg, -1, keepdim=True)
                v, ix = lg.topk(20, -1)
                tid.append(ix.int())
                tlp.append(v - lse)
            ST['chunks'].append((pos.cpu().numpy().copy(), multi[:n].contiguous().view(torch.int16).cpu().numpy(), torch.cat(tid).cpu().numpy(), torch.cat(tlp).cpu().numpy()))
        return (multi, sample, inj)
    mixer.combine_and_mix = cam
    drafter = None
    for nm in ('drafter', 'speculator', 'proposer'):
        d = getattr(runner, nm, None)
        if d is not None and isinstance(getattr(d, 'model', None), nn.Module):
            drafter = d.model
    msg = f'target mixer {mname}; inner {type(inner).__name__}'
    if drafter is not None:
        mtp = [m for m in drafter.modules() if type(m).__name__ == 'Qwen4ExpMTP']
        if mtp:
            mtp = mtp[0]
            odf = mtp.forward

            def dfwd(input_ids, positions, hidden_states=None, intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0):
                out = odf(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx=spec_step_idx)
                if ST['on'] and ST['draft'] and (spec_step_idx == 0):
                    p = _first_rows(positions)
                    n = p.shape[0]
                    ST['dchunks'].append((p.cpu().numpy().copy(), input_ids[:n].cpu().numpy().copy(), out[0][:n].contiguous().view(torch.int16).cpu().numpy()))
                return out
            mtp.forward = dfwd
            msg += '; drafter Qwen4ExpMTP hooked'
    return msg

def assemble(chunks, T):
    seen = np.zeros(T, np.int32)
    outs = None
    for c in chunks:
        pos = c[0]
        n = min(len(pos), *[len(x) for x in c[1:]])
        pos = pos[:n]
        ok = (pos >= 0) & (pos < T)
        if outs is None:
            outs = [np.zeros((T,) + x.shape[1:], x.dtype) for x in c[1:]]
        for o, x in zip(outs, c[1:]):
            o[pos[ok]] = x[:n][ok]
        np.add.at(seen, pos[ok], 1)
    return (outs, seen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='tr')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--skip', type=int, default=0, help='skip the first N chosen records (resume / shard)')
    ap.add_argument('--max-len', type=int, default=65536)
    ap.add_argument('--min-len', type=int, default=0)
    ap.add_argument('--draft-check', type=int, default=0, help="also dump vLLM's MTP step-0 output for the first N")
    ap.add_argument('--model', default=os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8'))
    ap.add_argument('--k', type=int, default=320)
    ap.add_argument('--bt', type=int, default=8192)
    ap.add_argument('--util', type=float, default=0.85)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kv', default='fp8_e4m3')
    ap.add_argument('--kv-skip', default='')
    a = ap.parse_args()
    recs = Path(a.recs)
    L = [json.loads(l) for l in open(recs / 'index.jsonl')]
    L = [x for x in L if x['split'] == a.split and a.min_len <= x['T'] <= a.max_len]
    rng = np.random.default_rng(a.seed)
    L = [L[i] for i in rng.permutation(len(L))][a.skip:a.skip + a.n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    spec = {'method': 'mtp', 'num_speculative_tokens': 3, 'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'block'}
    llm = LLM(model=a.model, tensor_parallel_size=1, language_model_only=True, enforce_eager=True, hf_overrides={'text_config': {'num_experts': a.k}}, kv_cache_dtype=a.kv, kv_cache_dtype_skip_layers=[x for x in a.kv_skip.split(',') if x], attention_config={'indexer_kv_dtype': 'fp8'}, mamba_ssm_cache_dtype='bfloat16', max_model_len=a.max_len + 64, max_num_seqs=1, max_num_batched_tokens=a.bt, enable_prefix_caching=False, gpu_memory_utilization=a.util, speculative_config=spec, use_replayssm=True, mamba_backend='triton', seed=4000)
    print(llm.collective_rpc(install), flush=True)
    sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.95, top_k=20)
    t0 = time.time()
    ntok = 0
    for i, x in enumerate(L):
        d = out / x['key']
        if (d / 'DONE').exists():
            continue
        z = np.load(recs / f"{x['key']}.npz")
        ids = z['ids'].astype(np.int64)
        T = len(ids)
        ST['chunks'], ST['dchunks'] = ([], [])
        ST['draft'] = i < a.draft_check
        ST['on'] = True
        llm.generate([{'prompt_token_ids': ids.tolist()}], sp, use_tqdm=False)
        ST['on'] = False
        (hid, tid, tlp), seen = assemble(ST['chunks'], T)
        assert (seen == 1).all(), f"{x['key']}: positions seen {np.bincount(seen)}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / 'hid.npy', hid.view(np.uint16))
        np.save(d / 'tid.npy', tid)
        np.save(d / 'tlp.npy', tlp)
        if ST['dchunks']:
            (dids, dft), dseen = assemble(ST['dchunks'], T)
            assert (dids[:T - 1] == ids[1:]).all(), 'drafter ids != shifted record ids'
            np.save(d / 'dft.npy', dft.view(np.uint16))
            np.save(d / 'dseen.npy', dseen.astype(np.int8))
        np.save(d / 'ids.npy', ids.astype(np.int32))
        np.save(d / 'gen.npy', z['gen'])
        (d / 'DONE').write_text('ok')
        ntok += T
        el = time.time() - t0
        print(f"[{i + 1}/{len(L)}] {x['key']} T={T} {ntok / el:.0f} tok/s {el:.0f}s", flush=True)
if __name__ == '__main__':
    main()
