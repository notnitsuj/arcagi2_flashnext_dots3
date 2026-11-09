import argparse, json, os
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', required=True)
    ap.add_argument('--split', default='ho')
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--gen', type=int, default=768)
    ap.add_argument('--max-ctx', type=int, default=24576)
    ap.add_argument('--model', default=os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8'))
    ap.add_argument('--kv-skip', default='')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seqs', type=int, default=24)
    ap.add_argument('--util', type=float, default=0.88)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    L = [json.loads(l) for l in open(Path(a.recs) / 'index.jsonl')]
    L = sorted((x for x in L if x['split'] == a.split), key=lambda x: x['key'])
    rng = np.random.default_rng(a.seed)
    L = [L[i] for i in rng.permutation(len(L))[:a.n]]
    prompts = []
    for x in L:
        z = np.load(Path(a.recs) / f"{x['key']}.npz")
        ids, gen = (z['ids'], z['gen'])
        cand = np.nonzero(gen[:a.max_ctx])[0]
        cand = cand[cand > 512]
        if len(cand) == 0:
            continue
        cut = int(cand[rng.integers(len(cand))])
        prompts.append({'prompt_token_ids': ids[:cut].astype(int).tolist()})
    from vllm import LLM, SamplingParams
    spec = {'method': 'mtp', 'num_speculative_tokens': a.k, 'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'block'}
    llm = LLM(model=a.model, tensor_parallel_size=1, language_model_only=True, hf_overrides={'text_config': {'num_experts': 320}}, kv_cache_dtype='fp8_e4m3', kv_cache_dtype_skip_layers=[x for x in a.kv_skip.split(',') if x], attention_config={'indexer_kv_dtype': 'fp8'}, mamba_ssm_cache_dtype='bfloat16', max_model_len=a.max_ctx + a.gen + 64, max_num_seqs=a.seqs, max_num_batched_tokens=4096, enable_prefix_caching=False, gpu_memory_utilization=a.util, speculative_config=spec, use_replayssm=True, mamba_backend='triton', seed=4000, disable_log_stats=False)
    sps = [SamplingParams(max_tokens=a.gen, min_tokens=a.gen, temperature=1.0, top_p=0.95, top_k=20, seed=1000 + i, ignore_eos=True) for i in range(len(prompts))]
    outs = llm.generate(prompts, sps)
    met = {}
    for m in llm.get_metrics():
        if 'spec_decode' in m.name:
            if hasattr(m, 'values'):
                met[m.name] = list(m.values)
            else:
                met[m.name] = getattr(m, 'value', None)
    nd = met.get('vllm:spec_decode_num_drafts')
    na = met.get('vllm:spec_decode_num_accepted_tokens')
    per = met.get('vllm:spec_decode_num_accepted_tokens_per_pos')
    res = {'n_prompts': len(prompts), 'gen': a.gen, 'kv_skip': a.kv_skip, 'num_drafts': nd, 'accepted': na, 'tok_step': round(1 + na / nd, 4) if nd else None, 'uncond_per_pos': [round(v / nd, 4) for v in per] if per and nd else None, 'out_tokens': sum((len(o.outputs[0].token_ids) for o in outs))}
    print('VACC', json.dumps(res), flush=True)
    json.dump({'res': res, 'metrics': met}, open(a.out, 'w'), indent=1)
if __name__ == '__main__':
    main()
