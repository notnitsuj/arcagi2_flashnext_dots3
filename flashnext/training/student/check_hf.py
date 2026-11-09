import argparse, json, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_fn as S
M = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--T', type=int, default=3000)
    ap.add_argument('--recs', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'student', 'recs.jsonl'))
    a = ap.parse_args()
    ids = None
    for l in open(a.recs):
        r = json.loads(l)
        if r['mode'] == 'repl' and len(r['ids']) >= a.T:
            ids = torch.tensor(r['ids'][:a.T])
            break
    t0 = time.time()
    st = S.Student(M, None, [0], trainable=(), layers=a.layers, src='fp8').eval()
    print(f'student built {time.time() - t0:.0f}s', flush=True)
    with torch.no_grad():
        S.PAD = 128
        hs = st.hidden(ids, ckpt=False)
        ls = (hs @ st.lm_head.t()).float()
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as Mq
    cfg = AutoConfig.from_pretrained(M).text_config
    cfg.num_hidden_layers = a.layers
    cfg.layer_types = cfg.layer_types[:a.layers]
    cfg._attn_implementation = 'eager'
    dev = torch.device('cuda:1')
    ck = st_ck = S.Ckpt(M)
    with torch.device('meta'):
        hf = Mq.Qwen4ExpTextModel(cfg)
    ple_s = [l.ple for l in st.layers if l.ple is not None][0]
    got = {}

    class Gather(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.register_buffer('weight', torch.zeros(1))

        def forward(self, rows):
            got['rows'] = rows.detach().cpu()
            r = rows.cpu().numpy()
            sh, rr = (r // ple_s.rows, r % ple_s.rows)
            import numpy as np
            out = np.empty(r.shape + (ple_s.dim,), np.uint8)
            for s in np.unique(sh):
                m = sh == s
                out[m] = ple_s.shards[int(s)][rr[m]]
            t = torch.from_numpy(out).view(torch.float8_e4m3fn).float() * ple_s.scale
            return t.to(torch.bfloat16).to(rows.device)
    sd = {}
    for k in ck.map:
        if not k.startswith(S.PFX) or 'visual' in k:
            continue
        n = k[len(S.PFX):]
        if n.startswith('layers.'):
            li = int(n.split('.')[1])
            if li >= a.layers:
                continue
        if 'ngram_embedding' in n or n.endswith('weight_scale_inv') or '.experts.' in n:
            continue
        sd[n] = ck.w(k, 'cpu') if k.endswith('weight') else ck.raw(k)
    for li in range(a.layers):
        gu, dn = ([], [])
        for e in range(cfg.num_experts):
            p = f'{S.PFX}layers.{li}.mlp.experts.{e}.'
            gu.append(torch.cat([ck.w(p + 'gate_proj.weight', 'cpu'), ck.w(p + 'up_proj.weight', 'cpu')]))
            dn.append(ck.w(p + 'down_proj.weight', 'cpu'))
        sd[f'layers.{li}.mlp.experts.gate_up_proj'] = torch.stack(gu)
        sd[f'layers.{li}.mlp.experts.down_proj'] = torch.stack(dn)
    for l in hf.layers:
        if l.ple is not None:
            l.ple.ple_embedding.ngram_embedding = Gather()
    hf.to_empty(device=dev)
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    print('missing', [m for m in missing if 'ngram_embedding' not in m and 'rotary' not in m][:20])
    print('unexpected', unexpected[:20])
    hf.rotary_emb.inv_freq = hf.rotary_emb.compute_default_rope_parameters(cfg)[0].to(dev)
    hf = hf.to(torch.bfloat16).eval()
    for l in hf.layers:
        if l.ple is not None:
            pe = l.ple.ple_embedding
            pe.layer_multipliers = ple_s.mult.to(dev)
            pe.ngram_heads_vocab_sizes = ple_s.sizes.to(dev)
            pe.ngram_heads_offsets = ple_s.offs.to(dev)
    with torch.no_grad():
        out = hf(input_ids=ids[None].to(dev), use_cache=False).last_hidden_state[0]
    rows_s = ple_s.ngram_ids(ids)
    print('ngram ids equal:', torch.equal(rows_s, got['rows'][0]))
    hs, out = (hs.float().cpu(), out.float().cpu())
    err = (hs - out).norm(dim=-1) / out.norm(dim=-1)
    print(f'mixer rel err: mean {err.mean():.4f} max {err.max():.4f}  first2k {err[:2000].mean():.4f} after2k {err[2100:].mean():.4f}')
    lh = (out.to(torch.bfloat16).to(st.last) @ st.lm_head.t()).float()
    kl = torch.nn.functional.kl_div(torch.log_softmax(ls, -1), torch.log_softmax(lh, -1), log_target=True, reduction='none').sum(-1).cpu()
    print(f'KL(hf||student): mean {kl.mean():.5f}  first2k {kl[:2000].mean():.5f} after2k {kl[2100:].mean():.5f}  top1 agree {(ls.argmax(-1) == lh.argmax(-1)).float().mean():.4f}')
if __name__ == '__main__':
    main()
