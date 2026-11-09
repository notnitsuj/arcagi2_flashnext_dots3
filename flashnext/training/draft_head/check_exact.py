import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept import BUILD, MODEL, load_rec, pad128
from mtp_torch import MTP
dev = torch.device('cuda')
m = MTP(MODEL, BUILD, dev)
args = sys.argv[1:]
if args and args[0] == '--delta':
    sd = torch.load(args[1], map_location='cpu')
    assert not m.load_trained(sd.get('params', sd))
    args = args[2:]
with torch.no_grad():
    for d in args:
        d = Path(d)
        ids, gen, hid, tid, tlp = load_rec(d, dev)
        T = ids.shape[0]
        ref = torch.from_numpy(np.load(d / 'dft.npy').view(np.int16).copy()).view(torch.bfloat16).to(dev)
        tok = pad128(torch.cat([ids[1:], ids.new_zeros(1)]).to(dev), T)
        sample, multi, st = m.step0(pad128(hid.to(dev), T), tok)
        a, b = (sample[:T - 1].float(), ref[:T - 1].float())
        rel = (a - b).norm(dim=-1) / b.norm(dim=-1)
        kl, agree = ([], [])
        for i in range(0, T - 1, 2048):
            la, lb = (m.logits(sample[i:min(i + 2048, T - 1)]), m.logits(ref[i:min(i + 2048, T - 1)]))
            pa, pb = (la.log_softmax(-1), lb.log_softmax(-1))
            kl.append((pb.exp() * (pb - pa)).sum(-1))
            agree.append((la.argmax(-1) == lb.argmax(-1)).float())
        kl, agree = (torch.cat(kl), torch.cat(agree))
        print(f"{d.name} T={T} hidden rel err mean {rel.mean():.4f} p99 {rel.quantile(0.99):.4f}; KL(vllm||torch) mean {kl.mean():.5f} p99 {kl.quantile(0.99):.4f}; top1 agree {agree.mean():.4f}; first2k KL {kl[:2048].mean():.5f} rest {(kl[2048:].mean() if T > 2050 else float('nan')):.5f}")
