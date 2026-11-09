import argparse, os
import torch
ap = argparse.ArgumentParser()
ap.add_argument('--reap', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'reap_scores.pt'))
ap.add_argument('--out', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'stats_reapsum.pt'))
ap.add_argument('--template', default=None, help='optional routing-statistics file whose count/tokens fields are kept')
a = ap.parse_args()
R = torch.load(a.reap)
T = torch.load(a.template) if a.template else None
out = {}
for l in sorted(R):
    p = f'language_model.model.layers.{l}.mlp'
    d = dict(T[p]) if T is not None else {'count': R[l]['count'].double(), 'tokens': int(R[l]['tokens'])}
    d['wsum'] = (R[l]['reap'] * R[l]['count']).float()
    out[p] = d
torch.save(out, a.out)
print('wrote', a.out, len(out), 'layers; wsum = sum over calibration tokens of gate weight x expert output norm')
