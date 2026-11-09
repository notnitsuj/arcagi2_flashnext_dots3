import json, os, re, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
M = Path(os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')).resolve()
out = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('FN_WORK', 'work'), 'dense'))
out.mkdir(parents=True, exist_ok=True)
idx = json.load(open(M / 'model.safetensors.index.json'))['weight_map']
shutil.copy(M / 'config.json', out / 'config.json')
wm, buf, part, size = ({}, {}, 0, 0)
hs = {}
for k in sorted(idx):
    if not (k.startswith('model.language_model.') or k == 'lm_head.weight'):
        continue
    if '.experts.' in k:
        continue
    if 'ngram_embedding.shard_' in k:
        wm[k] = str(M / idx[k])
        continue
    f = idx[k]
    if f not in hs:
        hs[f] = safe_open(str(M / f), 'pt')
    t = hs[f].get_tensor(k)
    buf[k] = t
    size += t.numel() * t.element_size()
    wm[k] = f'dense-{part:03d}.safetensors'
    if size > 4000000000.0:
        save_file(buf, str(out / f'dense-{part:03d}.safetensors'))
        buf, size = ({}, 0)
        part += 1
if buf:
    save_file(buf, str(out / f'dense-{part:03d}.safetensors'))
json.dump({'weight_map': wm}, open(out / 'model.safetensors.index.json', 'w'))
print('done', part + 1, 'files', len(wm), 'keys')
