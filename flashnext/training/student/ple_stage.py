import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_fn as S
d = Path(sys.argv[1] if len(sys.argv) > 1 else '/dev/shm/ple')
if (d / 'DONE').exists():
    sys.exit(0)
d.mkdir(parents=True, exist_ok=True)
ck = S.Ckpt(os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8'))
e = S.PFX + 'layers.1.ple.ple_embedding.ngram_embedding.'
i = 0
while ck.has(f'{e}shard_{i}.weight'):
    mm, shape, dt = ck.memmap(f'{e}shard_{i}.weight')
    f = d / f'shard_{i}.u8'
    if not f.exists() or f.stat().st_size != mm.shape[0]:
        np.asarray(mm).tofile(str(f) + '.tmp')
        os.replace(str(f) + '.tmp', f)
    i += 1
(d / 'DONE').write_text(str(i))
print('staged', i, 'shards')
