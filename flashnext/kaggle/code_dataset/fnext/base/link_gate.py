import os
import torch
ON = os.environ.get('ARC_OFFLOAD_AR_GATE', '0') == '1'
_W = {}

def word(dev):
    if not ON:
        return None
    dev = torch.device(dev)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    w = _W.get(key)
    if w is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('ARC_OFFLOAD_AR_GATE: gate word first requested during CUDA-graph capture')
        w = _W[key] = torch.zeros(1, dtype=torch.int32, device=torch.device('cuda', key))
    return w
