"""ARC_OFFLOAD_AR_GATE=1 (default off): keep the background (MIRROR=2) row copies off the
PCIe link while a host all-reduce runs. host_ar.HostAllReduce sets a per-device int32 word to 1 right before its
kernel and back to 0 right after (two tiny stream-ordered fills, captured into the CUDA graphs with the all-reduce);
the mirror side copy (csrc fill_copy_kernel, offload_fill_copy_launch_gated) checks the word between row batches and
naps while it is set (each wait bounded at 4 ms). Same rows, same destinations: bit-exact; only timing changes.
On Kaggle an all-reduce overlapping a fill copy took 1064 vs 498 us (cfgprobe nsys, ~6.8 ms/step)."""
import os

import torch

ON = os.environ.get("ARC_OFFLOAD_AR_GATE", "0") == "1"
_W = {}


def word(dev):
    """the gate word of device dev (int32 [1], zero-initialized), or None when the gate is off"""
    if not ON:
        return None
    dev = torch.device(dev)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    w = _W.get(key)
    if w is None:
        if torch.cuda.is_current_stream_capturing():   # (allocated by LayerOffload.__init__ before any capture)
            raise RuntimeError("ARC_OFFLOAD_AR_GATE: gate word first requested during CUDA-graph capture")
        w = _W[key] = torch.zeros(1, dtype=torch.int32, device=torch.device("cuda", key))
    return w
