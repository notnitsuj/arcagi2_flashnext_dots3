"""ARC_DELTA_FILE=<delta.safetensors>: load-time overlay of an in-format fine-tune (expert GPTQ scales, RMSNorms, router)
on top of an `arc_lowbit` checkpoint, so a student ships as a ~1 GB file next to the unchanged base weights instead of a
new 62 GB checkpoint. Default off (unset -> nothing is patched or read).

File (written by the delta exporter), all bit-exact to the full student checkpoint of the student checkpoint build:
  lowbit.L{ll}.experts  int64 [E]   expert ids, must equal the base lowbit-L{ll}.pt order
  lowbit.L{ll}.gu_s     int8/int16  student fp16 scale bits - base fp16 scale bits (ulp steps), [E, 2I, KG]
  lowbit.L{ll}.d_s      int8/int16  same for the down projection, [E, H, KGd]
  lowbit.L{ll}.base_ck  int64 [2]   checksum of the base gu_s / d_s bits (refuses a different base)
  dense.<ckpt tensor name>          replacement tensor in the checkpoint dtype (norms, router gates)
Hooks: ArcLowbitExperts.load (scales, before TP slicing / pair conversion) and DefaultModelLoader.get_all_weights
(dense tensors replace the base safetensors tensors in the weight stream, before vLLM's own loading/processing).
"""
import os

import torch

_cache = {}


def path():
    return os.environ.get("ARC_DELTA_FILE") or None


def _open():
    p = path()
    if p not in _cache:
        from safetensors import safe_open
        _cache[p] = safe_open(p, "pt")
    return _cache[p]


def _bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32)


def checksum(t):
    return int((_bits(t).to(torch.int64) * (torch.arange(t.numel(), dtype=torch.int64).view(t.shape) % 9973 + 1)).sum())


def apply_lowbit(layer, d):
    """Patch the scales of a loaded lowbit-L{layer}.pt dict in place (no-op if the delta has no entry for the layer)."""
    f = _open()
    k = f"lowbit.L{layer:02d}."
    if k + "experts" not in f.keys():
        return 0
    ids = f.get_tensor(k + "experts").tolist()
    assert ids == list(d["experts"]), f"ARC_DELTA_FILE: expert order mismatch in layer {layer}"
    ck = f.get_tensor(k + "base_ck").tolist()
    gu0 = torch.stack([d[e]["gu_s"] for e in d["experts"]])
    dn0 = torch.stack([d[e]["d_s"] for e in d["experts"]])
    assert [checksum(gu0), checksum(dn0)] == ck, f"ARC_DELTA_FILE: base checkpoint mismatch in layer {layer}"
    for nm, base in (("gu_s", gu0), ("d_s", dn0)):
        new = (_bits(base) + f.get_tensor(k + nm).to(torch.int32)).to(torch.int16).view(torch.float16)
        for i, e in enumerate(d["experts"]):
            d[e][nm] = new[i].clone()
    return 1


def wrap_weights(it):
    """Weight-stream filter: (name, tensor) pairs whose name is in the delta get the delta tensor."""
    f = _open()
    names = {k[6:] for k in f.keys() if k.startswith("dense.")}
    n = 0
    for name, t in it:
        if name in names:
            r = f.get_tensor("dense." + name)
            assert r.shape == t.shape and r.dtype == t.dtype, (name, r.shape, t.shape, r.dtype, t.dtype)
            t = r
            n += 1
        yield name, t
    print(f"[arc] ARC_DELTA_FILE: replaced {n}/{len(names)} dense tensors", flush=True)


def install():
    """Patch vLLM's default loader weight stream (called from arcstats.register when ARC_DELTA_FILE is set)."""
    from vllm.model_executor.model_loader import default_loader as dl
    orig = dl.DefaultModelLoader.get_all_weights

    def get_all_weights(self, *a, **kw):
        return wrap_weights(orig(self, *a, **kw))
    dl.DefaultModelLoader.get_all_weights = get_all_weights
