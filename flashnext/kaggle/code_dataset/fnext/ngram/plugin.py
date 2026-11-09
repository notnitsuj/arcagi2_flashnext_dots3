import importlib
import importlib.abc
import os
import sys
_TARGETS = {}

def _on(name):
    return os.environ.get(name, '0') == '1'

class _PostImport(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = f.find_spec(fullname, path, target) if hasattr(f, 'find_spec') else None
            if spec is not None and spec.loader is not None:
                orig = spec.loader.exec_module

                def exec_module(module, _orig=orig, _name=fullname):
                    _orig(module)
                    for cb in _TARGETS.get(_name, []):
                        cb(module)
                spec.loader.exec_module = exec_module
                return spec
        return None

def _when_imported(name, cb):
    if name in sys.modules:
        cb(sys.modules[name])
        return
    _TARGETS.setdefault(name, []).append(cb)
    if not any((isinstance(f, _PostImport) for f in sys.meta_path)):
        sys.meta_path.insert(0, _PostImport())

def _patch_ngram(mod):
    from fnext_plugins.ngram import host_table
    if getattr(mod, '_fn_ngram_patched', False):
        return
    mod.Qwen4ExpPLEPinnedHostEmbedding = host_table.SharedHostPLEEmbedding
    cls = mod.Qwen4ExpNGramEmbedding
    orig_init = cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        if getattr(self.ngram_embedding, 'shared_full_table', False):
            self.use_reduce_scatter = False
    cls.__init__ = __init__
    mod._fn_ngram_patched = True

def _patch_embed(mod):
    from fnext_plugins.ngram import host_table
    if getattr(mod, '_fn_embed_patched', False):
        return
    mod.VocabParallelEmbedding = host_table.SharedHostTokenEmbedding
    mod._fn_embed_patched = True

def register():
    if _on('FN_NGRAM_SHARED'):
        _when_imported('vllm.models.qwen4_exp.nvidia.ngram_embedding', _patch_ngram)
    if _on('FN_EMBED_HOST'):
        _when_imported('vllm.models.qwen4_exp.nvidia.model', _patch_embed)
        _when_imported('vllm.models.qwen4_exp.nvidia.mtp', _patch_embed)
