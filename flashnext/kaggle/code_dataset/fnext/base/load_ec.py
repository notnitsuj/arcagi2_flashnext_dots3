import os

def _pre_release(mod, name):
    import torch
    f = getattr(mod, name, None)
    if f is None or getattr(f, '_fn_load_ec', False):
        return False

    def w(*a, **k):
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return f(*a, **k)
    w._fn_load_ec = True
    setattr(mod, name, w)
    return True

def install():
    if os.environ.get('FN_LOAD_EC', '0') != '1':
        return False
    import inspect
    import torch
    import vllm.model_executor.model_loader.utils as U
    f0 = U.release_device_memory_under_pressure
    if getattr(f0, '_fn_load_ec', False):
        return True
    if 'release_device_memory_under_pressure(target_device)' not in inspect.getsource(U.process_weights_after_loading):
        print("[load_ec] vLLM's per-module release hook not found: NOT installed", flush=True)
        return False

    def release(device):
        r = f0(device)
        if getattr(device, 'type', None) == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return r
    release._fn_load_ec = True
    U.release_device_memory_under_pressure = release
    mb = os.environ.get('FN_LOAD_EC_SPLIT_MB', '')
    if mb.isdigit() and int(mb) > 20:
        try:
            import vllm.v1.worker.gpu_worker as GW
            W = GW.Worker
            f_sc, f_lm = (W._scoped_allocator_max_split, W.load_model)
            st = {'load': False}

            def scoped(self, max_split_size_mb):
                return f_sc(self, int(mb) if st['load'] else max_split_size_mb)

            def load_model(self, *a, **k):
                st['load'] = True
                try:
                    return f_lm(self, *a, **k)
                finally:
                    st['load'] = False
            W._scoped_allocator_max_split, W.load_model = (scoped, load_model)
        except Exception as e:
            print(f'[load_ec] FN_LOAD_EC_SPLIT_MB not applied: {e!r}'[:200], flush=True)
    if os.environ.get('FN_LOAD_EC_INNER', '1') == '1':
        try:
            from ..moe import lowbit_kernels as LK
            _pre_release(LK, '_to_pairs')
        except Exception:
            pass
        try:
            from vllm.model_executor.layers.quantization.online import fp8 as OF
            _pre_release(OF, '_fp8_quant_per_channel')
        except Exception:
            pass
    return True
