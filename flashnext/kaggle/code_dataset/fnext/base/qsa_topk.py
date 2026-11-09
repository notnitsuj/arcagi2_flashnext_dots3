import os
_SCOPE = []
_DONE = []

def _patch_classes(M=None, Q=None):
    if M is None:
        from vllm.models.qwen4_exp.nvidia import model as M
    if Q is None:
        from vllm.models.qwen4_exp.nvidia import qsa as Q
    QM, QA = (M.Qwen4ExpModel, Q.Qwen4ExpQSAAttention)
    if getattr(QM.__init__, '_fn_qsa_share', False):
        return
    f_m, f_a = (QM.__init__, QA.__init__)

    def init_model(self, *a, **k):
        _SCOPE.append([None, 0, 0])
        try:
            f_m(self, *a, **k)
        finally:
            _DONE.append(_SCOPE.pop())

    def init_attn(self, *a, **k):
        f_a(self, *a, **k)
        if not _SCOPE:
            return
        st = _SCOPE[-1]
        bufs = self._buffers
        buf = bufs.get('topk_indices_buffer')
        if buf is None:
            return
        sh = st[0]
        if sh is None:
            st[0], st[1] = (buf, 1)
            return
        if buf.shape != sh.shape or buf.dtype != sh.dtype or buf.device != sh.device:
            return
        bufs['topk_indices_buffer'] = sh
        st[1] += 1
        st[2] += buf.numel() * buf.element_size()
    init_model._fn_qsa_share = init_attn._fn_qsa_share = True
    QM.__init__, QA.__init__ = (init_model, init_attn)

def install(GW=None):
    if os.environ.get('FN_QSA_SHARE_TOPK', '0') != '1':
        return False
    if GW is None:
        import vllm.v1.worker.gpu_worker as GW
    W = GW.Worker
    f_lm = W.load_model
    if getattr(f_lm, '_fn_qsa_share', False):
        return True

    def load_model(self, *a, **k):
        try:
            _patch_classes()
        except Exception as e:
            print(f'[qsa_topk] not applied: {e!r}'[:200], flush=True)
        r = f_lm(self, *a, **k)
        for _, n, rel in _DONE:
            print(f'[qsa_topk] {n} QSA layers share one top-k buffer: {rel / 2 ** 20:.0f} MiB of buffers not kept', flush=True)
        _DONE.clear()
        return r
    load_model._fn_qsa_share = True
    W.load_model = load_model
    return True
