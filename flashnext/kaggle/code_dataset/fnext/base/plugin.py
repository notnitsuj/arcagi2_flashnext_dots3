import functools

def _draft_override(num_experts, hf_config):
    from vllm.config.speculative import SpeculativeConfig
    hf_config = SpeculativeConfig.hf_config_override(hf_config)
    for c in (hf_config, getattr(hf_config, 'text_config', None)):
        if c is not None and hasattr(c, 'num_experts'):
            c.num_experts = num_experts
    return hf_config

def _host_ar_patch():
    import os
    from vllm.distributed.device_communicators import cuda_communicator as cc
    from vllm.logger import init_logger
    log = init_logger('vllm.fnext.host_ar')
    orig_init, orig_ar = (cc.CudaCommunicator.__init__, cc.CudaCommunicator.all_reduce)

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._fn_har = None
        if self.world_size in (2, 4, 8) and self.unique_name.split(':')[0] == 'tp':
            from .host_ar import HostAllReduce
            from . import host_ar
            name = f'{os.getppid()}_{self.unique_name.replace(':', '_')}'
            self._fn_har = HostAllReduce(self.rank_in_group, self.world_size, name, self.cpu_group)
            host_ar._INSTANCES.setdefault('tp', self._fn_har)
            log.info('FN_HOST_AR: host all-reduce for %s (world %d, %.0f MB max)', self.unique_name, self.world_size, self._fn_har.maxn * 2 / 2 ** 20)

    def all_reduce(self, input_):
        h = getattr(self, '_fn_har', None)
        if h is not None and input_.is_cuda and h.supports(input_):
            return h(input_)
        return orig_ar(self, input_)
    cc.CudaCommunicator.__init__ = init
    cc.CudaCommunicator.all_reduce = all_reduce

def _logits_local_patch():
    from vllm.models.qwen4_exp.nvidia import model as qm

    def local(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def wrap_local(self, hidden_states):
        return self.language_model.compute_logits_local(hidden_states)
    qm.Qwen4ExpForCausalLM.compute_logits_local = local
    qm.Qwen4ExpForConditionalGeneration.compute_logits_local = wrap_local

def register():
    import os
    from . import fnplug
    fnplug.register()
    if os.environ.get('FN_LOAD_EC') == '1':
        from . import load_ec
        load_ec.install()
    if os.environ.get('FN_QSA_SHARE_TOPK') == '1':
        from . import qsa_topk
        qsa_topk.install()
    if os.environ.get('FN_HOST_AR') == '1':
        _host_ar_patch()
    if os.environ.get('FN_HOST_DP') == '1':
        from . import host_dp
        host_dp.patch()
        if os.environ.get('FN_EP_Q8_RS') == '1' or os.environ.get('FN_TP_Q8_RS') == '1':
            from . import q8_rs
            q8_rs.install()
        if os.environ.get('FN_HOST_DP_FUSE') == '1':
            from . import host_dp_fused
            host_dp_fused.install()
    if os.environ.get('FN_GDN_LAZY') == '1':
        from . import gdn_lazy
        gdn_lazy.patch()
    if os.environ.get('FN_GDN_BA_OVERLAP') == '1':
        from . import gdn_ba_overlap
        gdn_ba_overlap.install()
    if os.environ.get('FN_SAMPLE_SHARD') == '1':
        _logits_local_patch()
    if os.environ.get('FN_EP_Q8') == '1':
        from . import q8_dispatch
        q8_dispatch.install()
    if not (os.environ.get('FN_KEEP') or os.environ.get('FN_MOE_LOWBIT') or os.environ.get('FN_DRAFT_NEXP')):
        return
    from vllm.config.speculative import SpeculativeConfig
    orig = SpeculativeConfig.compose_draft_hf_overrides

    def compose(target_hf_overrides):
        if isinstance(target_hf_overrides, dict):
            ne = (target_hf_overrides.get('text_config') or {}).get('num_experts', target_hf_overrides.get('num_experts'))
            if ne is not None:
                return functools.partial(_draft_override, int(ne))
        return orig(target_hf_overrides)
    SpeculativeConfig.compose_draft_hf_overrides = staticmethod(compose)
