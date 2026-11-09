import ctypes
import mmap
import os
import socket
import tempfile
import threading
import time
import weakref
import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.triton_utils import tl, triton
logger = init_logger('vllm.fnext.ngram')
QWEN_PLE_FP8_SCALE = 0.00019931793212890625
SHM_DIR = os.environ.get('FN_NGRAM_SHM_DIR', '/dev/shm')

def _env(name, default='0'):
    return os.environ.get(name, default)

class HostStore:
    _bufs: dict = {}
    _reg: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key, nbytes, async_register=False):
        with cls._lock:
            if key in cls._bufs:
                t = cls._bufs[key]
                if t.numel() != nbytes:
                    raise RuntimeError(f'HostStore {key}: size {t.numel()} != {nbytes}')
                return (t, False)
            t, reg = _alloc_shared(key, nbytes, async_register)
            cls._bufs[key] = t
            cls._reg[key] = reg
            return (t, True)

    @classmethod
    def ready(cls, key):
        reg = cls._reg.get(key)
        if reg is None or reg.get('done'):
            return
        reg['thread'].join()
        import torch.distributed as dist
        g = _world()
        errs = [None] * g.world_size
        dist.all_gather_object(errs, reg['error'], group=g.cpu_group)
        bad = [f'rank {i}: {e}' for i, e in enumerate(errs) if e]
        if bad:
            raise RuntimeError(f'HostStore {key} registration failed: ' + '; '.join(bad))
        reg['done'] = True
        logger.info('HostStore %s registered (%.1f s, overlapped with loading)', key, reg['secs'])

def _world():
    from vllm.distributed.parallel_state import get_world_group
    return get_world_group()

def _register(owner, mm, nbytes, reg):
    t0 = time.time()
    try:
        ptr = owner.ctypes.data
        res = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
        if int(getattr(res, 'value', res)) != 0:
            raise RuntimeError(f'cudaHostRegister({nbytes}) -> {res}')
        fin = weakref.finalize(owner, _unregister, mm, ptr)
        fin.atexit = False
        if not torch.from_numpy(owner[:1]).is_pinned():
            raise RuntimeError('registered mapping is not seen as pinned')
    except Exception as e:
        reg['error'] = f'{type(e).__name__}: {e}'
    reg['secs'] = time.time() - t0

def _alloc_shared(key, nbytes, async_register=False):
    import torch.distributed as dist
    g = _world()
    W, r = (g.world_size, g.rank_in_group)
    hosts = [None] * W
    dist.all_gather_object(hosts, socket.gethostname(), group=g.cpu_group)
    if len(set(hosts)) != 1:
        raise RuntimeError(f'HostStore needs all ranks on one node, got {sorted(set(hosts))}')
    use_memfd = _env('FN_NGRAM_MEMFD') == '1'
    handle = err = None
    keep = None
    t0 = time.time()
    if r == 0:
        try:
            if use_memfd:
                fd = os.memfd_create(f'fn_{key}', 0)
                os.ftruncate(fd, nbytes)
                keep = fd
                handle = ('memfd', os.getpid(), fd)
            else:
                st = os.statvfs(SHM_DIR)
                free = st.f_bavail * st.f_frsize
                if free < nbytes:
                    raise RuntimeError(f'{SHM_DIR} has {free / 2 ** 30:.1f} GiB free, need {nbytes / 2 ** 30:.1f}')
                f = tempfile.NamedTemporaryFile(prefix=f'fn_{key}_', dir=SHM_DIR, delete=False)
                f.truncate(nbytes)
                fd = os.dup(f.fileno())
                f.close()
                handle = ('file', f.name, 0)
            if _env('FN_NGRAM_PREFAULT', '1') == '1':
                os.posix_fallocate(fd, 0, nbytes)
            if not use_memfd:
                os.close(fd)
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
    handle, err = g.broadcast_object((handle, err), src=0)
    if err:
        raise RuntimeError(f'HostStore {key}: rank 0 failed: {err}')
    kind, a, b = handle
    tensor = error = None
    reg = {'error': None, 'secs': 0.0, 'done': False}
    try:
        path = f'/proc/{a}/fd/{b}' if kind == 'memfd' else a
        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED)
        finally:
            os.close(fd)
        owner = np.frombuffer(mm, dtype=np.uint8)
        tensor = torch.from_numpy(owner)
    except Exception as e:
        error = f'{type(e).__name__}: {e}'
    errs = [None] * W
    dist.all_gather_object(errs, error, group=g.cpu_group)
    if r == 0 and kind == 'file':
        try:
            os.unlink(a)
        except OSError:
            pass
    bad = [f'rank {i}: {e}' for i, e in enumerate(errs) if e]
    if bad:
        raise RuntimeError(f'HostStore {key} failed: ' + '; '.join(bad))
    if keep is not None:
        _MEMFD_KEEP.append(keep)
    if async_register:
        dev = torch.cuda.current_device()

        def run():
            torch.cuda.set_device(dev)
            _register(owner, mm, nbytes, reg)
        reg['thread'] = threading.Thread(target=run, name=f'fn_reg_{key}', daemon=True)
        reg['thread'].start()
    else:
        _register(owner, mm, nbytes, reg)
        errs = [None] * W
        dist.all_gather_object(errs, reg['error'], group=g.cpu_group)
        bad = [f'rank {i}: {e}' for i, e in enumerate(errs) if e]
        if bad:
            raise RuntimeError(f'HostStore {key} registration failed: ' + '; '.join(bad))
        reg['done'] = True
    logger.info('HostStore %s: %.2f GiB shared by %d ranks (%s), alloc %.1f s, register %s', key, nbytes / 2 ** 30, W, kind, time.time() - t0, 'async' if async_register else f'{reg['secs']:.1f} s')
    return (tensor, reg)
_MEMFD_KEEP = []

def _unregister(mm, ptr):
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:
        pass

def _uva(t):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(t)

@triton.jit
def _gather_rows_kernel(table_ptr, ids_ptr, out_ptr, n_ids, n_rows, row_words, ROWS: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    wblk = tl.program_id(1)
    i = pid * ROWS + tl.arange(0, ROWS)
    imask = i < n_ids
    idx = tl.load(ids_ptr + i, mask=imask, other=0).to(tl.int64)
    ok = imask & (idx >= 0) & (idx < n_rows)
    idx = tl.where(ok, idx, 0)
    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    wmask = w < row_words
    src = table_ptr + idx[:, None] * row_words + w[None, :]
    v = tl.load(src, mask=ok[:, None] & wmask[None, :], other=0)
    dst = out_ptr + i.to(tl.int64)[:, None] * row_words + w[None, :]
    tl.store(dst, v, mask=imask[:, None] & wmask[None, :])

def gather_rows(table_uva_words, ids, out_words, n_rows):
    n = ids.numel()
    if n == 0:
        return out_words
    W = table_uva_words.shape[1]
    if W <= 64:
        BLOCK_W, ROWS = (triton.next_power_of_2(W), 16)
    else:
        BLOCK_W, ROWS = (256, 4)
    grid = (triton.cdiv(n, ROWS), triton.cdiv(W, BLOCK_W))
    _gather_rows_kernel[grid](table_uva_words, ids, out_words, n, n_rows, W, ROWS=ROWS, BLOCK_W=BLOCK_W, num_warps=4)
    return out_words

class _StoreReadyMethod(QuantizeMethodBase):
    requires_device_loading = False

    def __init__(self, check=None):
        super().__init__()
        self._check = check

    def create_weights(self, *a, **k):
        raise NotImplementedError

    def apply(self, *a, **k):
        raise NotImplementedError

    def embedding(self, layer, x):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        _world().barrier()
        if self._check is not None:
            self._check(layer)

def _my_rows(n_rows):
    g = _world()
    W, r = (g.world_size, g.rank_in_group)
    per = (n_rows + W - 1) // W
    return (r * per, min(n_rows, (r + 1) * per))

def _copy_rows(dst, src, dst_start, lo, hi, fp8_scale=None):
    a, b = (max(lo, dst_start), min(hi, dst_start + src.shape[0]))
    if a >= b:
        return (0, 0)
    s = src[a - dst_start:b - dst_start]
    d = dst[a:b]
    clipped = 0
    if fp8_scale is not None and s.dtype != torch.float8_e4m3fn:
        dev = torch.device('cuda', torch.cuda.current_device())
        sc = torch.tensor(fp8_scale, dtype=torch.bfloat16, device=dev)
        step = 1 << 20
        for k in range(0, s.shape[0], step):
            x = s[k:k + step].to(dev, dtype=torch.bfloat16) / sc
            clipped += int((x.abs() > 448).sum())
            d[k:k + step].copy_(x.to(torch.float8_e4m3fn))
        torch.cuda.synchronize(dev)
    else:
        if s.dtype != d.dtype:
            raise RuntimeError(f'host table dtype {d.dtype} != checkpoint {s.dtype}')
        d.copy_(s)
    return (b - a, clipped)

class SharedHostPLEEmbedding(nn.Module):
    supports_prefetch = True
    shared_full_table = True

    def __init__(self, num_embeddings, embedding_dim, *, params_dtype, padding_size, prefix, embedding_method, num_ngram_heads=1, max_total_tokens=0, data_parallel_rank=0, reduce_results=True):
        super().__init__()
        from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEFp8EmbeddingMethod
        del padding_size, data_parallel_rank, reduce_results
        self.prefix = prefix
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.org_vocab_size = num_embeddings
        self.ckpt_fp8 = isinstance(embedding_method, Qwen4ExpPLEFp8EmbeddingMethod)
        self.to_fp8 = not self.ckpt_fp8 and _env('FN_NGRAM_FP8') == '1'
        self.fp8_scale = float(_env('FN_NGRAM_FP8_SCALE', str(QWEN_PLE_FP8_SCALE)))
        dtype = torch.float8_e4m3fn if self.ckpt_fp8 or self.to_fp8 else params_dtype
        self.store_dtype = dtype
        esz = torch.empty((), dtype=dtype).element_size()
        nbytes = num_embeddings * embedding_dim * esz
        if nbytes % 4 or embedding_dim * esz % 4:
            raise RuntimeError('PLE rows must be a multiple of 4 bytes')
        key = prefix.replace('.', '_')
        self._key = key
        raw, _ = HostStore.get(key, nbytes, async_register=_env('FN_NGRAM_ASYNC_REG', '1') == '1')
        self._raw = raw
        self.weight = nn.Parameter(raw.view(dtype).view(num_embeddings, embedding_dim), requires_grad=False)
        self.weight.weight_loader = self._load
        self._uva_words = None
        self._row_words = embedding_dim * esz // 4
        self._lo, self._hi = _my_rows(num_embeddings)
        dev = torch.device('cuda', torch.cuda.current_device())
        if dtype == torch.float8_e4m3fn:
            if self.to_fp8:
                self.register_buffer('weight_scale', torch.full((1,), self.fp8_scale, dtype=torch.float32, device=dev), persistent=False)
            else:
                self.weight_scale = nn.Parameter(torch.full((1,), torch.finfo(torch.float32).min, dtype=torch.float32, device=dev), requires_grad=False)
                self.weight_scale.weight_loader = self._load_scale
        self.quant_method = _StoreReadyMethod(self._post_load)
        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * embedding_dim
        self._prefetch_rows = max_total_tokens
        self._graph = _env('FN_NGRAM_GRAPH') == '1'
        self._stream = None
        self._buf = None
        self._ids_buf = None
        self._n_loaded = 0
        self._n_clipped = 0
        self._alloc_lock = threading.Lock()
        logger.info('shared PLE %s: %d x %d %s (%s), node-shared, rank slice [%d, %d), graph=%s', prefix, num_embeddings, embedding_dim, dtype, 'bf16->fp8' if self.to_fp8 else 'as checkpoint', self._lo, self._hi, self._graph)

    def _load(self, param, loaded_weight, checkpoint_start=None):
        if checkpoint_start is None:
            checkpoint_start = 0
        n, c = _copy_rows(self.weight.data, loaded_weight, checkpoint_start, self._lo, self._hi, fp8_scale=self.fp8_scale if self.to_fp8 else None)
        self._n_loaded += n
        self._n_clipped += c

    def _load_scale(self, param, loaded_weight):
        if self.to_fp8:
            return
        param.data.copy_(loaded_weight.reshape(param.shape).to(param.dtype))

    def _post_load(self, layer):
        HostStore.ready(self._key)
        self._uva_words = _uva(self._raw).view(torch.int32).view(self.num_embeddings, self._row_words)
        if hasattr(self, 'weight_scale') and torch.any(self.weight_scale == torch.finfo(torch.float32).min):
            raise ValueError('FP8 PLE checkpoint is missing its global scale')
        logger.info('shared PLE %s: this rank wrote %d rows (%d values clipped above 448)', self.prefix, self._n_loaded, self._n_clipped)

    def dequantize(self, embeddings, output_dtype):
        if self.store_dtype != torch.float8_e4m3fn:
            return embeddings
        return embeddings.to(output_dtype) * self.weight_scale.to(output_dtype)

    def _ensure(self, device):
        if self._buf is not None:
            return
        with self._alloc_lock:
            if self._buf is not None:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('PLE buffers must be allocated before CUDA graph capture')
            self._stream = torch.cuda.Stream(device=device)
            self._buf = torch.empty(self._prefetch_rows, self._num_ngram_heads, self.embedding_dim, dtype=self.store_dtype, device=device)

    def lookup(self, ngram_ids, out=None):
        if out is None:
            out = torch.empty(*ngram_ids.shape, self.embedding_dim, dtype=self.store_dtype, device=ngram_ids.device)
        ow = out.view(torch.uint8).view(-1, self.embedding_dim * out.element_size()).view(torch.int32)
        gather_rows(self._uva_words, ngram_ids.reshape(-1), ow, self.num_embeddings)
        return out

    def _start(self, ngram_ids):
        self._ensure(ngram_ids.device)
        n = ngram_ids.shape[0]
        if n > self._buf.shape[0]:
            raise ValueError(f'PLE buffer holds {self._buf.shape[0]} tokens, batch needs {n}')
        s = self._stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.lookup(ngram_ids, out=self._buf[:n])

    def _finish(self, out):
        torch.cuda.current_stream().wait_stream(self._stream)
        out.copy_(self._buf[:out.shape[0]].flatten(-2))

    def start_prefetch(self, ngram_ids):
        if self._graph:
            self._ensure(ngram_ids.device)
            self._pending = ngram_ids
        else:
            _eager_start(self, ngram_ids)

    def finalize_prefetch(self, total_tokens):
        out = self._buf.new_empty((total_tokens, self._output_dim))
        if self._graph:
            ids = self._pending
            self.lookup(ids[:total_tokens], out=out.view(total_tokens, self._num_ngram_heads, self.embedding_dim))
        else:
            _eager_finish(self, out)
        return out

    def forward(self, input_):
        raise RuntimeError('shared PLE requires start_prefetch() and finalize_prefetch()')

def _eager_wrap():
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    @eager_break_during_capture
    def start(mod, ngram_ids):
        mod._start(ngram_ids)

    @eager_break_during_capture
    def finish(mod, out):
        mod._finish(out)
    return (start, finish)
_EAGER = None

def _eager_start(mod, ids):
    global _EAGER
    if _EAGER is None:
        _EAGER = _eager_wrap()
    _EAGER[0](mod, ids)

def _eager_finish(mod, out):
    global _EAGER
    if _EAGER is None:
        _EAGER = _eager_wrap()
    _EAGER[1](mod, out)

class SharedHostTokenEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, params_dtype=None, org_num_embeddings=None, padding_size=64, quant_config=None, prefix='', **kw):
        super().__init__()
        del org_num_embeddings, padding_size, quant_config, kw
        dtype = params_dtype or torch.get_default_dtype()
        self.num_embeddings = self.org_vocab_size = self.num_embeddings_padded = num_embeddings
        self.embedding_dim = embedding_dim
        esz = torch.empty((), dtype=dtype).element_size()
        nbytes = num_embeddings * embedding_dim * esz
        raw, fresh = HostStore.get(f'embed_tokens_{num_embeddings}x{embedding_dim}_{str(dtype)[6:]}', nbytes)
        self.weight = nn.Parameter(raw.view(dtype).view(num_embeddings, embedding_dim), requires_grad=False)
        self.weight.weight_loader = self._load
        self._uva_words = _uva(raw).view(torch.int32).view(num_embeddings, embedding_dim * esz // 4)
        self._lo, self._hi = _my_rows(num_embeddings)
        self.quant_method = _StoreReadyMethod()
        self._dtype = dtype
        logger.info('host token embedding %s: %d x %d %s, node-shared (%s)', prefix, num_embeddings, embedding_dim, dtype, 'new' if fresh else 'reused')

    def _load(self, param, loaded_weight, *a, **k):
        if loaded_weight.shape[0] < self.num_embeddings:
            raise RuntimeError(f'embed_tokens checkpoint rows {loaded_weight.shape[0]} < {self.num_embeddings}')
        _copy_rows(self.weight.data, loaded_weight[:self.num_embeddings].to(self._dtype), 0, self._lo, self._hi)

    def forward(self, input_):
        ids = input_.reshape(-1)
        out = torch.empty(ids.numel(), self.embedding_dim, dtype=self._dtype, device=ids.device)
        gather_rows(self._uva_words, ids, out.view(torch.int32), self.num_embeddings)
        return out.view(*input_.shape, self.embedding_dim)

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, host_shared=True'
