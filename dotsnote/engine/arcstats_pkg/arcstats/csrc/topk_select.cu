// Exact top-k selection for the TP-sharded DSA indexer (arcstats/fast_topk.py, ARC_FAST_TOPK).
//
// local_select: one CTA per logits row [Lc] (fp32 or fp16, column j = position j*P+rank). Only the row's valid
//   columns j < ceil((ctx-rank)/P) are read (the -inf tail is never touched, so the logits kernel need not write it).
//   Radix select on PyTorch's order-preserving key (topk's TopKTypeConfig: -0 < +0, NaN largest): digit passes of
//   11/11/10 bits with a shared-memory histogram (warp-aggregated atomics), stopping early once the boundary bin is
//   taken whole; then one ordered compaction pass: keys above the threshold plus the LOWEST-index columns equal to
//   it (torch.topk's tie rule), written in ascending column order straight into the all-gather payload
//   pk [rows, 2*kl] = [values fp32 | positions int32] (replaces topk + pos arithmetic + cat). Rows with <= kl valid
//   columns are copied whole and padded with -inf.
// merge_select: one CTA per row of the all-gathered [P, n, 2*kl] payload: bitonic sort of the P*kl (key, concat
//   index) pairs, descending key then ascending index (= torch.topk(sorted=True): stable order, lowest-index ties),
//   first kk written as positions (-1 where the value is +-inf), [kk, topk_tokens) = -1.
// Both are graph-capturable (no host sync) and run on the caller's stream.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t rkey(float v) {
  uint32_t x = __float_as_uint(v);
  uint32_t m = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (v == v) ? (x ^ m) : 0xffffffffu;
}
__device__ __forceinline__ uint32_t rkey(__half v) {   // 16-bit key in the top half (low 16 bits zero)
  uint32_t x = __half_as_ushort(v);
  uint32_t m = (x & 0x8000u) ? 0xffffu : 0x8000u;
  return (__hisnan(v) ? 0xffffu : (x ^ m)) << 16;
}
__device__ __forceinline__ float tofl(float v) { return v; }
__device__ __forceinline__ float tofl(__half v) { return __half2float(v); }

template <typename T, int VEC>
__device__ __forceinline__ void loadv(const T* p, T (&o)[VEC]) {
  if constexpr (sizeof(T) * VEC == 16) {
    uint4 u = *reinterpret_cast<const uint4*>(p);
    *reinterpret_cast<uint4*>(o) = u;
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) o[i] = p[i];
  }
}

// Block-wide exclusive scan of uint32 (NT threads); returns exclusive prefix, *total = block sum.
template <int NT>
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t v, uint32_t* wsum, uint32_t* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t t = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += t;
  }
  if (lane == 31) wsum[warp] = inc;
  __syncthreads();
  if (warp == 0) {
    uint32_t w = lane < NT / 32 ? wsum[lane] : 0u, wi = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t t = __shfl_up_sync(0xffffffffu, wi, o);
      if (lane >= o) wi += t;
    }
    if (lane < NT / 32) wsum[lane] = wi - w;   // exclusive warp offsets
    if (lane == NT / 32 - 1) wsum[32] = wi;
  }
  __syncthreads();
  uint32_t r = wsum[warp] + inc - v;
  *total = wsum[32];
  __syncthreads();
  return r;
}

constexpr int NB = 2048;   // 11-bit digits

template <typename T, int VEC, int NT, bool AGG, int U = (sizeof(T) == 2 && VEC > 1 ? 2 : 4)>
__global__ void __launch_bounds__(NT) local_select_kernel(const T* __restrict__ lg, long long stride,
                                                          const int* __restrict__ ctx, int rank, int P, int Lc, int kl,
                                                          float* __restrict__ pk) {
  __shared__ uint32_t hist[NB];
  __shared__ uint32_t wsum[33];
  __shared__ uint32_t sh_sel[3];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
  const int c = ctx[row];
  int n = c > rank ? (c - rank + P - 1) / P : 0;
  if (n > Lc) n = Lc;
  const T* x = lg + (long long)row * stride;
  float* pv = pk + (long long)row * 2 * kl;
  int* pp = reinterpret_cast<int*>(pv + kl);
  uint32_t prefix = 0, mask = 0;
  uint32_t need = (uint32_t)(n < kl ? n : kl);
  if (n > kl) {
    constexpr int NPASS = sizeof(T) == 2 ? 2 : 3;
#pragma unroll 1
    for (int pass = 0; pass < NPASS; ++pass) {
      const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
      const uint32_t dm = pass == 2 ? 0x3ffu : 0x7ffu;
      for (int i = tid; i < NB; i += NT) hist[i] = 0;
      __syncthreads();
      for (int b0 = tid * VEC; b0 < n; b0 += NT * VEC * U) {   // U coalesced vectors in flight per thread
        T v[U][VEC];
#pragma unroll
        for (int u = 0; u < U; ++u)
          if (b0 + u * NT * VEC < n) loadv<T, VEC>(x + b0 + u * NT * VEC, v[u]);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int base = b0 + u * NT * VEC;
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            const uint32_t k = rkey(v[u][e]);
            const bool ok = (base + e < n) && ((k & mask) == prefix);
            const uint32_t d = (k >> shift) & dm;
            if (AGG) {
              const uint32_t am = __ballot_sync(__activemask(), ok);
              if (ok) {
                const uint32_t g = __match_any_sync(am, d);
                if (lane == __ffs(g) - 1) atomicAdd(&hist[d], (uint32_t)__popc(g));
              }
            } else if (ok) {
              atomicAdd(&hist[d], 1u);
            }
          }
        }
      }
      __syncthreads();
      // thread t owns bins NB-1-t*BPT .. NB-BPT-t*BPT (descending); suffix counts via a block scan
      constexpr int BPT = NB / NT;
      uint32_t hb[BPT], s = 0;
#pragma unroll
      for (int j = 0; j < BPT; ++j) { hb[j] = hist[NB - 1 - tid * BPT - j]; s += hb[j]; }
      uint32_t tot;
      uint32_t before = block_excl_scan<NT>(s, wsum, &tot);
#pragma unroll
      for (int j = 0; j < BPT; ++j) {
        if (before < need && before + hb[j] >= need) {
          sh_sel[0] = (uint32_t)(NB - 1 - tid * BPT - j);
          sh_sel[1] = need - before;   // still needed from inside this bin
          sh_sel[2] = hb[j];
        }
        before += hb[j];
      }
      __syncthreads();
      const uint32_t d = sh_sel[0];
      need = sh_sel[1];
      prefix |= d << shift;
      mask |= dm << shift;
      const bool whole = sh_sel[2] == need;
      __syncthreads();
      if (whole) break;
    }
  }
  // ordered compaction: slot = (#above before) + min(#equal before, need)
  uint32_t G = 0, E = 0;
  const uint32_t kfill = (uint32_t)(n < kl ? n : kl);
  constexpr int CH = VEC * U;   // contiguous elements per thread per tile (ascending-order scan)
  for (int tb = 0; tb < n; tb += NT * CH) {
    const int base = tb + tid * CH;
    T v[CH];
    uint32_t cnt = 0, kk[CH];
#pragma unroll
    for (int u = 0; u < U; ++u)
      if (base + u * VEC < n) loadv<T, VEC>(x + base + u * VEC, *reinterpret_cast<T(*)[VEC]>(v + u * VEC));
#pragma unroll
    for (int e = 0; e < CH; ++e) {
      kk[e] = rkey(v[e]);
      if (base + e < n) {
        const uint32_t km = kk[e] & mask;
        cnt += km > prefix ? 1u : (km == prefix ? 0x10000u : 0u);
      }
    }
    uint32_t tot;
    const uint32_t ex = block_excl_scan<NT>(cnt, wsum, &tot);
    uint32_t gb = G + (ex & 0xffffu), eb = E + (ex >> 16);
    if (base < n) {
#pragma unroll
      for (int e = 0; e < CH; ++e) {
        if (base + e >= n) break;
        const uint32_t km = kk[e] & mask;
        int slot = -1;
        if (km > prefix) {
          slot = (int)(gb + (eb < need ? eb : need));
          ++gb;
        } else if (km == prefix) {
          if (eb < need) slot = (int)(gb + eb);
          ++eb;
        }
        if (slot >= 0) {
          pv[slot] = tofl(v[e]);
          pp[slot] = (base + e) * P + rank;
        }
      }
    }
    G += tot & 0xffffu;
    E += tot >> 16;
    if (G + (E < need ? E : need) >= kfill) break;   // every slot written
  }
  for (int s = n + tid; s < kl; s += NT) {   // fewer valid columns than kl: -inf padding (merged away as -1)
    pv[s] = -__int_as_float(0x7f800000);
    pp[s] = s * P + rank;
  }
}

template <int NT>
__global__ void __launch_bounds__(NT) merge_select_kernel(const float* __restrict__ allp, int n, int P, int kl, int kk,
                                                          int topk_tokens, int* __restrict__ out, long long ostride,
                                                          int npad) {
  extern __shared__ unsigned long long sk[];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int M = P * kl;
  for (int c = tid; c < npad; c += NT) {
    unsigned long long v = 0ull;
    if (c < M) {
      const int p = c / kl, i = c - p * kl;
      const float f = allp[((long long)p * n + row) * 2 * kl + i];
      v = ((unsigned long long)rkey(f) << 32) | (unsigned long long)(0xffffffffu - (uint32_t)c);
    }
    sk[c] = v;
  }
  __syncthreads();
  for (int size = 2; size <= npad; size <<= 1) {   // bitonic sort, descending
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int t = tid; t < npad / 2; t += NT) {
        const int lo = 2 * t - (t & (stride - 1));
        const int hi = lo + stride;
        const bool desc = (lo & size) == 0;
        const unsigned long long a = sk[lo], b = sk[hi];
        if ((a < b) == desc) { sk[lo] = b; sk[hi] = a; }
      }
      __syncthreads();
    }
  }
  int* o = out + (long long)row * ostride;
  for (int s = tid; s < topk_tokens; s += NT) {
    int r = -1;
    if (s < kk) {
      const int c = (int)(0xffffffffu - (uint32_t)(sk[s] & 0xffffffffull));
      const int p = c / kl, i = c - p * kl;
      const float* b = allp + ((long long)p * n + row) * 2 * kl;
      const float f = b[i];
      if (!isinf(f)) r = reinterpret_cast<const int*>(b + kl)[i];
    }
    o[s] = r;
  }
}

}  // namespace

extern "C" {

// dtype 0 = fp32, 1 = fp16; vec 1 or 4/8 (16-byte loads: caller checks alignment); agg: warp-aggregated atomics
int topk_local_launch(const void* lg, long long stride, const int* ctx, int rows, int rank, int P, int Lc, int kl,
                      float* pk, int dtype, int vec, int nt, int agg, void* stream) {
  if (rows <= 0) return 0;
  cudaStream_t s = (cudaStream_t)stream;
#define ARC_L(T, V, N, A) local_select_kernel<T, V, N, A><<<rows, N, 0, s>>>((const T*)lg, stride, ctx, rank, P, Lc, kl, pk)
#define ARC_LA(T, V, N) do { if (agg) ARC_L(T, V, N, true); else ARC_L(T, V, N, false); } while (0)
#define ARC_LN(T, V) do { if (nt == 1024) ARC_LA(T, V, 1024); else if (nt == 256) ARC_LA(T, V, 256); else ARC_LA(T, V, 512); } while (0)
  if (dtype == 0) {
    if (vec == 4) ARC_LN(float, 4); else ARC_LN(float, 1);
  } else {
    if (vec == 8) ARC_LN(__half, 8); else ARC_LN(__half, 1);
  }
#undef ARC_LN
#undef ARC_LA
#undef ARC_L
  return (int)cudaGetLastError();
}

int topk_merge_launch(const float* allp, int n, int P, int kl, int kk, int topk_tokens, int* out, long long ostride,
                      int nt, void* stream) {
  if (n <= 0) return 0;
  int npad = 1;
  while (npad < P * kl) npad <<= 1;
  const size_t smem = (size_t)npad * 8;
  cudaStream_t s = (cudaStream_t)stream;
  if (nt == 512) {
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(merge_select_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    merge_select_kernel<512><<<n, 512, smem, s>>>(allp, n, P, kl, kk, topk_tokens, out, ostride, npad);
  } else {
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(merge_select_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    merge_select_kernel<1024><<<n, 1024, smem, s>>>(allp, n, P, kl, kk, topk_tokens, out, ostride, npad);
  }
  return (int)cudaGetLastError();
}
}
