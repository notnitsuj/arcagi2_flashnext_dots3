// Sparse MLA decode attention over FP8 offload rows (sm_89+), split-K partials for _combine_kernel.
// One CTA = (token t, split sp): the token's 32 heads (M = 32) against TOPK/SPLIT selected rows, BN = 32 rows per
// tile. Rows are 576 B FP8 (512 latent + 64 rope), gathered with cp.async from the resolved location (row cache line,
// spill row or host row through UVA) into shared memory and fed to FP16 mma.sync straight from the FP8 bytes:
//   Q.K^T: B fragments = 4 consecutive bytes of a row (ldmatrix), the 16-dim k order is permuted consistently in Q
//          (staged in shared memory as FP16 in that order) -> no transpose, one cvt per 2 elements
//   P.V:   B fragments via ldmatrix.trans on byte pairs + one prmt per 2 regs (even/odd latent columns land in two
//          n8 tiles; the accumulator layout gives each thread 4 contiguous columns)
// res == nullptr: rows come straight from host[idx] (prefill over a staged row buffer).
// Numerics = the Triton split kernel with CD=float16: FP16 q / K / V / p, FP32 accumulation, exp2 online softmax.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int H = 32, D = 576, DN = 512, BN = 32;
constexpr int QS = D + 8;            // Q row stride (halves): 1168 B -> conflict-free ldmatrix
constexpr int KS = D + 16;           // K row stride (bytes): 592 B -> conflict-free ldmatrix / ldmatrix.trans
constexpr int PS = BN + 8;           // P row stride (halves): 80 B

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async16ca(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// two e4m3 bytes (low 16 bits of x, or high) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t e4m3x2_lo(uint32_t x) {
  uint32_t r;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}\n" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2_hi(uint32_t x) { return e4m3x2_lo(x >> 16); }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// ---- compact host rows (ARC_OFFLOAD_ROW_FMT, arcstats/offload_rowfmt.py): sign + E2M{M} latent codes with a 2^k
// exponent per BLK values, rope as e4m3 bytes. Readers of HOST rows dequantize 16-byte chunks of today's 576 B fp8 row
// (chunk c < 32: latent values 16c..16c+15, c >= 32: rope bytes) so the GPU row cache / spill / smem tiles keep their
// format. fmt = M | BLK << 4 (0 = plain fp8 host rows: the original code paths, compiled separately).
struct HostFmt {
  int M, blk, soff, roff, rb;
};
__host__ __device__ inline HostFmt host_fmt(int fmt) {
  HostFmt f;
  f.M = fmt & 15;
  f.blk = (fmt >> 4) & 255;   // (bit 16: ARC_OFFLOAD_CCACHE compact GPU rows)
  const int hb = f.M == 3 ? 128 : (f.M == 2 ? 64 : 0);
  f.soff = 256 + hb;
  f.roff = f.blk > 0 ? f.soff + 512 / f.blk : 0;
  f.rb = f.roff + 64;
  return f;
}
__device__ __forceinline__ float hf_val(uint32_t code, int M, int k) {
  const uint32_t mag = code & ((1u << (M + 2)) - 1);
  const int e2 = (int)(mag >> M);
  const uint32_t mm = mag & ((1u << M) - 1);
  const uint32_t mant = e2 ? (mm | (1u << M)) : mm;
  int ex = (e2 > 1 ? e2 : 1) - 1 - M + k;
  ex = ex < -126 ? -126 : (ex > 127 ? 127 : ex);
  float v = (float)mant * __int_as_float((ex + 127) << 23);
  v = fminf(v, 448.f);
  return ((code >> (M + 2)) & 1) ? -v : v;
}
// 16 latent values (lo nibbles, hi bits, block exponent k) -> 16 e4m3 bytes
__device__ __forceinline__ int4 hf_dq16(unsigned long long lo, uint32_t hi, int k, const HostFmt f) {
  uint32_t w[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t b4[4];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int j0 = 4 * q + 2 * p;
      float v[2];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int j = j0 + s;
        uint32_t code = (uint32_t)(lo >> (4 * j)) & 15u;
        if (f.M == 3) code |= ((hi >> (2 * j)) & 3u) << 4;
        else if (f.M == 2) code |= ((hi >> j) & 1u) << 4;
        v[s] = hf_val(code, f.M, k);
      }
      const __nv_fp8x2_storage_t r = __nv_cvt_float2_to_fp8x2(make_float2(v[0], v[1]), __NV_SATFINITE, __NV_E4M3);
      b4[2 * p] = r & 0xFF;
      b4[2 * p + 1] = (r >> 8) & 0xFF;
    }
    w[q] = b4[0] | (b4[1] << 8) | (b4[2] << 16) | (b4[3] << 24);
  }
  return make_int4((int)w[0], (int)w[1], (int)w[2], (int)w[3]);
}
// chunk c (0..35) of the fp8 row encoded by the compact host row `row` (16 B aligned): per-thread loads
__device__ __forceinline__ int4 hf_chunk(const uint8_t* row, int c, const HostFmt f) {
  if (c >= 32) return *reinterpret_cast<const int4*>(row + f.roff + (c - 32) * 16);
  const unsigned long long lo = *reinterpret_cast<const unsigned long long*>(row + 8 * c);
  uint32_t hi = 0;
  if (f.M == 3) hi = *reinterpret_cast<const uint32_t*>(row + 256 + 4 * c);
  else if (f.M == 2) hi = *reinterpret_cast<const uint16_t*>(row + 256 + 2 * c);
  const int k = (int)*reinterpret_cast<const int8_t*>(row + f.soff + (16 * c) / f.blk);
  return hf_dq16(lo, hi, k, f);
}
// warp form: lane l holds raw = bytes [16 l, 16 l + 16) of the compact row (one contiguous warp read of the whole row,
// as few PCIe read requests as possible); returns chunk `lane` in a and, for lane < 4, rope chunk 32 + lane in b.
// Every lane of the warp must call it (shuffles).
__device__ __forceinline__ uint32_t shfl_word(const int4 raw, int off) {   // the 32-bit word at byte offset off (4 B aligned)
  const int src = off >> 4, wd = (off >> 2) & 3;
  const uint32_t x = __shfl_sync(0xffffffffu, (uint32_t)raw.x, src), y = __shfl_sync(0xffffffffu, (uint32_t)raw.y, src),
                 z = __shfl_sync(0xffffffffu, (uint32_t)raw.z, src), w = __shfl_sync(0xffffffffu, (uint32_t)raw.w, src);
  return wd == 0 ? x : (wd == 1 ? y : (wd == 2 ? z : w));
}
__device__ __forceinline__ void hf_warp(const int4 raw, int lane, const HostFmt f, int4& a, int4& b) {
  const int s8 = lane >> 1;   // lo bytes [8 lane, +8) = int4 of lane / 2, words 2 (lane & 1), +1
  const uint32_t x = __shfl_sync(0xffffffffu, (uint32_t)raw.x, s8), y = __shfl_sync(0xffffffffu, (uint32_t)raw.y, s8),
                 z = __shfl_sync(0xffffffffu, (uint32_t)raw.z, s8), w = __shfl_sync(0xffffffffu, (uint32_t)raw.w, s8);
  const unsigned long long lo = (lane & 1) ? ((unsigned long long)w << 32 | z) : ((unsigned long long)y << 32 | x);
  uint32_t hi = 0;
  if (f.M == 3) hi = shfl_word(raw, 256 + 4 * lane);
  else if (f.M == 2) hi = (shfl_word(raw, 256 + ((2 * lane) & ~3)) >> (16 * (lane & 1))) & 0xFFFFu;
  const int so = f.soff + (16 * lane) / f.blk;
  const int k = (int)(int8_t)((shfl_word(raw, so & ~3) >> (8 * (so & 3))) & 0xFFu);
  a = hf_dq16(lo, hi, k, f);
  const int rs = (f.roff >> 4) + (lane & 3);
  b = make_int4((int)__shfl_sync(0xffffffffu, (uint32_t)raw.x, rs), (int)__shfl_sync(0xffffffffu, (uint32_t)raw.y, rs),
                (int)__shfl_sync(0xffffffffu, (uint32_t)raw.z, rs), (int)__shfl_sync(0xffffffffu, (uint32_t)raw.w, rs));
}

// ---- overlapped row-cache fill (ARC_OFFLOAD_OVERLAP): the fill copy runs on a side stream concurrently with the
// attention; rows it may still be copying are marked in res (line + FRESH; every spill row) and carry a flag:
// 1 = pending, 2 = being copied, 0 = written. Whoever moves a flag 1 -> 2 (copy kernel or the attention's loader
// threads of that row) copies the row, fences and sets 0; the others wait only on 2. No one ever waits on a pending
// row nobody has claimed, so the attention cannot deadlock on copy CTAs that are not resident.
constexpr int FRESH = 1 << 30;
__device__ __forceinline__ int ld_acquire(const int* f) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(f) : "memory");
  return v;
}
__device__ __forceinline__ void wait_flag(const int* f) {
  while (ld_acquire(f) != 0) __nanosleep(64);
}
// the G = LPT loader threads of one row (consecutive lanes): claim and copy the row host -> dst, or wait for its copier
template <int G, bool HC = false>
__device__ __forceinline__ void fetch_or_wait(int* f, const uint8_t* src, uint8_t* dst, int sub,
                                              const HostFmt hf = HostFmt{}) {
  const int lane = threadIdx.x & 31;
  const unsigned gmask = ((1u << G) - 1) << (lane & ~(G - 1));
  int st = 0;
  if (sub == 0) st = atomicCAS(f, 1, 2);
  st = __shfl_sync(gmask, st, lane & ~(G - 1));
  if (st == 1) {
    for (int c = sub; c < 36; c += G)
      reinterpret_cast<int4*>(dst)[c] = HC ? hf_chunk(src, c, hf) : reinterpret_cast<const int4*>(src)[c];
    __threadfence();
    __syncwarp(gmask);
    if (sub == 0) atomicExch(f, 0);
  }
  wait_flag(f);
}

template <int ST, int NW>
constexpr int smem_bytes() { return H * QS * 2 + ST * BN * KS + H * PS * 2 + 2 * 4 * H * 4 + ST * BN; }

// NW = 4: warp w -> S columns 8w..8w+7 of both 16-head m-blocks, P.V latent columns [128 w, +128)
// NW = 8: warp w -> S columns 8 (w & 3).. of m-block w >> 2, P.V latent columns [64 w, +64) (2x the warps per SM,
//         K fragments converted twice)
template <int ST, int NW, bool HC = false>
__global__ void __launch_bounds__(NW * 32, 1)
offload_split_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host,
                     const uint8_t* __restrict__ cache, const uint8_t* __restrict__ spill,
                     const int* __restrict__ idx, const int* __restrict__ res,
                     float* __restrict__ m_out, float* __restrict__ l_out, float* __restrict__ acc_out,
                     long long R, long long stride_q_t, long long stride_q_h, long long stride_i_t,
                     int topk, int split, float sm_scale, __nv_bfloat16* __restrict__ out, long long sot,
                     long long soh, float kv_scale, int* lflag, int* sflag, int fmt = 0) {
  constexpr int NT = NW * 32;
  const HostFmt hf = host_fmt(fmt);
  const long long HD = HC ? hf.rb : D;          // host row stride (bytes)
  constexpr int MPW = NW == 4 ? 2 : 1;          // 16-head m-blocks per warp in Q.K
  constexpr int CBW = DN / NW / 16;             // 16-column latent blocks per warp in P.V
  constexpr int LPT = NT / BN;                  // loader threads per tile row
  extern __shared__ __align__(16) uint8_t smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);                          // [H][QS]
  uint8_t* Ks = smem + H * QS * 2;                                       // [ST][BN][KS]
  __half* Ps = reinterpret_cast<__half*>(Ks + ST * BN * KS);             // [H][PS]
  float* red = reinterpret_cast<float*>(Ps + H * PS);                    // [4 column blocks][H]
  float* redl = red + 4 * H;                                             // [4][H]
  uint8_t* vflag = reinterpret_cast<uint8_t*>(redl + 4 * H);             // [ST][BN]

  const int t = blockIdx.x, sp = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int nbw = warp & 3, mb0 = NW == 4 ? 0 : warp >> 2;
  const int chunk = topk / split, row0 = sp * chunk, ntiles = chunk / BN;
  const int* idx_t = idx + (long long)t * stride_i_t + row0;
  const int* res_t = res ? res + (long long)t * topk + row0 : nullptr;   // null: every row read from host[h]

  // ---- loader role: thread -> tile row (tid / LPT), 16 B chunks (tid % LPT) + LPT k
  const int lrow = tid / LPT, lsub = tid % LPT;
  // HC: host rows are compact (hostc = the returned row is one): dequantized by the loader threads, not cp.async
  auto row_src = [&](int tile, bool& valid, bool& hostc) -> const uint8_t* {
    int h = idx_t[tile * BN + lrow];
    valid = (h >= 0) && ((long long)h < R);
    hostc = false;
    int r = (valid && res_t) ? res_t[tile * BN + lrow] : -1;
    if (!valid) return cache;
    if (r >= 0) {
      if (r >= FRESH) {          // line being filled by the overlapped copy
        r -= FRESH;
        if (lflag) fetch_or_wait<LPT, HC>(lflag + r, host + (long long)h * HD, const_cast<uint8_t*>(cache) + (long long)r * D, lsub, hf);
      }
      return cache + (long long)r * D;
    }
    if (r <= -2) {
      if (sflag)
        fetch_or_wait<LPT, HC>(sflag + (-r - 2), host + (long long)h * HD, const_cast<uint8_t*>(spill) + (long long)(-r - 2) * D, lsub, hf);
      return spill + (long long)(-r - 2) * D;
    }
    hostc = HC && res_t != nullptr;
    return host + (long long)h * HD;
  };
  auto issue = [&](int tile) {
    if (tile < ntiles) {
      bool v, hc;
      const uint8_t* src = row_src(tile, v, hc);
      uint8_t* dst = Ks + (tile % ST) * BN * KS + lrow * KS;
      if (HC && hc) {
#pragma unroll
        for (int k = 0; k < (D / 16 + LPT - 1) / LPT; ++k) {
          int c = lsub + LPT * k;
          if (c < D / 16) *reinterpret_cast<int4*>(dst + c * 16) = hf_chunk(src, c, hf);
        }
      } else {
#pragma unroll
        for (int k = 0; k < (D / 16 + LPT - 1) / LPT; ++k) {
          int c = lsub + LPT * k;
          if (c < D / 16) cp_async16(smem_u32(dst + c * 16), src + c * 16, v);
        }
      }
      if (lsub == 0) vflag[(tile % ST) * BN + lrow] = v;
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) issue(s);

  // ---- Q (bf16 global) -> FP16 shared, dims of each 16-group in the k order of the K fragments:
  //      dims 4j..4j+3 -> positions 2j, 2j+1, 2j+8, 2j+9
  const __nv_bfloat16* qt = q + (long long)t * stride_q_t;
  for (int e = tid; e < H * (D / 4); e += NT) {
    int hh = e / (D / 4), c4 = e % (D / 4);
    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(qt + hh * stride_q_h + c4 * 4);
    float2 a = __bfloat1622float2(src[0]), b = __bfloat1622float2(src[1]);
    int grp = c4 >> 2, j = c4 & 3;
    uint32_t* dst = reinterpret_cast<uint32_t*>(Qs + hh * QS + grp * 16);
    dst[j] = pack_h2(a.x, a.y);
    dst[4 + j] = pack_h2(b.x, b.y);
  }

  float acc[2][2 * CBW][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 2 * CBW; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  float mrow[4];             // running max of rows g, g+8, 16+g, 24+g (all threads)
  float lrow_[2 * MPW];      // partial sums over this warp's S columns: rows of its m-block(s)
#pragma unroll
  for (int i = 0; i < 4; ++i) mrow[i] = -1.0e30f;
#pragma unroll
  for (int i = 0; i < 2 * MPW; ++i) lrow_[i] = 0.f;

  const uint32_t qs_base = smem_u32(Qs);
  for (int it = 0; it < ntiles; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    issue(it + ST - 1);
    const int stg = it % ST;
    const uint8_t* Kt = Ks + stg * BN * KS;
    const uint32_t kt_base = smem_u32(Kt);

    // ---- S = Q K^T for tile rows 8 nbw..8 nbw+7 (one n8 tile)
    float s[MPW][4];
#pragma unroll
    for (int a = 0; a < MPW; ++a) s[a][0] = s[a][1] = s[a][2] = s[a][3] = 0.f;
    // ldmatrix (non-trans) on K: matrix i = rows 8 nbw + (lane & 7), bytes 16 (kk + i) -> thread: row g, bytes 4tq..+3
    const uint32_t kaddr = kt_base + (nbw * 8 + (lane & 7)) * KS + (lane >> 3) * 16;
    // ldmatrix A on Q: matrices (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15)
    const uint32_t qaddr = qs_base + ((mb0 * 16 + (lane & 15)) * QS + (lane >> 4) * 8) * 2;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk += 4) {
      uint32_t kb[4];
      ldsm_x4(kaddr + kk * 16, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t b0 = e4m3x2_lo(kb[i]), b1 = e4m3x2_hi(kb[i]);
#pragma unroll
        for (int mb = 0; mb < MPW; ++mb) {
          uint32_t a0, a1, a2, a3;
          ldsm_x4(qaddr + (mb * 16 * QS + (kk + i) * 16) * 2, a0, a1, a2, a3);
          mma16816(s[mb], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    // ---- online softmax across the 4 column blocks
    const bool v0 = vflag[stg * BN + nbw * 8 + 2 * tq], v1 = vflag[stg * BN + nbw * 8 + 2 * tq + 1];
    float mx[2 * MPW];
#pragma unroll
    for (int mb = 0; mb < MPW; ++mb) {
      s[mb][0] = v0 ? s[mb][0] * sm_scale : -1.0e30f;
      s[mb][1] = v1 ? s[mb][1] * sm_scale : -1.0e30f;
      s[mb][2] = v0 ? s[mb][2] * sm_scale : -1.0e30f;
      s[mb][3] = v1 ? s[mb][3] * sm_scale : -1.0e30f;
      mx[2 * mb] = fmaxf(s[mb][0], s[mb][1]);
      mx[2 * mb + 1] = fmaxf(s[mb][2], s[mb][3]);
    }
#pragma unroll
    for (int i = 0; i < 2 * MPW; ++i) {
      mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 1));
      mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 2));
    }
    // value i of this warp: m-block mb0 + (i >> 1), half i & 1 -> head 16 (mb0 + (i >> 1)) + 8 (i & 1) + g
    if (tq == 0) {
#pragma unroll
      for (int i = 0; i < 2 * MPW; ++i) red[nbw * H + (mb0 + (i >> 1)) * 16 + (i & 1) * 8 + g] = mx[i];
    }
    __syncthreads();
    float re[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int hrow = (i >> 1) * 16 + (i & 1) * 8 + g;
      float tm = fmaxf(fmaxf(red[hrow], red[H + hrow]), fmaxf(red[2 * H + hrow], red[3 * H + hrow]));
      float mn = fmaxf(mrow[i], tm);
      re[i] = exp2f(mrow[i] - mn);
      mrow[i] = mn;
    }
#pragma unroll
    for (int mb = 0; mb < MPW; ++mb) {
      const int ma = mb0 + mb;
      float p0 = v0 ? exp2f(s[mb][0] - mrow[2 * ma]) : 0.f;
      float p1 = v1 ? exp2f(s[mb][1] - mrow[2 * ma]) : 0.f;
      float p2 = v0 ? exp2f(s[mb][2] - mrow[2 * ma + 1]) : 0.f;
      float p3 = v1 ? exp2f(s[mb][3] - mrow[2 * ma + 1]) : 0.f;
      lrow_[2 * mb] = lrow_[2 * mb] * re[2 * ma] + p0 + p1;
      lrow_[2 * mb + 1] = lrow_[2 * mb + 1] * re[2 * ma + 1] + p2 + p3;
      *reinterpret_cast<uint32_t*>(Ps + (ma * 16 + g) * PS + nbw * 8 + 2 * tq) = pack_h2(p0, p1);
      *reinterpret_cast<uint32_t*>(Ps + (ma * 16 + 8 + g) * PS + nbw * 8 + 2 * tq) = pack_h2(p2, p3);
    }
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
      for (int nb = 0; nb < 2 * CBW; ++nb) {
        acc[mb][nb][0] *= re[2 * mb];
        acc[mb][nb][1] *= re[2 * mb];
        acc[mb][nb][2] *= re[2 * mb + 1];
        acc[mb][nb][3] *= re[2 * mb + 1];
      }
    __syncthreads();
    // ---- O += P V over latent columns [16 CBW w, +16 CBW): blocks of 16 bytes, each -> 2 n8 tiles (even/odd)
    const uint32_t ps_base = smem_u32(Ps);
#pragma unroll
    for (int kc = 0; kc < BN / 16; ++kc) {
      uint32_t pa[2][4];
#pragma unroll
      for (int mb = 0; mb < 2; ++mb)
        ldsm_x4(ps_base + ((mb * 16 + (lane & 15)) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[mb][0], pa[mb][1],
                pa[mb][2], pa[mb][3]);
      // ldmatrix.trans: matrices (rows kc16 + 0-7, cb), (rows +8-15, cb), (rows 0-7, cb+1), (rows 8-15, cb+1)
      const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + warp * CBW * 16 +
                             (lane >> 4) * 16;
#pragma unroll
      for (int cb2 = 0; cb2 < CBW / 2; ++cb2) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
          uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
          uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);   // even / odd latent column
          uint32_t be0 = e4m3x2_lo(xe), be1 = e4m3x2_hi(xe), bo0 = e4m3x2_lo(xo), bo1 = e4m3x2_hi(xo);
          int cb = cb2 * 2 + h2;
#pragma unroll
          for (int mb = 0; mb < 2; ++mb) {
            mma16816(acc[mb][2 * cb], pa[mb][0], pa[mb][1], pa[mb][2], pa[mb][3], be0, be1);
            mma16816(acc[mb][2 * cb + 1], pa[mb][0], pa[mb][1], pa[mb][2], pa[mb][3], bo0, bo1);
          }
        }
      }
    }
  }
  cp_wait<0>();

  // ---- row sums: over tq lanes, then over the 4 column blocks
#pragma unroll
  for (int i = 0; i < 2 * MPW; ++i) {
    lrow_[i] += __shfl_xor_sync(0xffffffff, lrow_[i], 1);
    lrow_[i] += __shfl_xor_sync(0xffffffff, lrow_[i], 2);
  }
  if (tq == 0) {
#pragma unroll
    for (int i = 0; i < 2 * MPW; ++i) redl[nbw * H + (mb0 + (i >> 1)) * 16 + (i & 1) * 8 + g] = lrow_[i];
  }
  __syncthreads();
  float lsum[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int hrow = (i >> 1) * 16 + (i & 1) * 8 + g;
    lsum[i] = redl[hrow] + redl[H + hrow] + redl[2 * H + hrow] + redl[3 * H + hrow];
  }
  if (out != nullptr) {   // split == 1: normalized bf16 output, no partials / combine
    float inv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) inv[i] = kv_scale / fmaxf(lsum[i], 1e-30f);
    __nv_bfloat16* ot = out + (long long)t * sot;
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
      for (int cb = 0; cb < CBW; ++cb) {
        int col = warp * CBW * 16 + cb * 16 + 4 * tq;
        float a = inv[2 * mb], b = inv[2 * mb + 1];
        __nv_bfloat162 r0[2] = {__floats2bfloat162_rn(acc[mb][2 * cb][0] * a, acc[mb][2 * cb + 1][0] * a),
                                __floats2bfloat162_rn(acc[mb][2 * cb][1] * a, acc[mb][2 * cb + 1][1] * a)};
        __nv_bfloat162 r1[2] = {__floats2bfloat162_rn(acc[mb][2 * cb][2] * b, acc[mb][2 * cb + 1][2] * b),
                                __floats2bfloat162_rn(acc[mb][2 * cb][3] * b, acc[mb][2 * cb + 1][3] * b)};
        *reinterpret_cast<uint2*>(ot + (mb * 16 + g) * soh + col) = *reinterpret_cast<uint2*>(r0);
        *reinterpret_cast<uint2*>(ot + (mb * 16 + 8 + g) * soh + col) = *reinterpret_cast<uint2*>(r1);
      }
    return;
  }
  const long long ob = ((long long)t * split + sp) * H;
  if (warp == 0 && tq == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int hrow = (i >> 1) * 16 + (i & 1) * 8 + g;
      m_out[ob + hrow] = mrow[i];
      l_out[ob + hrow] = lsum[i];
    }
  }
  // thread holds, per 16-column block cb: columns 16 cb + 4 tq + {0: even c0, 1: odd c0, 2: even c1, 3: odd c1}
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int cb = 0; cb < CBW; ++cb) {
      int col = warp * CBW * 16 + cb * 16 + 4 * tq;
      float4 r0 = make_float4(acc[mb][2 * cb][0], acc[mb][2 * cb + 1][0], acc[mb][2 * cb][1], acc[mb][2 * cb + 1][1]);
      float4 r1 = make_float4(acc[mb][2 * cb][2], acc[mb][2 * cb + 1][2], acc[mb][2 * cb][3], acc[mb][2 * cb + 1][3]);
      *reinterpret_cast<float4*>(acc_out + (ob + mb * 16 + g) * DN + col) = r0;
      *reinterpret_cast<float4*>(acc_out + (ob + mb * 16 + 8 + g) * DN + col) = r1;
    }
}

template <int ST, int NW, bool HC = false>
int launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx, const void* res,
           void* m, void* l, void* acc, long long T, long long R, long long sqt, long long sqh, long long sit,
           int topk, int split, float sm_scale, void* out, long long sot, long long soh, float kv_scale,
           void* stream, const void* lflag = nullptr, const void* sflag = nullptr, int fmt = 0) {
  constexpr int SM = smem_bytes<ST, NW>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(offload_split_kernel<ST, NW, HC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  dim3 grid((unsigned)T, (unsigned)split);
  offload_split_kernel<ST, NW, HC><<<grid, NW * 32, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)idx,
      (const int*)res, (float*)m, (float*)l, (float*)acc, R, sqt, sqh, sit, topk, split, sm_scale,
      (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)lflag, (int*)sflag, fmt);
  return (int)cudaGetLastError();
}

// ARC_OFFLOAD_AR_GATE: a background copy warp waits while the device word *gate is non-zero
// (set by the host all-reduce around its kernel on the main stream), so the copy's host reads stay off the PCIe link
// during all-reduces. Lane 0 polls; bounded by GATE_MAX_NS per wait so a stale word can never hang the copy.
#define GATE_MAX_NS 4000000ull
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void gate_wait(const int* gate) {
  if ((threadIdx.x & 31) == 0 && *(volatile const int*)gate) {
    const unsigned long long t0 = gtimer();
    while (*(volatile const int*)gate && gtimer() - t0 < GATE_MAX_NS) __nanosleep(500);
  }
  __syncwarp();
}

// overlapped fill copy: a warp moves RPW rows (36 x 16 B each) per iteration, rows in miss-list order; d = cpd[i]:
// cache line >= 0, spill row -d - 2, -1 none. After the rows are written: fence, then their flags are cleared (the
// attention waits on them). gate (ARC_OFFLOAD_AR_GATE, mirror copies only): pause between iterations while *gate != 0.
template <int RPW, bool HC = false>
__global__ void __launch_bounds__(128) fill_copy_kernel(const long long* __restrict__ ml, const int* __restrict__ mcnt,
                                                        const long long* __restrict__ cpd, const uint8_t* host,
                                                        uint8_t* cache, uint8_t* spill, int* lflag, int* sflag,
                                                        const long long* __restrict__ perm, int ns, int fmt = 0,
                                                        const int* gate = nullptr) {
  const HostFmt hf = host_fmt(fmt);
  const bool RAWC = (fmt >> 16) & 1;   // ARC_OFFLOAD_CCACHE (compact GPU rows: plain copy of hf.rb bytes)
  // perm (ns entries): visit order of the first ns misses (miss index perm[i]); the rest in list order
  const int n = *mcnt;
  const int lane = threadIdx.x & 31;
  const int nw = gridDim.x * (blockDim.x >> 5);
  constexpr int XC = D / 16 - 32;   // chunks past the first 32 (4)
  for (int i0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW; i0 < n; i0 += nw * RPW) {
    if (gate) gate_wait(gate);
    long long d[RPW];
    int4 a[RPW], b[RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      const int k = i0 + j < ns ? (int)perm[i0 + j] : i0 + j;
      d[j] = i0 + j < n ? cpd[k] : -1;
      int st = 0;
      if (lane == 0 && d[j] != -1) st = lflag ? atomicCAS(d[j] >= 0 ? lflag + d[j] : sflag + (-d[j] - 2), 1, 2) : 1;
      if (__shfl_sync(0xffffffffu, st, 0) != 1) d[j] = -1;   // already claimed by an attention loader
      if (d[j] != -1) {
        if (HC) {   // compact host row: one contiguous warp read (lane -> bytes 16 lane..), dequantized below
          const uint8_t* row = host + ml[k] * hf.rb;
          a[j] = make_int4(0, 0, 0, 0);
          if (lane * 16 < hf.rb) a[j] = *reinterpret_cast<const int4*>(row + lane * 16);
        } else {
          const int4* src = reinterpret_cast<const int4*>(host + ml[k] * D);
          a[j] = src[lane];
          if (lane < XC) b[j] = src[32 + lane];
        }
      }
    }
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      if (HC && RAWC && d[j] != -1) {   // ARC_OFFLOAD_CCACHE: compact GPU rows, the host row's bytes as they are
        uint8_t* dst = d[j] >= 0 ? cache + d[j] * hf.rb : spill + (-d[j] - 2) * hf.rb;
        if (lane * 16 < hf.rb) reinterpret_cast<int4*>(dst)[lane] = a[j];
        continue;
      }
      if (HC && d[j] != -1) {   // (d[j] is warp-uniform)
        int4 raw = a[j];
        hf_warp(raw, lane, hf, a[j], b[j]);
      }
      if (d[j] != -1) {
        int4* dst = reinterpret_cast<int4*>(d[j] >= 0 ? cache + d[j] * D : spill + (-d[j] - 2) * D);
        dst[lane] = a[j];
        if (lane < XC) dst[32 + lane] = b[j];
      }
    }
    __threadfence();
    __syncwarp();
#pragma unroll
    for (int j = 0; j < RPW; ++j)
      if (lane == j && d[j] != -1 && lflag) atomicExch(d[j] >= 0 ? lflag + d[j] : sflag + (-d[j] - 2), 0);
  }
}

// =====================================================================================================================
// Request-packed decode (ARC_OFFLOAD_CUDA_PK). The G MTP tokens of a request share most of their top-k rows (union
// ~2400 of 4 x 2048 on real traffic), so each union row is loaded / converted once per request and head group.
// pk_union_kernel: CTA per request -> u = its distinct rows (-1 padded to nmax), mk = bitmask of the tokens that
//   selected each row, cnt = #distinct. Shared-memory hash set (row and token bits in one word), then a stable counting
//   sort by mask: rows selected by the same token subset are contiguous, so MMA blocks (n8 for Q.K, k16 for P.V) of a
//   token that selected none of their rows are skipped. Deterministic order (first occurrence, by mask bucket).
// pk_attn_kernel: CTA = (head group hb, union split sp, request r); M = 64 query rows = (64 / HG) tokens x HG heads
//   (row m -> token m / HG, head hb * HG + m % HG) against BN-row tiles of the union, 8 warps, per-(token, row)
//   membership mask. Fragments and numerics as offload_split_kernel (FP16 q / K / V / p, FP32 accumulation).
//   Partials use the split layout ((t * KS + sp) * 32 + head) so _combine_kernel merges them; KS == 1 + out: direct.
constexpr int UN_NT = 1024, UN_LOG = 14, UN_TBL = 1 << UN_LOG, UN_PPT = 12, UN_GB = 6, UN_NB = 1 << UN_GB;
constexpr int UN_SMEM = UN_TBL * 4 + 32 * UN_NB * 4 + UN_NB * 4 + 16;

__global__ void __launch_bounds__(UN_NT, 1)
pk_union_kernel(const int* __restrict__ idx, long long sit, int T, int ql, const int* __restrict__ cu, long long R,
                int topk, int nmax, int* __restrict__ u, uint8_t* __restrict__ mk, int* __restrict__ cnt) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint32_t* tbl = reinterpret_cast<uint32_t*>(smem);     // (row + 1) << UN_GB | token bits; 0 = empty
  int* wcnt = reinterpret_cast<int*>(tbl + UN_TBL);       // [32 warps][UN_NB] per-round counts -> offsets
  int* bcur = wcnt + 32 * UN_NB;                          // [UN_NB] bucket cursors
  int* tot = bcur + UN_NB;
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t0 = cu ? cu[r] : min(r * ql, T);
  const int t1 = cu ? cu[r + 1] : min(t0 + ql, T);
  const int G = min(max(t1 - t0, 0), UN_GB);
  const int n = min(G * topk, UN_PPT * UN_NT);
  for (int i = tid; i < UN_TBL; i += UN_NT) tbl[i] = 0u;
  if (tid < UN_NB) bcur[tid] = 0;
  __syncthreads();
  int ids[UN_PPT], sl[UN_PPT];
#pragma unroll
  for (int i = 0; i < UN_PPT; ++i) {
    const int p = tid + i * UN_NT;
    ids[i] = -1;
    sl[i] = -1;
    if (p < n) {
      const int j = p / topk, k = p - j * topk;
      const int h = idx[(long long)(t0 + j) * sit + k];
      if (h >= 0 && (long long)h < R) {
        const uint32_t key = (uint32_t)(h + 1), bit = 1u << j;
        uint32_t s = ((uint32_t)h * 2654435761u) >> (32 - UN_LOG);
        while (true) {
          uint32_t cur = *reinterpret_cast<volatile uint32_t*>(&tbl[s]);
          if (cur == 0u) {
            cur = atomicCAS(&tbl[s], 0u, (key << UN_GB) | bit);
            if (cur == 0u) break;
          }
          if ((cur >> UN_GB) == key) {
            atomicOr(&tbl[s], bit);
            break;
          }
          s = (s + 1) & (UN_TBL - 1);
        }
        ids[i] = h;
        sl[i] = (int)s;
      }
    }
  }
  __syncthreads();
  for (int s = tid; s < UN_TBL; s += UN_NT) {
    const uint32_t w = tbl[s];
    if (w) atomicAdd(&bcur[(UN_NB - 1) - (int)(w & (UN_NB - 1))], 1);   // full-mask rows first
  }
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    for (int b = 0; b < UN_NB; ++b) {
      const int c = bcur[b];
      bcur[b] = run;
      run += c;
    }
    *tot = run;
    cnt[r] = run;
  }
  __syncthreads();
  const long long ub = (long long)r * nmax;
  const int nround = (n + UN_NT - 1) / UN_NT;
#pragma unroll
  for (int i = 0; i < UN_PPT; ++i) {
    if (i < nround) {   // block-uniform
      for (int e = lane; e < UN_NB; e += 32) wcnt[warp * UN_NB + e] = 0;
      __syncwarp();
      int b = -1;
      uint32_t m = 0;
      if (sl[i] >= 0) {
        const int j = (tid + i * UN_NT) / topk;
        m = tbl[sl[i]] & (UN_NB - 1);
        if ((m & ((1u << j) - 1u)) == 0u) b = (UN_NB - 1) - (int)m;   // first occurrence = lowest selecting token
      }
      const unsigned peers = __match_any_sync(0xffffffffu, b);
      const int rank = __popc(peers & ((1u << lane) - 1u));
      if (b >= 0 && rank == 0) wcnt[warp * UN_NB + b] = __popc(peers);
      __syncthreads();
      if (tid < UN_NB) {
        int run = bcur[tid];
        for (int w = 0; w < 32; ++w) {
          const int c = wcnt[w * UN_NB + tid];
          wcnt[w * UN_NB + tid] = run;
          run += c;
        }
        bcur[tid] = run;
      }
      __syncthreads();
      if (b >= 0) {
        const int d = wcnt[warp * UN_NB + b] + rank;
        u[ub + d] = ids[i];
        mk[ub + d] = (uint8_t)m;
      }
      __syncwarp();
    }
  }
  for (int d = *tot + tid; d < nmax; d += UN_NT) u[ub + d] = -1;
}

constexpr int PK_M = 64, PK_NW = 8, PK_RS = 8, PK_MD = PK_RS - 1;   // metadata ring: tiles in flight / ahead
constexpr int QSL = DN + 8;   // Q latent row stride (halves): 1040 B -> conflict-free ldmatrix; rope dims live in registers
template <int BN, int ST>
constexpr int pk_smem() {
  return PK_M * QSL * 2 + ST * BN * KS + PK_M * (BN + 8) * 2 + PK_NW * (BN / 8) * 32 * 16 + PK_M * 4 + PK_RS * BN * 9 +
         ST * BN;
}

template <int HG>
__device__ __forceinline__ constexpr uint32_t pk_tokbits(int mb) {   // tokens of query rows [16 mb, 16 mb + 16)
  return ((2u << ((mb * 16 + 15) / HG)) - 1u) & ~((1u << ((mb * 16) / HG)) - 1u);
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void bar_pair(int id) { asm volatile("bar.sync %0, 64;\n" ::"r"(id)); }

// Warp roles. Q.K: warp w -> m-block mb = w / 2 (16 query rows), k-half kh = w % 2 (latent k16 steps [16 kh, +16) and
// rope steps 32 + 2 kh, +1), ALL n8 blocks of the tile: every Q fragment is read from shared memory once per tile
// (the rope fragments stay in registers). The pair (mb, 0/1) swaps partial scores through shared memory (named
// barrier), both then hold the full 16 x BN scores of mb: the row max is pair-local; warp kh writes P for the n8
// blocks nb % 2 == kh and the pair's rescale factors go to shared memory for P.V. P.V: warp w -> latent columns
// [64 w, +64) of all 64 rows.
template <int BN, int ST, int HG>
__global__ void __launch_bounds__(PK_NW * 32, 1)
pk_attn_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host, const uint8_t* __restrict__ cache,
               const uint8_t* __restrict__ spill, const int* __restrict__ u, const uint8_t* __restrict__ mk,
               const int* __restrict__ res, const int* __restrict__ cnt, const int* __restrict__ cu,
               float* __restrict__ m_out, float* __restrict__ l_out, float* __restrict__ acc_out, long long R,
               long long stride_q_t, long long stride_q_h, int T, int ql, int nmax, float sm_scale,
               __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale, int* __restrict__ sem) {
  constexpr int M = PK_M, NW = PK_NW, NT = NW * 32, MB = M / 16;
  constexpr int NB = BN / 8;               // n8 column blocks of a tile
  constexpr int CBW = DN / NW / 16;        // P.V: 16-column latent blocks per warp
  constexpr int PSB = BN + 8;
  constexpr int LPT = NT / BN;
  static_assert(BN % 16 == 0 && BN <= 32 && MB * 2 == NW, "tile");
  extern __shared__ __align__(16) uint8_t smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);                        // [M][QSL] latent dims, permuted k order
  uint8_t* Ks = smem + M * QSL * 2;                                    // [ST][BN][KS]
  __half* Ps = reinterpret_cast<__half*>(Ks + ST * BN * KS);           // [M][PSB]
  float4* xs = reinterpret_cast<float4*>(Ps + M * PSB);                // [NW][NB][32] partial scores
  float* res_ = reinterpret_cast<float*>(xs + NW * NB * 32);           // [M] rescale factors (end: final max)
  int* ring_u = reinterpret_cast<int*>(res_ + M);                      // [PK_RS][BN] u / res / mk of tiles ahead
  int* ring_r = ring_u + PK_RS * BN;
  uint8_t* ring_m = reinterpret_cast<uint8_t*>(ring_r + PK_RS * BN);   // [PK_RS][BN]
  uint8_t* msk = ring_m + PK_RS * BN;                                  // [ST][BN] token bits (0: invalid row)

  const int hb = blockIdx.x, sp = blockIdx.y, r = blockIdx.z, ksplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int mb = warp >> 1, kh = warp & 1;
  const int t0 = cu ? cu[r] : min(r * ql, T);
  const int t1 = cu ? cu[r + 1] : min(t0 + ql, T);
  const int G = min(t1 - t0, M / HG);
  if (G <= 0) return;
  const int n = cnt[r];
  const int per = (n + ksplit - 1) / ksplit, chunk = (per + BN - 1) / BN * BN;
  const int lo = sp * chunk, hi = min(lo + chunk, n);
  const int ntiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  const long long ub = (long long)r * nmax;

  // ---- loader: thread -> tile row tid / LPT, 16 B chunks tid % LPT + LPT k. The rows' (u, res, mk) entries travel
  //      by cp.async PK_MD tiles ahead of the row data (same commit groups) into a shared ring, so the row gather
  //      never waits on a global-load round trip for its addresses.
  const int lrow = tid / LPT, lsub = tid % LPT;
  auto meta = [&](int tile) {   // async copy of tile's metadata into ring slot tile % PK_RS
    if (tile < ntiles) {
      const int i0 = lo + tile * BN, sl = (tile % PK_RS) * BN;
      if (tid < BN) {
        cp_async4(smem_u32(ring_u + sl + tid), u + ub + i0 + tid, i0 + tid < hi);
      } else if (tid < 2 * BN) {
        cp_async4(smem_u32(ring_r + sl + tid - BN), (res ? res : u) + ub + i0 + tid - BN, i0 + tid - BN < hi);
      } else if (tid < 2 * BN + BN / 16) {   // 16 masks per 16 B (i0 and nmax are multiples of 16)
        const int c = tid - 2 * BN;
        cp_async16ca(smem_u32(ring_m + sl + c * 16), mk + ub + i0 + c * 16, i0 + c * 16 < hi);
      }
    }
  };
  auto issue = [&](int tile) {
    if (tile < ntiles) {
      const int sl = (tile % PK_RS) * BN + lrow;
      const long long h = ring_u[sl];
      const int nr = res ? ring_r[sl] : -1;
      const bool v = lo + tile * BN + lrow < hi && h >= 0 && h < R;
      const uint8_t* src =
          !v ? cache : (nr >= 0 ? cache + (long long)nr * D : (nr <= -2 ? spill + (long long)(-nr - 2) * D : host + h * D));
      uint8_t* dst = Ks + (tile % ST) * BN * KS + lrow * KS;
#pragma unroll
      for (int k = 0; k < (D / 16 + LPT - 1) / LPT; ++k) {
        const int c = lsub + LPT * k;
        if (c < D / 16) cp_async16(smem_u32(dst + c * 16), src + c * 16, v);
      }
      if (lsub == 0) msk[(tile % ST) * BN + lrow] = v ? ring_m[sl] : 0;
    }
    meta(tile + PK_MD);
    cp_commit();
  };
  for (int t = 0; t < PK_MD; ++t) meta(t);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    issue(s);
    __syncthreads();   // issue(s + 1)'s metadata copy reuses the ring slot issue(s) just read
  }

  // ---- Q (bf16 global) -> FP16, dims 4j..4j+3 of each 16-group at k positions 2j, 2j+1, 2j+8, 2j+9 (the K fragment
  //      order). Latent dims -> shared; this warp's rope k16 steps (32 + 2 kh + i) -> A fragments in registers.
  //      Row m -> token m / HG, head hb * HG + m % HG; dead tokens -> 0.
  auto qrow = [&](int m) -> const __nv_bfloat16* {
    const int j = m / HG;
    return j < G ? q + (long long)(t0 + j) * stride_q_t + (hb * HG + m % HG) * stride_q_h : nullptr;
  };
  {   // 64 rows x 64 chunks of 8 dims: 16 chunks per thread, loads first (one round trip), then convert + store
    constexpr int QC = M * (DN / 8) / NT;
    uint4 v[QC];
#pragma unroll
    for (int k = 0; k < QC; ++k) {
      const int e = tid + k * NT, m = e / (DN / 8), c8 = e % (DN / 8);
      const __nv_bfloat16* qr = qrow(m);
      v[k] = qr ? *reinterpret_cast<const uint4*>(qr + c8 * 8) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int k = 0; k < QC; ++k) {
      const int e = tid + k * NT, m = e / (DN / 8), c8 = e % (DN / 8);
      // dims 8 c8 .. +7 = 16-group c8 / 2, quads j = 2 (c8 & 1), +1 -> positions 2j, 2j+1 (dst[j]) and 2j+8, 2j+9 (dst[4+j])
      uint32_t* dst = reinterpret_cast<uint32_t*>(Qs + m * QSL + (c8 >> 1) * 16);
      const int j = 2 * (c8 & 1);
      const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * h]));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * h + 1]));
        dst[j + h] = pack_h2(a.x, a.y);
        dst[4 + j + h] = pack_h2(b.x, b.y);
      }
    }
  }
  uint32_t qa_r[2][4];   // A fragment (rows g / g+8 of mb, k positions 2tq.. / 2tq+8..) of rope steps 32 + 2 kh + i
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const __nv_bfloat16* qr = qrow(mb * 16 + hf * 8 + g);
      uint32_t lo2 = 0u, hi2 = 0u;
      if (qr) {
        const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(qr + DN + (2 * kh + i) * 16 + 4 * tq);
        const float2 a = __bfloat1622float2(src[0]), b = __bfloat1622float2(src[1]);
        lo2 = pack_h2(a.x, a.y);
        hi2 = pack_h2(b.x, b.y);
      }
      qa_r[i][hf] = lo2;       // a0 / a1: k positions 2tq, 2tq+1 = dims 4tq, 4tq+1
      qa_r[i][2 + hf] = hi2;   // a2 / a3: k positions 2tq+8, 2tq+9 = dims 4tq+2, 4tq+3
    }

  float acc[MB][2 * CBW][4];
#pragma unroll
  for (int a = 0; a < MB; ++a)
#pragma unroll
    for (int b = 0; b < 2 * CBW; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  float mrow[2] = {-1.0e30f, -1.0e30f};   // running max of rows 16 mb + g, + 8 (pair-local)
  float lsum[2] = {0.f, 0.f};             // partial sums over this warp's P columns
  const uint32_t tokb = pk_tokbits<HG>(mb);
  const int j0 = (mb * 16 + g) / HG, j1 = (mb * 16 + 8 + g) / HG;

  const uint32_t qs_base = smem_u32(Qs), ps_base = smem_u32(Ps);
  for (int it = 0; it < ntiles; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();   // tile it landed; P.V(it - 1) done: P, res_, xs and stage (it - 1) % ST are free
    const int stg = it % ST;
    const uint8_t* mt = msk + stg * BN;
    const uint32_t kt_base = smem_u32(Ks + stg * BN * KS);
    const uint32_t lbits = lane < BN ? (uint32_t)mt[lane] : 0u;

    // ---- partial S (this k-half) for every n8 block selected by some token of mb
    float s[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int c = 0; c < 4; ++c) s[nb][c] = 0.f;
    if (__reduce_or_sync(0xffffffffu, lbits) & tokb) {   // some row of the tile selected by a token of mb
      const uint32_t kaddr = kt_base + (lane & 7) * KS + (lane >> 3) * 16 + kh * 256;
      const uint32_t qaddr = qs_base + ((mb * 16 + (lane & 15)) * QSL + (lane >> 4) * 8) * 2 + kh * 512;
#pragma unroll
      for (int kg = 0; kg < 4; ++kg) {   // 4 k16 steps: K bytes of NB n8 blocks + 4 Q fragments loaded up front
        uint32_t kb[NB][4], qf[4][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
          ldsm_x4(kaddr + nb * 8 * KS + kg * 64, kb[nb][0], kb[nb][1], kb[nb][2], kb[nb][3]);
#pragma unroll
        for (int i = 0; i < 4; ++i) ldsm_x4(qaddr + (kg * 4 + i) * 32, qf[i][0], qf[i][1], qf[i][2], qf[i][3]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int nb = 0; nb < NB; ++nb)
            mma16816(s[nb], qf[i][0], qf[i][1], qf[i][2], qf[i][3], e4m3x2_lo(kb[nb][i]), e4m3x2_hi(kb[nb][i]));
      }
      const uint32_t raddr = kt_base + (lane & 7) * KS + DN + kh * 32 + ((lane >> 3) & 1) * 16;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        uint32_t k0, k1;
        ldsm_x2(raddr + nb * 8 * KS, k0, k1);
        mma16816(s[nb], qa_r[0][0], qa_r[0][1], qa_r[0][2], qa_r[0][3], e4m3x2_lo(k0), e4m3x2_hi(k0));
        mma16816(s[nb], qa_r[1][0], qa_r[1][1], qa_r[1][2], qa_r[1][3], e4m3x2_lo(k1), e4m3x2_hi(k1));
      }
    }
    issue(it + ST - 1);   // next tile's gather: its address chain overlaps the Q.K MMAs in flight
    // ---- swap partials within the pair -> full scores of mb for all BN columns in both warps
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) xs[(warp * NB + nb) * 32 + lane] = make_float4(s[nb][0], s[nb][1], s[nb][2], s[nb][3]);
    bar_pair(1 + mb);
    float mx0 = -1.0e30f, mx1 = -1.0e30f;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const float4 o = xs[((warp ^ 1) * NB + nb) * 32 + lane];
      const uint32_t c0 = mt[nb * 8 + 2 * tq], c1 = mt[nb * 8 + 2 * tq + 1];
      s[nb][0] = ((c0 >> j0) & 1u) ? (s[nb][0] + o.x) * sm_scale : -1.0e30f;
      s[nb][1] = ((c1 >> j0) & 1u) ? (s[nb][1] + o.y) * sm_scale : -1.0e30f;
      s[nb][2] = ((c0 >> j1) & 1u) ? (s[nb][2] + o.z) * sm_scale : -1.0e30f;
      s[nb][3] = ((c1 >> j1) & 1u) ? (s[nb][3] + o.w) * sm_scale : -1.0e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    // lazy max: the reference max moves only when a score exceeds it by > 8 (log2 units), so p <= 256 (FP16-safe)
    // and the accumulator rescale (all 128 accumulators of every P.V warp) is rare; m / l stay consistent
    const float mn0 = mx0 > mrow[0] + 8.f ? mx0 : mrow[0], mn1 = mx1 > mrow[1] + 8.f ? mx1 : mrow[1];
    const float re0 = exp2f(mrow[0] - mn0), re1 = exp2f(mrow[1] - mn1);
    mrow[0] = mn0;
    mrow[1] = mn1;
    if (kh == 0 && tq == 0) {
      res_[mb * 16 + g] = re0;
      res_[mb * 16 + 8 + g] = re1;
    }
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {   // this warp's n8 blocks nb % 2 == kh; masked entries are -1e30: exp2 -> 0
      if ((nb & 1) != kh) continue;     // (compile-time nb: s[] stays in registers)
      const float p0 = exp2f(s[nb][0] - mn0), p1 = exp2f(s[nb][1] - mn0);
      const float p2 = exp2f(s[nb][2] - mn1), p3 = exp2f(s[nb][3] - mn1);
      ps0 += p0 + p1;
      ps1 += p2 + p3;
      *reinterpret_cast<uint32_t*>(Ps + (mb * 16 + g) * PSB + nb * 8 + 2 * tq) = pack_h2(p0, p1);
      *reinterpret_cast<uint32_t*>(Ps + (mb * 16 + 8 + g) * PSB + nb * 8 + 2 * tq) = pack_h2(p2, p3);
    }
    lsum[0] = lsum[0] * re0 + ps0;
    lsum[1] = lsum[1] * re1 + ps1;
    __syncthreads();   // P and rescale factors of all 64 rows ready
    // ---- O = O * re + P V over latent columns [16 CBW w, +16 CBW); k16 blocks without a selected row skipped per mb
#pragma unroll
    for (int b = 0; b < MB; ++b) {
      const float r0 = res_[b * 16 + g], r1 = res_[b * 16 + 8 + g];
      if (__any_sync(0xffffffffu, r0 != 1.f || r1 != 1.f)) {
#pragma unroll
        for (int c = 0; c < 2 * CBW; ++c) {
          acc[b][c][0] *= r0;
          acc[b][c][1] *= r0;
          acc[b][c][2] *= r1;
          acc[b][c][3] *= r1;
        }
      }
    }
#pragma unroll
    for (int kc = 0; kc < BN / 16; ++kc) {
      const uint32_t or16 = __reduce_or_sync(0xffffffffu, (lane >> 4) == kc ? lbits : 0u);
      bool lp[MB];
      bool anyp = false;
#pragma unroll
      for (int b = 0; b < MB; ++b) {
        lp[b] = (or16 & pk_tokbits<HG>(b)) != 0u;
        anyp |= lp[b];
      }
      if (!anyp) continue;
      uint32_t pa[MB][4];
#pragma unroll
      for (int b = 0; b < MB; ++b)
        if (lp[b])
          ldsm_x4(ps_base + ((b * 16 + (lane & 15)) * PSB + kc * 16 + (lane >> 4) * 8) * 2, pa[b][0], pa[b][1],
                  pa[b][2], pa[b][3]);
      const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + warp * CBW * 16 +
                             (lane >> 4) * 16;
#pragma unroll
      for (int cb2 = 0; cb2 < CBW / 2; ++cb2) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
          const uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
          const uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
          const uint32_t be0 = e4m3x2_lo(xe), be1 = e4m3x2_hi(xe), bo0 = e4m3x2_lo(xo), bo1 = e4m3x2_hi(xo);
          const int cb = cb2 * 2 + h2;
#pragma unroll
          for (int b = 0; b < MB; ++b) {
            if (lp[b]) {
              mma16816(acc[b][2 * cb], pa[b][0], pa[b][1], pa[b][2], pa[b][3], be0, be1);
              mma16816(acc[b][2 * cb + 1], pa[b][0], pa[b][1], pa[b][2], pa[b][3], bo0, bo1);
            }
          }
        }
      }
    }
  }
  cp_wait<0>();

  // ---- final max / sum of every row -> shared (res_ = max, xs = sums) for the column-owning P.V warps
  float* lsh = reinterpret_cast<float*>(xs);   // [2][M]
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    float x = lsum[hf];
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    lsum[hf] = x;
  }
  __syncthreads();   // last P.V read res_
  if (tq == 0) {
    lsh[kh * M + mb * 16 + g] = lsum[0];
    lsh[kh * M + mb * 16 + 8 + g] = lsum[1];
    if (kh == 0) {
      res_[mb * 16 + g] = mrow[0];
      res_[mb * 16 + 8 + g] = mrow[1];
    }
  }
  __syncthreads();
#pragma unroll
  for (int b = 0; b < MB; ++b)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int row = b * 16 + hf * 8 + g, j = row / HG;
      if (j >= G) continue;
      const float ls = lsh[row] + lsh[M + row];
      const long long t = t0 + j;
      const int hh = hb * HG + row % HG;
      if (ksplit == 1) {   // normalized bf16
        const float inv = kv_scale / fmaxf(ls, 1e-30f);
        __nv_bfloat16* ot = out + t * sot + hh * soh;
#pragma unroll
        for (int cb = 0; cb < CBW; ++cb) {
          const int col = warp * CBW * 16 + cb * 16 + 4 * tq;
          __nv_bfloat162 v2[2] = {
              __floats2bfloat162_rn(acc[b][2 * cb][2 * hf] * inv, acc[b][2 * cb + 1][2 * hf] * inv),
              __floats2bfloat162_rn(acc[b][2 * cb][2 * hf + 1] * inv, acc[b][2 * cb + 1][2 * hf + 1] * inv)};
          *reinterpret_cast<uint2*>(ot + col) = *reinterpret_cast<uint2*>(v2);
        }
      } else {
        const long long ob = (t * ksplit + sp) * H + hh;
        if (warp == 0 && tq == 0) {
          m_out[ob] = res_[row];
          l_out[ob] = ls;
        }
#pragma unroll
        for (int cb = 0; cb < CBW; ++cb) {
          const int col = warp * CBW * 16 + cb * 16 + 4 * tq;
          *reinterpret_cast<float4*>(acc_out + ob * DN + col) =
              make_float4(acc[b][2 * cb][2 * hf], acc[b][2 * cb + 1][2 * hf], acc[b][2 * cb][2 * hf + 1],
                          acc[b][2 * cb + 1][2 * hf + 1]);
        }
      }
    }
  if (ksplit == 1 || out == nullptr) return;
  // ---- split-K fixup: the last of the ksplit CTAs of (r, hb) merges the partials (still in L2) into bf16 out and
  //      re-arms the counter (graph replay safe: every launch leaves sem at 0)
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    int* sm_ = sem + (long long)r * gridDim.x + hb;
    const int last = atomicAdd(sm_, 1) == ksplit - 1;
    if (last) *sm_ = 0;
    s_last = last;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float* wsm = reinterpret_cast<float*>(xs);   // [M][ksplit + 1]: split weights, then kv_scale / L
  if (tid < M) {
    const int row = tid, j = row / HG;
    if (j < G) {
      const long long b0 = ((long long)(t0 + j) * ksplit) * H + hb * HG + row % HG;
      float mx = -1.0e30f;
      for (int k = 0; k < ksplit; ++k) mx = fmaxf(mx, __ldcg(m_out + b0 + (long long)k * H));
      float L = 0.f;
      for (int k = 0; k < ksplit; ++k) {
        const float w = exp2f(__ldcg(m_out + b0 + (long long)k * H) - mx);
        wsm[row * (ksplit + 1) + k] = w;
        L += w * __ldcg(l_out + b0 + (long long)k * H);
      }
      wsm[row * (ksplit + 1) + ksplit] = kv_scale / fmaxf(L, 1e-30f);
    }
  }
  __syncthreads();
  for (int e = tid; e < M * (DN / 4); e += NT) {
    const int row = e / (DN / 4), c4 = e % (DN / 4), j = row / HG;
    if (j >= G) continue;
    const long long t = t0 + j;
    const int hh = hb * HG + row % HG;
    const long long b0 = (t * ksplit) * H + hh;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < ksplit; ++k) {
      const float w = wsm[row * (ksplit + 1) + k];
      const float4 x = __ldcg(reinterpret_cast<const float4*>(acc_out + (b0 + (long long)k * H) * DN) + c4);
      a.x += w * x.x;
      a.y += w * x.y;
      a.z += w * x.z;
      a.w += w * x.w;
    }
    const float inv = wsm[row * (ksplit + 1) + ksplit];
    __nv_bfloat162 v2[2] = {__floats2bfloat162_rn(a.x * inv, a.y * inv), __floats2bfloat162_rn(a.z * inv, a.w * inv)};
    *reinterpret_cast<uint2*>(out + t * sot + hh * soh + c4 * 4) = *reinterpret_cast<uint2*>(v2);
  }
}

template <int BN, int ST, int HG>
int pk_launch(const void* q, const void* host, const void* cache, const void* spill, const void* u, const void* mk,
              const void* res, const void* cnt, const void* cu, void* m, void* l, void* acc, int nreq, int T, int ql,
              long long R, long long sqt, long long sqh, int nmax, int ks, float sm_scale, void* out, long long sot,
              long long soh, float kv_scale, void* sem, void* stream) {
  constexpr int SM = pk_smem<BN, ST>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(pk_attn_kernel<BN, ST, HG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  dim3 grid(H / HG, (unsigned)ks, (unsigned)nreq);
  pk_attn_kernel<BN, ST, HG><<<grid, PK_NW * 32, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)u,
      (const uint8_t*)mk, (const int*)res, (const int*)cnt, (const int*)cu, (float*)m, (float*)l, (float*)acc, R, sqt,
      sqh, T, ql, nmax, sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)sem);
  return (int)cudaGetLastError();
}

template <int BN, int ST>
int pk_launch_hg(int hg, const void* q, const void* host, const void* cache, const void* spill, const void* u,
                 const void* mk, const void* res, const void* cnt, const void* cu, void* m, void* l, void* acc, int nreq,
                 int T, int ql, long long R, long long sqt, long long sqh, int nmax, int ks, float sm_scale, void* out,
                 long long sot, long long soh, float kv_scale, void* sem, void* stream) {
#define PKARGS q, host, cache, spill, u, mk, res, cnt, cu, m, l, acc, nreq, T, ql, R, sqt, sqh, nmax, ks, sm_scale, out, sot, soh, kv_scale, sem, stream
  switch (hg) {
    case 32: return pk_launch<BN, ST, 32>(PKARGS);
    case 16: return pk_launch<BN, ST, 16>(PKARGS);
    case 8: return pk_launch<BN, ST, 8>(PKARGS);
  }
#undef PKARGS
  return -3;
}

}  // namespace

extern "C" int offload_fill_copy_launch(const void* ml, const void* mcnt, const void* cpd, const void* host,
                                        void* cache, void* spill, void* lflag, void* sflag, int ctas, void* stream,
                                        const void* perm, int ns) {
  // ctas: grid size; + 1000 * rpw (rows per warp per iteration: 1, 2, 4; default 2)
  const int rpw = ctas / 1000 ? ctas / 1000 : 2;
  ctas %= 1000;
  if (ctas <= 0) return -1;
#define FCARGS (const long long*)ml, (const int*)mcnt, (const long long*)cpd, (const uint8_t*)host, (uint8_t*)cache, (uint8_t*)spill, (int*)lflag, (int*)sflag, (const long long*)perm, perm ? ns : 0
  static bool init = false;
  if (!init) {   // run with the max-shared-memory carveout: the attention CTAs (78 KB smem) must fit beside these
    for (auto f : {fill_copy_kernel<1>, fill_copy_kernel<2>, fill_copy_kernel<4>}) {
      cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout,
                                           (int)cudaSharedmemCarveoutMaxShared);
      if (e != cudaSuccess) return (int)e;
    }
    init = true;
  }
  switch (rpw) {
    case 1: fill_copy_kernel<1><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    case 2: fill_copy_kernel<2><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    case 4: fill_copy_kernel<4><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    default: return -2;
  }
#undef FCARGS
  return (int)cudaGetLastError();
}

extern "C" int offload_split_launch2(const void* q, const void* host, const void* cache, const void* spill,
                                     const void* idx, const void* res, void* m, void* l, void* acc, long long T,
                                     long long R, long long sqt, long long sqh, long long sit, int topk, int split,
                                     float sm_scale, int cfg, void* stream, void* out, long long sot, long long soh,
                                     float kv_scale, const void* lflag, const void* sflag) {
  // offload_split_launch + overlapped-fill flags (lflag [C], sflag [spill rows] int32; null = no waits). Only the
  // 4-warp configs: the waiting CTAs must leave room on every SM for the copy's CTAs (1 CTA/SM by shared memory).
  if (topk % (split * BN) || (out && split != 1)) return -1;
  if ((lflag || sflag) && cfg != 2 && cfg != 3) return -3;
#define ARGS q, host, cache, spill, idx, res, m, l, acc, T, R, sqt, sqh, sit, topk, split, sm_scale, out, sot, soh, kv_scale, stream, lflag, sflag
  switch (cfg) {
    case 2: return launch<2, 4>(ARGS);
    case 3: return launch<3, 4>(ARGS);
    case 12: return launch<2, 8>(ARGS);
    case 13: return launch<3, 8>(ARGS);
  }
#undef ARGS
  return -2;
}

extern "C" int offload_split_launch(const void* q, const void* host, const void* cache, const void* spill,
                                    const void* idx, const void* res, void* m, void* l, void* acc, long long T,
                                    long long R, long long sqt, long long sqh, long long sit, int topk, int split,
                                    float sm_scale, int cfg, void* stream, void* out, long long sot, long long soh,
                                    float kv_scale) {
  // cfg = stages (2 | 3) + 10 * (8 warps). out != null (split must be 1): normalized bf16 [T, 32, 512] (strides
  // sot, soh) instead of partials m / l / acc
  if (topk % (split * BN) || (out && split != 1)) return -1;
#define ARGS q, host, cache, spill, idx, res, m, l, acc, T, R, sqt, sqh, sit, topk, split, sm_scale, out, sot, soh, kv_scale, stream
  switch (cfg) {
    case 2: return launch<2, 4>(ARGS);
    case 3: return launch<3, 4>(ARGS);
    case 12: return launch<2, 8>(ARGS);
    case 13: return launch<3, 8>(ARGS);
  }
#undef ARGS
  return -2;
}

extern "C" int offload_pk_union_launch(const void* idx, long long sit, int nreq, int T, int ql, const void* cu, int gmax,
                                       long long R, int topk, int nmax, void* u, void* mk, void* cnt, void* stream) {
  // idx [T, topk] int32 (row stride sit); request r = tokens [cu[r], cu[r+1]) (cu == null: [r ql, min(r ql + ql, T)));
  // gmax = max tokens per request. u [nreq * nmax] int32, mk [nreq * nmax] uint8, cnt [nreq] int32
  if (gmax < 1 || gmax > UN_GB || gmax * topk > UN_PPT * UN_NT || gmax * topk > nmax || R >= (1LL << 26)) return -1;
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(pk_union_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, UN_SMEM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  if (nreq <= 0) return 0;
  pk_union_kernel<<<nreq, UN_NT, UN_SMEM, (cudaStream_t)stream>>>((const int*)idx, sit, T, ql, (const int*)cu, R, topk,
                                                                  nmax, (int*)u, (uint8_t*)mk, (int*)cnt);
  return (int)cudaGetLastError();
}

extern "C" int offload_pk_attn_launch(const void* q, const void* host, const void* cache, const void* spill,
                                      const void* u, const void* mk, const void* res, const void* cnt, const void* cu,
                                      void* m, void* l, void* acc, int nreq, int T, int ql, int gmax, long long R,
                                      long long sqt, long long sqh, int nmax, int ks, float sm_scale, int cfg,
                                      void* stream, void* out, long long sot, long long soh, float kv_scale,
                                      void* sem) {
  // cfg = stages (2 | 3) + 10 * (BN 32 instead of 16). Head group HG = 64 / tokens per CTA, from gmax.
  // out != null: normalized bf16 [T, 32, 512] (strides sot, soh): ks == 1 directly, ks > 1 through partials m / l / acc
  // merged in-kernel by the last CTA of each (request, head group) (sem: int32 [nreq * 32 / HG], zero-initialized,
  // left zeroed). out == null: partials m / l [T * ks * 32], acc [T * ks * 32, 512] for _combine_kernel(SPLIT = ks)
  if (ks < 1 || gmax < 1 || gmax > 8 || nmax % 16 || ks > 31) return -1;
  if (ks > 1 && (!m || !l || !acc || (out && !sem))) return -1;
  if (nreq <= 0) return 0;
  const int hg = gmax <= 2 ? 32 : (gmax <= 4 ? 16 : 8);
#define ARGS2 hg, q, host, cache, spill, u, mk, res, cnt, cu, m, l, acc, nreq, T, ql, R, sqt, sqh, nmax, ks, sm_scale, out, sot, soh, kv_scale, sem, stream
  switch (cfg) {
    case 2: return pk_launch_hg<16, 2>(ARGS2);
    case 3: return pk_launch_hg<16, 3>(ARGS2);
    case 12: return pk_launch_hg<32, 2>(ARGS2);
    case 13: return pk_launch_hg<32, 3>(ARGS2);
  }
#undef ARGS2
  return -2;
}

// ---- compact host rows (ARC_OFFLOAD_ROW_FMT): the overlapped / mirror fill copy and the split attention with host rows
// in format fmt (offload_rowfmt.FMT); fmt == 0 runs the plain entry points' kernels.
extern "C" int offload_fill_copy_launch_fmt(const void* ml, const void* mcnt, const void* cpd, const void* host,
                                            void* cache, void* spill, void* lflag, void* sflag, int ctas, void* stream,
                                            const void* perm, int ns, int fmt) {
  if (fmt == 0) return offload_fill_copy_launch(ml, mcnt, cpd, host, cache, spill, lflag, sflag, ctas, stream, perm, ns);
  const int rpw = ctas / 1000 ? ctas / 1000 : 2;
  ctas %= 1000;
  if (ctas <= 0) return -1;
  const HostFmt f = host_fmt(fmt);
  if ((f.M < 1 || f.M > 3) || (f.blk != 16 && f.blk != 32)) return -4;
#define FCARGS (const long long*)ml, (const int*)mcnt, (const long long*)cpd, (const uint8_t*)host, (uint8_t*)cache, (uint8_t*)spill, (int*)lflag, (int*)sflag, (const long long*)perm, perm ? ns : 0, fmt
  static bool init = false;
  if (!init) {
    for (auto fn : {fill_copy_kernel<1, true>, fill_copy_kernel<2, true>, fill_copy_kernel<4, true>}) {
      cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                                           (int)cudaSharedmemCarveoutMaxShared);
      if (e != cudaSuccess) return (int)e;
    }
    init = true;
  }
  switch (rpw) {
    case 1: fill_copy_kernel<1, true><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    case 2: fill_copy_kernel<2, true><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    case 4: fill_copy_kernel<4, true><<<ctas, 128, 0, (cudaStream_t)stream>>>(FCARGS); break;
    default: return -2;
  }
#undef FCARGS
  return (int)cudaGetLastError();
}

// ARC_OFFLOAD_AR_GATE: offload_fill_copy_launch_fmt with a gate word (device int*, may be null); fmt 0 = fp8 rows.
extern "C" int offload_fill_copy_launch_gated(const void* ml, const void* mcnt, const void* cpd, const void* host,
                                              void* cache, void* spill, void* lflag, void* sflag, int ctas, void* stream,
                                              const void* perm, int ns, int fmt, const void* gate) {
  const int rpw = ctas / 1000 ? ctas / 1000 : 2;
  ctas %= 1000;
  if (ctas <= 0) return -1;
  if (fmt) {
    const HostFmt f = host_fmt(fmt);
    if ((f.M < 1 || f.M > 3) || (f.blk != 16 && f.blk != 32)) return -4;
  }
  static bool init = false;
  if (!init) {
    for (auto fn : {fill_copy_kernel<1, true>, fill_copy_kernel<2, true>, fill_copy_kernel<4, true>,
                    fill_copy_kernel<1, false>, fill_copy_kernel<2, false>, fill_copy_kernel<4, false>}) {
      cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                                           (int)cudaSharedmemCarveoutMaxShared);
      if (e != cudaSuccess) return (int)e;
    }
    init = true;
  }
#define FCARGS (const long long*)ml, (const int*)mcnt, (const long long*)cpd, (const uint8_t*)host, (uint8_t*)cache, (uint8_t*)spill, (int*)lflag, (int*)sflag, (const long long*)perm, perm ? ns : 0, fmt, (const int*)gate
  cudaStream_t s = (cudaStream_t)stream;
  if (fmt) {
    switch (rpw) {
      case 1: fill_copy_kernel<1, true><<<ctas, 128, 0, s>>>(FCARGS); break;
      case 2: fill_copy_kernel<2, true><<<ctas, 128, 0, s>>>(FCARGS); break;
      case 4: fill_copy_kernel<4, true><<<ctas, 128, 0, s>>>(FCARGS); break;
      default: return -2;
    }
  } else {
    switch (rpw) {
      case 1: fill_copy_kernel<1, false><<<ctas, 128, 0, s>>>(FCARGS); break;
      case 2: fill_copy_kernel<2, false><<<ctas, 128, 0, s>>>(FCARGS); break;
      case 4: fill_copy_kernel<4, false><<<ctas, 128, 0, s>>>(FCARGS); break;
      default: return -2;
    }
  }
#undef FCARGS
  return (int)cudaGetLastError();
}

extern "C" int offload_split_launch_fmt(const void* q, const void* host, const void* cache, const void* spill,
                                        const void* idx, const void* res, void* m, void* l, void* acc, long long T,
                                        long long R, long long sqt, long long sqh, long long sit, int topk, int split,
                                        float sm_scale, int cfg, void* stream, void* out, long long sot, long long soh,
                                        float kv_scale, const void* lflag, const void* sflag, int fmt) {
  if (fmt == 0)
    return offload_split_launch2(q, host, cache, spill, idx, res, m, l, acc, T, R, sqt, sqh, sit, topk, split, sm_scale,
                                 cfg, stream, out, sot, soh, kv_scale, lflag, sflag);
  if ((fmt >> 16) & 1) return -5;   // ARC_OFFLOAD_CCACHE rows: only the v2 kernel 5 (offload_attn2.cu) reads them
  if (topk % (split * BN) || (out && split != 1)) return -1;
  if ((lflag || sflag) && cfg != 2 && cfg != 3) return -3;
  const HostFmt f = host_fmt(fmt);
  if ((f.M < 1 || f.M > 3) || (f.blk != 16 && f.blk != 32)) return -4;
#define ARGS q, host, cache, spill, idx, res, m, l, acc, T, R, sqt, sqh, sit, topk, split, sm_scale, out, sot, soh, kv_scale, stream, lflag, sflag, fmt
  switch (cfg) {
    case 2: return launch<2, 4, true>(ARGS);
    case 3: return launch<3, 4, true>(ARGS);
    case 12: return launch<2, 8, true>(ARGS);
    case 13: return launch<3, 8, true>(ARGS);
  }
#undef ARGS
  return -2;
}

// test hook: dequantize n compact host rows (row ids rows[i]) into fp8 rows out[i] (one warp per row)
__global__ void hf_rows_kernel(const uint8_t* host, const long long* rows, int n, uint8_t* out, int fmt) {
  const HostFmt f = host_fmt(fmt);
  const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (w >= n) return;
  const uint8_t* row = host + rows[w] * f.rb;
  int4 a, b;
  if (w & 1) {   // odd rows: the warp form (fill copy), even rows: per-thread chunks (attention loaders)
    int4 raw = make_int4(0, 0, 0, 0);
    if (lane * 16 < f.rb) raw = *reinterpret_cast<const int4*>(row + lane * 16);
    hf_warp(raw, lane, f, a, b);
  } else {
    a = hf_chunk(row, lane, f);
    b = hf_chunk(row, 32 + (lane & 3), f);
  }
  reinterpret_cast<int4*>(out + (long long)w * D)[lane] = a;
  if (lane < 4) reinterpret_cast<int4*>(out + (long long)w * D)[32 + lane] = b;
}
extern "C" int offload_hf_rows_launch(const void* host, const void* rows, int n, void* out, int fmt, void* stream) {
  if (n <= 0) return 0;
  hf_rows_kernel<<<(n + 3) / 4, 128, 0, (cudaStream_t)stream>>>((const uint8_t*)host, (const long long*)rows, n,
                                                               (uint8_t*)out, fmt);
  return (int)cudaGetLastError();
}
