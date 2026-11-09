// Offload decode attention v2 (ARC_OFFLOAD_ATTN2, default off) + L4 microbenchmarks.
// Separate from offload_attn.cu (the shipping kernel) so builds of either cannot break the other.
//
// v2 decode kernel: one CTA = one token (split 1, writes the normalized bf16 output), 8 warps, 32-row tiles of FP8
// rows (576 B: 512 latent + 64 rope) gathered with cp.async from the resolved location (cache line / spill / host).
//   Q.K:  FP16 mma.m16n8k16, Q fragments held in REGISTERS (warp (kq, nh): k-quarter kq of the 576 dims, tile rows
//         16 nh..16 nh + 15; no per-tile Q reads from shared memory), K bytes -> f16 via cvt (each K byte converted
//         once per CTA). Partial scores of the 4 k-quarters are summed through shared memory.
//   softmax: thread -> (head, 4 rows); online max / sum in FP32 registers; P quantized to e4m3 (x 256) or f16.
//   P.V:  PVF8: FP8 mma.m16n8k32 (e4m3 P x e4m3 V straight from the row bytes, ldmatrix.trans + prmt, no cvt);
//         else FP16 mma.m16n8k16 (P f16, V via cvt) = the v1 numerics. Warp w -> latent columns [64 w, +64).
//   Row metadata (idx, res) of the whole token is staged in shared memory by cp.async at the start (META).
// Numerics vs offload_attn.cu: QK identical (FP16 q / K, FP32 accumulation; summation order of the 4 k-quarters
// differs), PVF8 = P rounded to e4m3 (3 mantissa bits), V exact (e4m3 rows are exactly the stored values).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int H = 32, D = 576, DN = 512, BN = 32;
constexpr int QS = D + 8;            // (v1 copy) Q row stride (halves)
constexpr int KS = D + 16;           // K row stride (bytes): 592 B -> conflict-free ldmatrix / ldmatrix.trans
constexpr int PS = BN + 8;           // (v1 copy) P row stride (halves)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
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
__device__ __forceinline__ void mma16816h(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16832f8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                           uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_e4m3x2(float lo, float hi) {   // lo -> low byte
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

// (copied verbatim from offload_attn.cu: compact host rows + overlapped-fill claims)
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
  f.blk = (fmt >> 4) & 255;   // (bit 16: ARC_OFFLOAD_CCACHE compact GPU rows, handled by the callers)
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

// ---- ARC_OFFLOAD_CCACHE (compact GPU rows): cache lines, spill rows and host rows all hold the compact e2m2b32 row
// (400 B: lo nibble plane 256 | sign plane 64 | int8 block exponents 16 | rope e4m3 64). The v5 loader gathers the
// compact bytes with cp.async and decodes them in shared memory into the 576 B e4m3 tile row the MMA path reads, with
// a 16-entry per-block code -> e4m3 byte table (prmt lookups). Same bytes as hf_dq16 (offload_rowfmt numerics), so the
// attention is bit-identical to the fp8 GPU row path.
constexpr int CC_RB = 400;   // e2m2b32 row bytes
// e4m3 bytes of magnitude codes 0..15 (E2M2, value (e2 ? 4 + m : m) * 2^(max(e2,1) - 3 + k)) for block exponent k
__device__ __forceinline__ void cc_table(int k, uint32_t* T) {
  if (k >= -4 && k <= 6) {   // every entry a normal e4m3 value (or 0): exponent field linear in k
    const uint32_t add = (uint32_t)(k + 4) * 0x08080808u;
    T[0] = (0x14100800u + add) & 0xFFFFFF00u;
    T[1] = 0x1E1C1A18u + add;
    T[2] = 0x26242220u + add;
    T[3] = 0x2E2C2A28u + add;
  } else {                   // subnormal / saturating blocks: exactly as hf_dq16
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float v[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) v[b] = hf_val((uint32_t)(4 * w + b), 2, k);
      const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[0], v[1]), __NV_SATFINITE, __NV_E4M3);
      const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[2], v[3]), __NV_SATFINITE, __NV_E4M3);
      T[w] = (lo & 0xFFFFu) | (hi << 16);
    }
  }
}
// 32 values of one block: lo = 4 words of magnitude nibbles (value j of a word in nibble j), sb = 32 sign bits,
// k = block exponent -> 8 e4m3 words. Per 4 values: 2 table prmts + 1 merge prmt (nibble bit 3 picks the table half)
// + sign bytes by a carry-free multiply (bit j -> bit 8 j + 7).
__device__ __forceinline__ void cc_block(const uint4 lo, uint32_t sb, int k, uint32_t* o) {
  uint32_t T[4];
  cc_table(k, T);
  const uint32_t w[4] = {lo.x, lo.y, lo.z, lo.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t sel = w[q] & 0x77777777u;
    const uint32_t mrg = ((w[q] >> 1) & 0x44444444u) | 0x32103210u;
    const uint32_t s8 = sb >> (8 * q);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t sl = h ? sel >> 16 : sel, mg = h ? mrg >> 16 : mrg;
      const uint32_t a = prmt(T[0], T[1], sl), b = prmt(T[2], T[3], sl);
      const uint32_t sg = (((h ? s8 >> 4 : s8) & 15u) * 0x10204080u) & 0x80808080u;
      o[2 * q + h] = prmt(a, b, mg) | sg;
    }
  }
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
// overlapped-fill claim with compact GPU rows: the G loader threads of one row copy the raw compact row host -> dst
template <int G>
__device__ __forceinline__ void fetch_or_wait_raw(int* f, const uint8_t* src, uint8_t* dst, int sub) {
  const int lane = threadIdx.x & 31;
  const unsigned gmask = ((1u << G) - 1) << (lane & ~(G - 1));
  int st = 0;
  if (sub == 0) st = atomicCAS(f, 1, 2);
  st = __shfl_sync(gmask, st, lane & ~(G - 1));
  if (st == 1) {
    for (int c = sub; c < CC_RB / 16; c += G) reinterpret_cast<int4*>(dst)[c] = reinterpret_cast<const int4*>(src)[c];
    __threadfence();
    __syncwarp(gmask);
    if (sub == 0) atomicExch(f, 0);
  }
  wait_flag(f);
}

// =====================================================================================================================
// L4 microbenchmarks: tensor-core rates (f16 -> f32 / f16 accumulate, e4m3 -> f32), cvt e4m3x2 -> f16x2 rate,
// and the SM clock under load (clock64 vs globaltimer of CTA 0).
__global__ void __launch_bounds__(128) mb_kernel(int kind, int iters, float* out, long long* clk) {
  long long c0 = clock64(), g0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  const uint32_t x = threadIdx.x * 0x01010101u + 0x3c3c3c3cu;
  float acc[8][4];
  uint32_t hacc[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    hacc[i][0] = hacc[i][1] = 0u;
  }
  uint32_t a0 = x, a1 = x ^ 1u, a2 = x ^ 2u, a3 = x ^ 3u, b0 = x ^ 5u, b1 = x ^ 7u;
  if (kind == 0) {
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int i = 0; i < 8; ++i) mma16816(acc[i], a0, a1, a2, a3, b0, b1);
  } else if (kind == 1) {
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int i = 0; i < 8; ++i) mma16816h(hacc[i], a0, a1, a2, a3, b0, b1);
  } else if (kind == 2) {
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int i = 0; i < 8; ++i) mma16832f8(acc[i], a0, a1, a2, a3, b0, b1);
  } else if (kind == 3 || kind == 5) {   // cvt throughput: 8 cvts of a loop-varying 32-bit word (kind 5: no cvt)
    uint32_t v = x, r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = 0u;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] ^= kind == 3 ? (i & 1 ? e4m3x2_hi(v) : e4m3x2_lo(v + i)) : (v + i);
      v += 0x00010003u;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i][0] = (float)r[i];
  } else if (kind == 4) {   // mixed: 1 cvt pair per f16 mma (the v1 K-fragment ratio)
    uint32_t v = x;
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint32_t c0_ = e4m3x2_lo(v + i), c1_ = e4m3x2_hi(v + i);
        mma16816(acc[i], a0, a1, a2, a3, c0_, c1_);
        v += 0x01000001u;
      }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3] + __int_as_float(hacc[i][0] ^ hacc[i][1]);
  if (s == 1.2345f) out[threadIdx.x] = s;
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    long long g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    clk[0] = clock64() - c0;
    clk[1] = g1 - g0;
  }
}

// =====================================================================================================================
// Ablation copy of offload_attn.cu's offload_split_kernel (cache-resident rows only, split 1 -> bf16 out).
// ABL bits: 1 no row loads after the prologue (stale tiles), 2 no idx/res global loads (fixed rows), 4 no fp8->f16
// cvt (raw bytes into the MMA), 8 no Q.K MMA, 16 no P.V MMA, 32 metadata staged in smem at the start (cp.async).
template <int ST, int NW, int ABL>
constexpr int abl_smem() { return H * QS * 2 + ST * BN * KS + H * PS * 2 + 2 * 4 * H * 4 + ST * BN + ((ABL & 32) ? 2 * 2048 * 4 : 0); }

template <int ST, int NW, int ABL>
__global__ void __launch_bounds__(NW * 32, 1)
abl_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache, const int* __restrict__ idx,
           const int* __restrict__ res, long long R, long long stride_q_t, long long stride_q_h, long long stride_i_t,
           int topk, float sm_scale, __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale) {
  constexpr int NT = NW * 32;
  constexpr int MPW = NW == 4 ? 2 : 1;
  constexpr int CBW = DN / NW / 16;
  constexpr int LPT = NT / BN;
  extern __shared__ __align__(16) uint8_t smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);
  uint8_t* Ks = smem + H * QS * 2;
  __half* Ps = reinterpret_cast<__half*>(Ks + ST * BN * KS);
  float* red = reinterpret_cast<float*>(Ps + H * PS);
  float* redl = red + 4 * H;
  uint8_t* vflag = reinterpret_cast<uint8_t*>(redl + 4 * H);
  int* mi = reinterpret_cast<int*>(vflag + ((ST * BN + 15) & ~15));
  int* mr = mi + 2048;

  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int nbw = warp & 3, mb0 = NW == 4 ? 0 : warp >> 2;
  const int ntiles = topk / BN;
  const int* idx_t = idx + (long long)t * stride_i_t;
  const int* res_t = res + (long long)t * topk;
  if (ABL & 32) {
    for (int c = tid; c < topk / 4; c += NT) {
      cp_async16(smem_u32(mi + 4 * c), idx_t + 4 * c, true);
      cp_async16(smem_u32(mr + 4 * c), res_t + 4 * c, true);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }
  const int lrow = tid / LPT, lsub = tid % LPT;
  auto issue = [&](int tile) {
    if (tile < ntiles) {
      bool v = true;
      const uint8_t* src;
      if (ABL & 2) {
        src = cache + (long long)(tile * BN + lrow) * D;
      } else {
        int h = (ABL & 32) ? mi[tile * BN + lrow] : idx_t[tile * BN + lrow];
        v = (h >= 0) && ((long long)h < R);
        int r = v ? ((ABL & 32) ? mr[tile * BN + lrow] : res_t[tile * BN + lrow]) : 0;
        src = cache + (long long)r * D;
      }
      uint8_t* dst = Ks + (tile % ST) * BN * KS + lrow * KS;
      if (!(ABL & 1) || tile < ST) {
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
  float mrow[4];
  float lrow_[2 * MPW];
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
    float s[MPW][4];
#pragma unroll
    for (int a = 0; a < MPW; ++a) s[a][0] = s[a][1] = s[a][2] = s[a][3] = 0.f;
    const uint32_t kaddr = kt_base + (nbw * 8 + (lane & 7)) * KS + (lane >> 3) * 16;
    const uint32_t qaddr = qs_base + ((mb0 * 16 + (lane & 15)) * QS + (lane >> 4) * 8) * 2;
    if (!(ABL & 8)) {
#pragma unroll
      for (int kk = 0; kk < D / 16; kk += 4) {
        uint32_t kb[4];
        ldsm_x4(kaddr + kk * 16, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t b0 = (ABL & 4) ? kb[i] : e4m3x2_lo(kb[i]), b1 = (ABL & 4) ? kb[i] ^ 0x1234u : e4m3x2_hi(kb[i]);
#pragma unroll
          for (int mb = 0; mb < MPW; ++mb) {
            uint32_t a0, a1, a2, a3;
            ldsm_x4(qaddr + (mb * 16 * QS + (kk + i) * 16) * 2, a0, a1, a2, a3);
            mma16816(s[mb], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
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
    const uint32_t ps_base = smem_u32(Ps);
    if (!(ABL & 16)) {
#pragma unroll
      for (int kc = 0; kc < BN / 16; ++kc) {
        uint32_t pa[2][4];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb)
          ldsm_x4(ps_base + ((mb * 16 + (lane & 15)) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[mb][0], pa[mb][1],
                  pa[mb][2], pa[mb][3]);
        const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + warp * CBW * 16 +
                               (lane >> 4) * 16;
#pragma unroll
        for (int cb2 = 0; cb2 < CBW / 2; ++cb2) {
          uint32_t r0, r1, r2, r3;
          ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
            uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
            uint32_t be0 = (ABL & 4) ? xe : e4m3x2_lo(xe), be1 = (ABL & 4) ? xe ^ 7u : e4m3x2_hi(xe),
                     bo0 = (ABL & 4) ? xo : e4m3x2_lo(xo), bo1 = (ABL & 4) ? xo ^ 7u : e4m3x2_hi(xo);
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
  }
  cp_wait<0>();
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
}

template <int ST, int NW, int ABL>
int abl_launch(const void* q, const void* cache, const void* idx, const void* res, long long T, long long R,
               long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out, long long sot,
               long long soh, float kv_scale, void* stream) {
  constexpr int SM = abl_smem<ST, NW, ABL>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(abl_kernel<ST, NW, ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  abl_kernel<ST, NW, ABL><<<(unsigned)T, NW * 32, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)cache, (const int*)idx, (const int*)res, R, sqt, sqh, sit, topk,
      sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale);
  return (int)cudaGetLastError();
}

// ---- phase profile (ABL & 64, bench only): thread 0 of every CTA accumulates clock64 deltas per phase
//      0 wait tile + barrier, 1 issue loads, 2 Q.K, 3 barrier, 4 softmax, 5 barrier, 6 P.V; [7] CTAs;
//      [8] / [9]: clock64 / globaltimer ns over CTA 0's lifetime
__device__ unsigned long long g_prof[16];
#define PROF_INIT                                                                                                      \
  long long pacc[7] = {0, 0, 0, 0, 0, 0, 0}, pc = 0, pc0 = 0, pg0 = 0;                                                \
  if ((ABL & 64) && tid == 0) {                                                                                        \
    pc = pc0 = clock64();                                                                                              \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(pg0));                                                           \
  }
#define PROF_T(i)                                                                                                      \
  if ((ABL & 64) && tid == 0) {                                                                                        \
    const long long c_ = clock64();                                                                                    \
    pacc[i] += c_ - pc;                                                                                                \
    pc = c_;                                                                                                           \
  }
#define PROF_END                                                                                                       \
  if ((ABL & 64) && tid == 0) {                                                                                        \
    for (int i_ = 0; i_ < 7; ++i_) atomicAdd(&g_prof[i_], (unsigned long long)pacc[i_]);                               \
    atomicAdd(&g_prof[7], 1ull);                                                                                       \
    if (blockIdx.x == 0) {                                                                                             \
      long long g1_;                                                                                                   \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1_));                                                         \
      g_prof[8] = (unsigned long long)(clock64() - pc0);                                                               \
      g_prof[9] = (unsigned long long)(g1_ - pg0);                                                                     \
    }                                                                                                                  \
  }

// =====================================================================================================================
// v2 decode kernel (see the header). 8 warps; warp w: Q.K role (kq = w & 3, nh = w >> 2), P.V role latent cols [64 w, +64).
constexpr int V2_NW = 8, V2_NT = V2_NW * 32, V2_LPT = V2_NT / BN;   // 8 loader threads per tile row
constexpr int V2_TOPK = 2048;                                        // metadata staged for up to this many rows
constexpr int SPS = BN + 8;                                          // partial-score row stride (floats): 40
constexpr int P8S = 48;                                              // e4m3 P row stride (bytes), conflict-free lds.32
template <int ST, bool PVF8>
constexpr int v2_smem() {
  return ST * BN * KS + 4 * H * SPS * 4 + (PVF8 ? H * P8S : H * PS * 2) + 2 * H * 4 + ((ST * BN + 15) & ~15) +
         2 * V2_TOPK * 4;
}
// k position of tile row r in the permuted k32 order of the FP8 P.V fragments (ldmatrix.trans + prmt):
// B reg 0 of thread tq holds rows {2tq, 2tq+1, 8+2tq, 9+2tq}, reg 1 the same + 16.
__device__ __forceinline__ int p8pos(int r) {
  return (r >> 4) * 16 + ((r & 7) >> 1) * 4 + ((r >> 3) & 1) * 2 + (r & 1);
}

template <int ST, bool PVF8, bool HC, int ABL = 0>   // ABL (bench only): 1 no row loads, 8 no Q.K mma, 16 no P.V mma
__global__ void __launch_bounds__(V2_NT, 1)
v2_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host, const uint8_t* __restrict__ cache,
          const uint8_t* __restrict__ spill, const int* __restrict__ idx, const int* __restrict__ res, long long R,
          long long stride_q_t, long long stride_q_h, long long stride_i_t, int topk, float sm_scale,
          __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale, int* lflag, int* sflag,
          int fmt) {
  constexpr int NT = V2_NT, LPT = V2_LPT;
  const HostFmt hf = host_fmt(fmt);
  const long long HD = HC ? hf.rb : D;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Ks = smem;                                                    // [ST][BN][KS]
  float* Sp = reinterpret_cast<float*>(Ks + ST * BN * KS);               // [4 kq][H][SPS]
  uint8_t* Pp = reinterpret_cast<uint8_t*>(Sp + 4 * H * SPS);            // e4m3 [H][P8S] | f16 [H][PS]
  float* re_s = reinterpret_cast<float*>(Pp + (PVF8 ? H * P8S : H * PS * 2));   // [H] rescale (end: row sums)
  float* l_s = re_s + H;
  uint8_t* vflag = reinterpret_cast<uint8_t*>(l_s + H);                  // [ST][BN]
  int* mi = reinterpret_cast<int*>(vflag + ((ST * BN + 15) & ~15));      // [topk] host row ids
  int* mr = mi + V2_TOPK;                                                // [topk] resolved locations

  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int kq = warp & 3, nh = warp >> 2;
  const int ntiles = topk / BN;
  {   // ---- row metadata of the token -> shared (one round trip for the whole CTA)
    const int* idx_t = idx + (long long)t * stride_i_t;
    const int* res_t = res + (long long)t * topk;
    for (int c = tid; c < topk / 4; c += NT) {
      cp_async16(smem_u32(mi + 4 * c), idx_t + 4 * c, true);
      cp_async16(smem_u32(mr + 4 * c), res_t + 4 * c, true);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }

  const int lrow = tid / LPT, lsub = tid % LPT;
  auto row_src = [&](int tile, bool& valid, bool& hostc) -> const uint8_t* {
    const int h = mi[tile * BN + lrow];
    valid = (h >= 0) && ((long long)h < R);
    hostc = false;
    if (!valid) return cache;
    int r = mr[tile * BN + lrow];
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
    hostc = HC;
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
      } else if (!(ABL & 1) || tile < ST) {
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

  // ---- Q fragments in registers: m-block mb (heads 16 mb + g, +8), k16 steps 9 kq + j. Dims 4tq..4tq+3 of the
  //      16-group sit at k positions 2tq, 2tq+1 (a0/a1) and 2tq+8, 2tq+9 (a2/a3) = the K fragment order of v1.
  uint32_t qf[2][9][4];
  {
    const __nv_bfloat16* qt = q + (long long)t * stride_q_t;
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const __nv_bfloat16* qr = qt + (long long)(16 * mb + 8 * hh + g) * stride_q_h + 4 * tq;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          const uint2 v = *reinterpret_cast<const uint2*>(qr + 16 * (9 * kq + j));
          const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
          const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
          qf[mb][j][hh] = pack_h2(a.x, a.y);
          qf[mb][j][2 + hh] = pack_h2(b.x, b.y);
        }
      }
  }

  constexpr int NB8 = 8;   // P.V: n8 tiles per warp (64 columns = 4 blocks of 16, even / odd tile each)
  float acc[2][NB8][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < NB8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  // softmax role: head sh = tid >> 3, tile rows 4 sc .. 4 sc + 3 (8 consecutive lanes per head)
  const int sh = tid >> 3, sc = tid & 7;
  float m_run = -1.0e30f, l_part = 0.f;
  PROF_INIT
  for (int it = 0; it < ntiles; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    PROF_T(0)
    issue(it + ST - 1);
    PROF_T(1)
    const int stg = it % ST;
    const uint8_t* Kt = Ks + stg * BN * KS;
    const uint32_t kt_base = smem_u32(Kt);

    // ---- partial S over k-quarter kq for tile rows 16 nh .. +15 (2 n8 blocks) x 32 heads
    {
      float s[2][2][4];
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) s[a][b][0] = s[a][b][1] = s[a][b][2] = s[a][b][3] = 0.f;
      const int m4 = lane >> 3;
      if (!(ABL & 8)) {
      const uint32_t kbase = kt_base + (16 * nh + 8 * (m4 >> 1) + (lane & 7)) * KS + 16 * (9 * kq + (m4 & 1));
#pragma unroll
      for (int j = 0; j < 8; j += 2) {   // matrices (nb0, j), (nb0, j+1), (nb1, j), (nb1, j+1)
        uint32_t kb[4];
        ldsm_x4(kbase + 16 * j, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const uint32_t b0 = e4m3x2_lo(kb[m]), b1 = e4m3x2_hi(kb[m]);
          const int nb = m >> 1, jj = j + (m & 1);
#pragma unroll
          for (int mb = 0; mb < 2; ++mb)
            mma16816(s[mb][nb], qf[mb][jj][0], qf[mb][jj][1], qf[mb][jj][2], qf[mb][jj][3], b0, b1);
        }
      }
      {   // k16 step 8: matrices (nb0, 8), (nb1, 8)
        uint32_t kb[2];
        const uint32_t a8 = kt_base + (16 * nh + 8 * ((lane >> 3) & 1) + (lane & 7)) * KS + 16 * (9 * kq + 8);
        ldsm_x2(a8, kb[0], kb[1]);
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
          const uint32_t b0 = e4m3x2_lo(kb[nb]), b1 = e4m3x2_hi(kb[nb]);
#pragma unroll
          for (int mb = 0; mb < 2; ++mb)
            mma16816(s[mb][nb], qf[mb][8][0], qf[mb][8][1], qf[mb][8][2], qf[mb][8][3], b0, b1);
        }
      }
      }
#pragma unroll
      for (int mb = 0; mb < 2; ++mb)
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
          const int row = 16 * nh + 8 * nb + 2 * tq;
          *reinterpret_cast<float2*>(Sp + (kq * H + 16 * mb + g) * SPS + row) = make_float2(s[mb][nb][0], s[mb][nb][1]);
          *reinterpret_cast<float2*>(Sp + (kq * H + 16 * mb + 8 + g) * SPS + row) = make_float2(s[mb][nb][2], s[mb][nb][3]);
        }
    }
    PROF_T(2)
    __syncthreads();
    PROF_T(3)
    // ---- softmax: head sh, rows 4 sc .. 4 sc + 3
    {
      float4 a = *reinterpret_cast<const float4*>(Sp + (0 * H + sh) * SPS + 4 * sc);
#pragma unroll
      for (int k = 1; k < 4; ++k) {
        const float4 b = *reinterpret_cast<const float4*>(Sp + (k * H + sh) * SPS + 4 * sc);
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      }
      const uint32_t vf = *reinterpret_cast<const uint32_t*>(vflag + stg * BN + 4 * sc);
      float sv[4] = {a.x * sm_scale, a.y * sm_scale, a.z * sm_scale, a.w * sm_scale};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!((vf >> (8 * i)) & 0xFF)) sv[i] = -1.0e30f;
      float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
      const float mn = fmaxf(m_run, mx);
      const float re = exp2f(m_run - mn);
      m_run = mn;
      float p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = ((vf >> (8 * i)) & 0xFF) ? exp2f(sv[i] - mn) : 0.f;
      l_part = l_part * re + (p[0] + p[1]) + (p[2] + p[3]);
      if (sc == 0) re_s[sh] = re;
      if (PVF8) {
        const int r0 = 4 * sc;
        uint8_t* pr = Pp + sh * P8S;
        *reinterpret_cast<uint16_t*>(pr + p8pos(r0)) = (uint16_t)pack_e4m3x2(p[0] * 256.f, p[1] * 256.f);
        *reinterpret_cast<uint16_t*>(pr + p8pos(r0 + 2)) = (uint16_t)pack_e4m3x2(p[2] * 256.f, p[3] * 256.f);
      } else {
        __half* pr = reinterpret_cast<__half*>(Pp) + sh * PS + 4 * sc;
        *reinterpret_cast<uint2*>(pr) = make_uint2(pack_h2(p[0], p[1]), pack_h2(p[2], p[3]));
      }
    }
    PROF_T(4)
    __syncthreads();
    PROF_T(5)
    // ---- O = O * re + P V over latent columns [64 w, +64)
    {
      float rs[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) rs[i] = re_s[(i >> 1) * 16 + (i & 1) * 8 + g];
#pragma unroll
      for (int mb = 0; mb < 2; ++mb)
#pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
          acc[mb][nb][0] *= rs[2 * mb];
          acc[mb][nb][1] *= rs[2 * mb];
          acc[mb][nb][2] *= rs[2 * mb + 1];
          acc[mb][nb][3] *= rs[2 * mb + 1];
        }
      if (ABL & 16) {
      } else if (PVF8) {
        uint32_t pa[2][4];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
          const uint8_t* p0 = Pp + (16 * mb + g) * P8S + 4 * tq;
          pa[mb][0] = *reinterpret_cast<const uint32_t*>(p0);
          pa[mb][1] = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S);
          pa[mb][2] = *reinterpret_cast<const uint32_t*>(p0 + 16);
          pa[mb][3] = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S + 16);
        }
        const uint32_t vaddr = kt_base + lane * KS + 64 * warp;   // matrices j = rows 8 j .. 8 j + 7
#pragma unroll
        for (int cb = 0; cb < 4; ++cb) {
          uint32_t r0, r1, r2, r3;
          ldsm_x4_t(vaddr + 16 * cb, r0, r1, r2, r3);
          const uint32_t be0 = prmt(r0, r1, 0x6420), be1 = prmt(r2, r3, 0x6420);
          const uint32_t bo0 = prmt(r0, r1, 0x7531), bo1 = prmt(r2, r3, 0x7531);
#pragma unroll
          for (int mb = 0; mb < 2; ++mb) {
            mma16832f8(acc[mb][2 * cb], pa[mb][0], pa[mb][1], pa[mb][2], pa[mb][3], be0, be1);
            mma16832f8(acc[mb][2 * cb + 1], pa[mb][0], pa[mb][1], pa[mb][2], pa[mb][3], bo0, bo1);
          }
        }
      } else {
        const uint32_t ps_base = smem_u32(Pp);
#pragma unroll
        for (int kc = 0; kc < BN / 16; ++kc) {
          uint32_t pa[2][4];
#pragma unroll
          for (int mb = 0; mb < 2; ++mb)
            ldsm_x4(ps_base + ((mb * 16 + (lane & 15)) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[mb][0], pa[mb][1],
                    pa[mb][2], pa[mb][3]);
          const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + 64 * warp +
                                 (lane >> 4) * 16;
#pragma unroll
          for (int cb2 = 0; cb2 < 2; ++cb2) {
            uint32_t r0, r1, r2, r3;
            ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
              uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
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
    }
    PROF_T(6)
  }
  cp_wait<0>();
  PROF_END
  // ---- row sums: 8 lanes per head
  l_part += __shfl_xor_sync(0xffffffff, l_part, 1);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 2);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 4);
  if (sc == 0) l_s[sh] = l_part;
  __syncthreads();
  float inv[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    inv[i] = (PVF8 ? kv_scale * (1.f / 256.f) : kv_scale) / fmaxf(l_s[(i >> 1) * 16 + (i & 1) * 8 + g], 1e-30f);
  __nv_bfloat16* ot = out + (long long)t * sot;
#pragma unroll
  for (int mb = 0; mb < 2; ++mb)
#pragma unroll
    for (int cb = 0; cb < 4; ++cb) {
      int col = 64 * warp + cb * 16 + 4 * tq;
      float a = inv[2 * mb], b = inv[2 * mb + 1];
      __nv_bfloat162 r0[2] = {__floats2bfloat162_rn(acc[mb][2 * cb][0] * a, acc[mb][2 * cb + 1][0] * a),
                              __floats2bfloat162_rn(acc[mb][2 * cb][1] * a, acc[mb][2 * cb + 1][1] * a)};
      __nv_bfloat162 r1[2] = {__floats2bfloat162_rn(acc[mb][2 * cb][2] * b, acc[mb][2 * cb + 1][2] * b),
                              __floats2bfloat162_rn(acc[mb][2 * cb][3] * b, acc[mb][2 * cb + 1][3] * b)};
      *reinterpret_cast<uint2*>(ot + (mb * 16 + g) * soh + col) = *reinterpret_cast<uint2*>(r0);
      *reinterpret_cast<uint2*>(ot + (mb * 16 + 8 + g) * soh + col) = *reinterpret_cast<uint2*>(r1);
    }
}

template <int ST, bool PVF8, bool HC, int ABL = 0>
int v2_launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx, const void* res,
              long long T, long long R, long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out,
              long long sot, long long soh, float kv_scale, const void* lflag, const void* sflag, int fmt, void* stream) {
  constexpr int SM = v2_smem<ST, PVF8>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(v2_kernel<ST, PVF8, HC, ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  v2_kernel<ST, PVF8, HC, ABL><<<(unsigned)T, V2_NT, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)idx,
      (const int*)res, R, sqt, sqh, sit, topk, sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)lflag,
      (int*)sflag, fmt);
  return (int)cudaGetLastError();
}

// =====================================================================================================================
// v3 decode kernel: head-split CTAs so that TWO CTAs co-reside per L4 SM (their Q.K / softmax / P.V phases overlap,
// which one CTA with CTA-wide barriers cannot do). CTA = (token t, 16-head group hg), 4 warps, <= 255 registers,
// ~44 KB shared memory (ST 2). Adjacent CTAs are the two head groups of one token (K tiles shared through L2).
//   Q.K:  warp (kh = w & 1, nh = w >> 1): k-half kh (18 k16 steps, Q in registers), tile rows 16 nh .. +15.
//   P.V:  warp w -> latent columns [128 w, +128) of the 16 heads.
//   Row metadata (idx, res) of tile it + ST is prefetched into registers one tile ahead (no shared staging).
constexpr int V3_NT = 128, V3_LPT = V3_NT / BN;   // 4 loader threads per tile row
template <int ST, bool PVF8>
constexpr int v3_smem() {
  return ST * BN * KS + 2 * 16 * SPS * 4 + (PVF8 ? 16 * P8S : 16 * PS * 2) + 2 * 16 * 4 + ((ST * BN + 15) & ~15);
}

template <int ST, bool PVF8, bool HC, int ABL = 0>
__global__ void __launch_bounds__(V3_NT, 2)
v3_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host, const uint8_t* __restrict__ cache,
          const uint8_t* __restrict__ spill, const int* __restrict__ idx, const int* __restrict__ res, long long R,
          long long stride_q_t, long long stride_q_h, long long stride_i_t, int topk, float sm_scale,
          __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale, int* lflag, int* sflag,
          int fmt) {
  constexpr int NT = V3_NT, LPT = V3_LPT;
  const HostFmt hf = host_fmt(fmt);
  const long long HD = HC ? hf.rb : D;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Ks = smem;                                                    // [ST][BN][KS]
  float* Sp = reinterpret_cast<float*>(Ks + ST * BN * KS);               // [2 kh][16][SPS]
  uint8_t* Pp = reinterpret_cast<uint8_t*>(Sp + 2 * 16 * SPS);           // e4m3 [16][P8S] | f16 [16][PS]
  float* re_s = reinterpret_cast<float*>(Pp + (PVF8 ? 16 * P8S : 16 * PS * 2));
  float* l_s = re_s + 16;
  uint8_t* vflag = reinterpret_cast<uint8_t*>(l_s + 16);                 // [ST][BN]

  const int t = blockIdx.x >> 1, hg = blockIdx.x & 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int kh = warp & 1, nh = warp >> 1;
  const int ntiles = topk / BN;
  const int* idx_t = idx + (long long)t * stride_i_t;
  const int* res_t = res + (long long)t * topk;
  const int lrow = tid / LPT, lsub = tid % LPT;

  auto issue = [&](int tile, int h, int r) {
    if (tile < ntiles) {
      bool v = (h >= 0) && ((long long)h < R);
      bool hc = false;
      const uint8_t* src = cache;
      if (v) {
        if (r >= 0) {
          if (r >= FRESH) {
            r -= FRESH;
            if (lflag) fetch_or_wait<LPT, HC>(lflag + r, host + (long long)h * HD, const_cast<uint8_t*>(cache) + (long long)r * D, lsub, hf);
          }
          src = cache + (long long)r * D;
        } else if (r <= -2) {
          if (sflag)
            fetch_or_wait<LPT, HC>(sflag + (-r - 2), host + (long long)h * HD, const_cast<uint8_t*>(spill) + (long long)(-r - 2) * D, lsub, hf);
          src = spill + (long long)(-r - 2) * D;
        } else {
          hc = HC;
          src = host + (long long)h * HD;
        }
      }
      uint8_t* dst = Ks + (tile % ST) * BN * KS + lrow * KS;
      if (HC && hc) {
#pragma unroll
        for (int k = 0; k < D / 16 / LPT; ++k) {
          int c = lsub + LPT * k;
          *reinterpret_cast<int4*>(dst + c * 16) = hf_chunk(src, c, hf);
        }
      } else if (!(ABL & 1) || tile < ST) {
#pragma unroll
        for (int k = 0; k < D / 16 / LPT; ++k) {
          int c = lsub + LPT * k;
          cp_async16(smem_u32(dst + c * 16), src + c * 16, v);
        }
      }
      if (lsub == 0) vflag[(tile % ST) * BN + lrow] = v;
    }
    cp_commit();
  };
  auto meta = [&](int tile, int& h, int& r) {
    h = -1;
    r = 0;
    if (tile < ntiles) {
      h = idx_t[tile * BN + lrow];
      r = res_t[tile * BN + lrow];
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    int h, r;
    meta(s, h, r);
    issue(s, h, r);
  }
  int nxh, nxr;   // metadata of the next tile to issue (it + ST - 1), loaded one iteration ahead
  meta(ST - 1, nxh, nxr);

  // ---- Q fragments: heads 16 hg + g (+8), k16 steps 18 kh + j
  uint32_t qf[18][4];
  {
    const __nv_bfloat16* qt = q + (long long)t * stride_q_t;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const __nv_bfloat16* qr = qt + (long long)(16 * hg + 8 * hh + g) * stride_q_h + 4 * tq;
#pragma unroll
      for (int j = 0; j < 18; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(qr + 16 * (18 * kh + j));
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        qf[j][hh] = pack_h2(a.x, a.y);
        qf[j][2 + hh] = pack_h2(b.x, b.y);
      }
    }
  }

  float acc[16][4];   // 16 n8 tiles: 8 blocks of 16 columns, even / odd
#pragma unroll
  for (int b = 0; b < 16; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const int sh = tid >> 3, sc = tid & 7;   // softmax: head sh (of 16), rows 4 sc .. +3
  float m_run = -1.0e30f, l_part = 0.f;
  PROF_INIT
  for (int it = 0; it < ntiles; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    PROF_T(0)
    issue(it + ST - 1, nxh, nxr);
    meta(it + ST, nxh, nxr);
    PROF_T(1)
    const int stg = it % ST;
    const uint32_t kt_base = smem_u32(Ks + stg * BN * KS);
    {
      float s[2][4];
#pragma unroll
      for (int b = 0; b < 2; ++b) s[b][0] = s[b][1] = s[b][2] = s[b][3] = 0.f;
      if (!(ABL & 8)) {
        const int m4 = lane >> 3;
        const uint32_t kbase = kt_base + (16 * nh + 8 * (m4 >> 1) + (lane & 7)) * KS + 16 * (18 * kh + (m4 & 1));
#pragma unroll
        for (int j = 0; j < 18; j += 2) {   // matrices (nb0, j), (nb0, j+1), (nb1, j), (nb1, j+1)
          uint32_t kb[4];
          ldsm_x4(kbase + 16 * j, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const uint32_t b0 = e4m3x2_lo(kb[m]), b1 = e4m3x2_hi(kb[m]);
            const int nb = m >> 1, jj = j + (m & 1);
            mma16816(s[nb], qf[jj][0], qf[jj][1], qf[jj][2], qf[jj][3], b0, b1);
          }
        }
      }
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        const int row = 16 * nh + 8 * nb + 2 * tq;
        *reinterpret_cast<float2*>(Sp + (kh * 16 + g) * SPS + row) = make_float2(s[nb][0], s[nb][1]);
        *reinterpret_cast<float2*>(Sp + (kh * 16 + 8 + g) * SPS + row) = make_float2(s[nb][2], s[nb][3]);
      }
    }
    PROF_T(2)
    __syncthreads();
    PROF_T(3)
    {
      float4 a = *reinterpret_cast<const float4*>(Sp + sh * SPS + 4 * sc);
      const float4 b = *reinterpret_cast<const float4*>(Sp + (16 + sh) * SPS + 4 * sc);
      a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      const uint32_t vf = *reinterpret_cast<const uint32_t*>(vflag + stg * BN + 4 * sc);
      float sv[4] = {a.x * sm_scale, a.y * sm_scale, a.z * sm_scale, a.w * sm_scale};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!((vf >> (8 * i)) & 0xFF)) sv[i] = -1.0e30f;
      float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
      const float mn = fmaxf(m_run, mx);
      const float re = exp2f(m_run - mn);
      m_run = mn;
      float p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = ((vf >> (8 * i)) & 0xFF) ? exp2f(sv[i] - mn) : 0.f;
      l_part = l_part * re + (p[0] + p[1]) + (p[2] + p[3]);
      if (sc == 0) re_s[sh] = re;
      if (PVF8) {
        const int r0 = 4 * sc;
        uint8_t* pr = Pp + sh * P8S;
        *reinterpret_cast<uint16_t*>(pr + p8pos(r0)) = (uint16_t)pack_e4m3x2(p[0] * 256.f, p[1] * 256.f);
        *reinterpret_cast<uint16_t*>(pr + p8pos(r0 + 2)) = (uint16_t)pack_e4m3x2(p[2] * 256.f, p[3] * 256.f);
      } else {
        __half* pr = reinterpret_cast<__half*>(Pp) + sh * PS + 4 * sc;
        *reinterpret_cast<uint2*>(pr) = make_uint2(pack_h2(p[0], p[1]), pack_h2(p[2], p[3]));
      }
    }
    PROF_T(4)
    __syncthreads();
    PROF_T(5)
    {
      const float r0s = re_s[g], r1s = re_s[8 + g];
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        acc[nb][0] *= r0s;
        acc[nb][1] *= r0s;
        acc[nb][2] *= r1s;
        acc[nb][3] *= r1s;
      }
      if (ABL & 16) {
      } else if (PVF8) {
        const uint8_t* p0 = Pp + g * P8S + 4 * tq;
        const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(p0), pa1 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S),
                       pa2 = *reinterpret_cast<const uint32_t*>(p0 + 16),
                       pa3 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S + 16);
        const uint32_t vaddr = kt_base + lane * KS + 128 * warp;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) {
          uint32_t r0, r1, r2, r3;
          ldsm_x4_t(vaddr + 16 * cb, r0, r1, r2, r3);
          mma16832f8(acc[2 * cb], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x6420), prmt(r2, r3, 0x6420));
          mma16832f8(acc[2 * cb + 1], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x7531), prmt(r2, r3, 0x7531));
        }
      } else {
        const uint32_t ps_base = smem_u32(Pp);
#pragma unroll
        for (int kc = 0; kc < BN / 16; ++kc) {
          uint32_t pa[4];
          ldsm_x4(ps_base + ((lane & 15) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[0], pa[1], pa[2], pa[3]);
          const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + 128 * warp +
                                 (lane >> 4) * 16;
#pragma unroll
          for (int cb2 = 0; cb2 < 4; ++cb2) {
            uint32_t r0, r1, r2, r3;
            ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
              uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
              int cb = cb2 * 2 + h2;
              mma16816(acc[2 * cb], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xe), e4m3x2_hi(xe));
              mma16816(acc[2 * cb + 1], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xo), e4m3x2_hi(xo));
            }
          }
        }
      }
    }
    PROF_T(6)
  }
  cp_wait<0>();
  PROF_END
  l_part += __shfl_xor_sync(0xffffffff, l_part, 1);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 2);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 4);
  if (sc == 0) l_s[sh] = l_part;
  __syncthreads();
  const float ks_ = PVF8 ? kv_scale * (1.f / 256.f) : kv_scale;
  const float a = ks_ / fmaxf(l_s[g], 1e-30f), b = ks_ / fmaxf(l_s[8 + g], 1e-30f);
  __nv_bfloat16* ot = out + (long long)t * sot + (long long)(16 * hg) * soh;
#pragma unroll
  for (int cb = 0; cb < 8; ++cb) {
    int col = 128 * warp + cb * 16 + 4 * tq;
    __nv_bfloat162 r0[2] = {__floats2bfloat162_rn(acc[2 * cb][0] * a, acc[2 * cb + 1][0] * a),
                            __floats2bfloat162_rn(acc[2 * cb][1] * a, acc[2 * cb + 1][1] * a)};
    __nv_bfloat162 r1[2] = {__floats2bfloat162_rn(acc[2 * cb][2] * b, acc[2 * cb + 1][2] * b),
                            __floats2bfloat162_rn(acc[2 * cb][3] * b, acc[2 * cb + 1][3] * b)};
    *reinterpret_cast<uint2*>(ot + g * soh + col) = *reinterpret_cast<uint2*>(r0);
    *reinterpret_cast<uint2*>(ot + (8 + g) * soh + col) = *reinterpret_cast<uint2*>(r1);
  }
}

template <int ST, bool PVF8, bool HC, int ABL = 0>
int v3_launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx, const void* res,
              long long T, long long R, long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out,
              long long sot, long long soh, float kv_scale, const void* lflag, const void* sflag, int fmt, void* stream) {
  constexpr int SM = v3_smem<ST, PVF8>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(v3_kernel<ST, PVF8, HC, ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  v3_kernel<ST, PVF8, HC, ABL><<<(unsigned)(2 * T), V3_NT, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)idx,
      (const int*)res, R, sqt, sqh, sit, topk, sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)lflag,
      (int*)sflag, fmt);
  return (int)cudaGetLastError();
}

// =====================================================================================================================
// v4 decode kernel: warp-specialized ping-pong. CTA = one token, 12 warps (9 would get the same 168-register cap):
//   warps 8-11  loaders (8 tile rows each): row metadata (coalesced, one tile ahead in registers), overlapped-fill claims, compact host-row
//               dequant, cp.async gather into ST stages; signals full[s] (mbarrier, cp.async.mbarrier.arrive) and waits
//               on empty[s] before reusing a stage.
//   warps 0-3   head group A (heads 0-15), warps 4-7 head group B (heads 16-31). Per group as v3 (Q.K warp (kh, nh),
//               Q in registers; P.V warp -> 128 latent columns), group-local named barriers only.
// Per group and tile the MMAs run as one block {Q.K(i+1), P.V(i)}; the two groups take strict turns on that block
// (named-barrier token passing, as FA3's warpgroup ping-pong), so one group's softmax runs under the other's MMAs.
// The L4 runs this kernel power-limited (~1.1-1.4 GHz): v2's phase profile had the tensor pipe idle for ~1/3 of
// every tile (row issue + softmax between CTA-wide barriers).
constexpr int V4_NT = 384, V4_ST = 4, V4_NLW = 4;
template <int ST, bool PVF8>
constexpr int v4_smem() {
  return ST * BN * KS + 2 * 2 * 16 * SPS * 4 + 2 * (PVF8 ? 16 * P8S : 16 * PS * 2) + 2 * 2 * 16 * 4 +
         ((ST * BN + 15) & ~15) + 2 * ST * 8;
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%0]; }\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_cp_arrive(uint64_t* b) {   // arrives when this thread's prior cp.asyncs land
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  uint32_t ok = 0;
  while (true) {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    if (ok) break;
  }
}
__device__ __forceinline__ void nbar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }

template <int ST, bool PVF8, bool HC, int ABL = 0>
__global__ void __launch_bounds__(V4_NT, 1)
v4_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host, const uint8_t* __restrict__ cache,
          const uint8_t* __restrict__ spill, const int* __restrict__ idx, const int* __restrict__ res, long long R,
          long long stride_q_t, long long stride_q_h, long long stride_i_t, int topk, float sm_scale,
          __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale, int* lflag, int* sflag,
          int fmt) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Ks = smem;                                                    // [ST][BN][KS]
  float* SpAll = reinterpret_cast<float*>(Ks + ST * BN * KS);            // [2 groups][2 kh][16][SPS]
  constexpr int PB = PVF8 ? 16 * P8S : 16 * PS * 2;
  uint8_t* PpAll = reinterpret_cast<uint8_t*>(SpAll + 2 * 2 * 16 * SPS);  // [2 groups][PB]
  float* reAll = reinterpret_cast<float*>(PpAll + 2 * PB);               // [2 groups][16] rescale, [2][16] row sums
  float* lAll = reAll + 2 * 16;
  uint8_t* vflag = reinterpret_cast<uint8_t*>(lAll + 2 * 16);            // [ST][BN]
  uint64_t* full = reinterpret_cast<uint64_t*>(vflag + ((ST * BN + 15) & ~15));
  uint64_t* empty = full + ST;

  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int ntiles = topk / BN;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(full + s, 64 * V4_NLW);   // per loader lane: one cp.async arrival + one plain arrival
      mbar_init(empty + s, 8);    // one per compute warp
    }
  }
  __syncthreads();

  if (warp >= 8) {   // ================================ loader warps: rows 8 lw .. 8 lw + 7
    const HostFmt hf = host_fmt(fmt);
    const long long HD = HC ? hf.rb : D;
    const int* idx_t = idx + (long long)t * stride_i_t;
    const int* res_t = res + (long long)t * topk;
    const int lsub = lane & 7, lw = warp - 8;
    int mh = idx_t[lane], mrr = res_t[lane];   // metadata of tile 0 (lane = tile row)
    for (int j = 0; j < ntiles; ++j) {
      const int s = j % ST;
      int nh_ = -1, nr_ = 0;
      if (j + 1 < ntiles) {   // next tile's metadata in flight during this tile's issue
        nh_ = idx_t[(j + 1) * BN + lane];
        nr_ = res_t[(j + 1) * BN + lane];
      }
      if (j >= ST) mbar_wait(empty + s, ((j / ST) - 1) & 1);
      {
#pragma unroll
        for (int half = 0; half < 2; ++half) {   // 4 rows per half: lanes 8 m .. 8 m + 7 -> one row
          const int row = 8 * lw + 4 * half + (lane >> 3);
          const int h = __shfl_sync(0xffffffffu, mh, row);
          int r = __shfl_sync(0xffffffffu, mrr, row);
          const bool v = (h >= 0) && ((long long)h < R);
          bool hc = false;
          const uint8_t* src = cache;
          if (v) {
            if (r >= 0) {
              if (r >= FRESH) {
                r -= FRESH;
                if (lflag) fetch_or_wait<8, HC>(lflag + r, host + (long long)h * HD, const_cast<uint8_t*>(cache) + (long long)r * D, lsub, hf);
              }
              src = cache + (long long)r * D;
            } else if (r <= -2) {
              if (sflag)
                fetch_or_wait<8, HC>(sflag + (-r - 2), host + (long long)h * HD, const_cast<uint8_t*>(spill) + (long long)(-r - 2) * D, lsub, hf);
              src = spill + (long long)(-r - 2) * D;
            } else {
              hc = HC;
              src = host + (long long)h * HD;
            }
          }
          uint8_t* dst = Ks + s * BN * KS + row * KS;
          if (HC && hc) {
#pragma unroll
            for (int k = 0; k < 5; ++k) {
              const int c = lsub + 8 * k;
              if (c < D / 16) *reinterpret_cast<int4*>(dst + c * 16) = hf_chunk(src, c, hf);
            }
          } else {
#pragma unroll
            for (int k = 0; k < 5; ++k) {
              const int c = lsub + 8 * k;
              if (c < D / 16) cp_async16(smem_u32(dst + c * 16), src + c * 16, v);
            }
          }
          if (lsub == 0) vflag[s * BN + row] = v;
        }
      }
      mbar_cp_arrive(full + s);
      __threadfence_block();
      mbar_arrive(full + s);
      mh = nh_;
      mrr = nr_;
    }
    return;
  }

  // ================================ compute warps: group grp (heads 16 grp ..), warp gw in the group
  const int grp = warp >> 2, gw = warp & 3, gtid = tid & 127;
  const int kh = gw & 1, nh = gw >> 1;
  float* Sp = SpAll + grp * 2 * 16 * SPS;
  uint8_t* Pp = PpAll + grp * PB;
  float* re_s = reAll + grp * 16;
  float* l_s = lAll + grp * 16;
  const int gbar = 1 + grp;                         // group barrier (128 threads)
  const int my_turn = 3 + grp, other_turn = 4 - grp;   // MMA-block turn barriers (256 threads)

  uint32_t qf[18][4];
  {
    const __nv_bfloat16* qt = q + (long long)t * stride_q_t;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const __nv_bfloat16* qr = qt + (long long)(16 * grp + 8 * hh + g) * stride_q_h + 4 * tq;
#pragma unroll
      for (int j = 0; j < 18; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(qr + 16 * (18 * kh + j));
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        qf[j][hh] = pack_h2(a.x, a.y);
        qf[j][2 + hh] = pack_h2(b.x, b.y);
      }
    }
  }
  float acc[16][4];
#pragma unroll
  for (int b = 0; b < 16; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const int sh = gtid >> 3, sc = gtid & 7;
  float m_run = -1.0e30f, l_part = 0.f;

  // partial S of tile j (stage j % ST) over k-half kh, rows 16 nh .. +15 -> Sp
  auto qk = [&](int j) {
    const uint32_t kt_base = smem_u32(Ks + (j % ST) * BN * KS);
    float s4[4][4];   // 4 independent accumulator chains (n8 block, k parity): one warp per SMSP issues the MMAs
#pragma unroll
    for (int b = 0; b < 4; ++b) s4[b][0] = s4[b][1] = s4[b][2] = s4[b][3] = 0.f;
    const int m4 = lane >> 3;
    const uint32_t kbase = kt_base + (16 * nh + 8 * (m4 >> 1) + (lane & 7)) * KS + 16 * (18 * kh + (m4 & 1));
#pragma unroll
    for (int jj = 0; jj < 18; jj += 2) {
      uint32_t kb[4];
      ldsm_x4(kbase + 16 * jj, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const uint32_t b0 = e4m3x2_lo(kb[m]), b1 = e4m3x2_hi(kb[m]);
        const int kk = jj + (m & 1);
        mma16816(s4[m], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
      }
    }
    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int c = 0; c < 4; ++c) s[nb][c] = s4[2 * nb][c] + s4[2 * nb + 1][c];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      const int row = 16 * nh + 8 * nb + 2 * tq;
      *reinterpret_cast<float2*>(Sp + (kh * 16 + g) * SPS + row) = make_float2(s[nb][0], s[nb][1]);
      *reinterpret_cast<float2*>(Sp + (kh * 16 + 8 + g) * SPS + row) = make_float2(s[nb][2], s[nb][3]);
    }
  };
  // softmax of tile j: Sp -> P, re_s (group-local)
  auto softmax = [&](int j) {
    float4 a = *reinterpret_cast<const float4*>(Sp + sh * SPS + 4 * sc);
    const float4 b = *reinterpret_cast<const float4*>(Sp + (16 + sh) * SPS + 4 * sc);
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    const uint32_t vf = *reinterpret_cast<const uint32_t*>(vflag + (j % ST) * BN + 4 * sc);
    float sv[4] = {a.x * sm_scale, a.y * sm_scale, a.z * sm_scale, a.w * sm_scale};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!((vf >> (8 * i)) & 0xFF)) sv[i] = -1.0e30f;
    float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
    const float mn = fmaxf(m_run, mx);
    const float re = exp2f(m_run - mn);
    m_run = mn;
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = ((vf >> (8 * i)) & 0xFF) ? exp2f(sv[i] - mn) : 0.f;
    l_part = l_part * re + (p[0] + p[1]) + (p[2] + p[3]);
    if (sc == 0) re_s[sh] = re;
    if (PVF8) {
      const int r0 = 4 * sc;
      uint8_t* pr = Pp + sh * P8S;
      *reinterpret_cast<uint16_t*>(pr + p8pos(r0)) = (uint16_t)pack_e4m3x2(p[0] * 256.f, p[1] * 256.f);
      *reinterpret_cast<uint16_t*>(pr + p8pos(r0 + 2)) = (uint16_t)pack_e4m3x2(p[2] * 256.f, p[3] * 256.f);
    } else {
      __half* pr = reinterpret_cast<__half*>(Pp) + sh * PS + 4 * sc;
      *reinterpret_cast<uint2*>(pr) = make_uint2(pack_h2(p[0], p[1]), pack_h2(p[2], p[3]));
    }
  };
  // O = O * re + P V for tile j (stage j % ST), columns [128 gw, +128)
  auto pv = [&](int j) {
    const uint32_t kt_base = smem_u32(Ks + (j % ST) * BN * KS);
    const float r0s = re_s[g], r1s = re_s[8 + g];
#pragma unroll
    for (int nb = 0; nb < 16; ++nb) {
      acc[nb][0] *= r0s;
      acc[nb][1] *= r0s;
      acc[nb][2] *= r1s;
      acc[nb][3] *= r1s;
    }
    if (PVF8) {
      const uint8_t* p0 = Pp + g * P8S + 4 * tq;
      const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(p0), pa1 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S),
                     pa2 = *reinterpret_cast<const uint32_t*>(p0 + 16),
                     pa3 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S + 16);
      const uint32_t vaddr = kt_base + lane * KS + 128 * gw;
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vaddr + 16 * cb, r0, r1, r2, r3);
        mma16832f8(acc[2 * cb], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x6420), prmt(r2, r3, 0x6420));
        mma16832f8(acc[2 * cb + 1], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x7531), prmt(r2, r3, 0x7531));
      }
    } else {
      const uint32_t ps_base = smem_u32(Pp);
#pragma unroll
      for (int kc = 0; kc < BN / 16; ++kc) {
        uint32_t pa[4];
        ldsm_x4(ps_base + ((lane & 15) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[0], pa[1], pa[2], pa[3]);
        const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + 128 * gw +
                               (lane >> 4) * 16;
#pragma unroll
        for (int cb2 = 0; cb2 < 4; ++cb2) {
          uint32_t r0, r1, r2, r3;
          ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
            uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
            int cb = cb2 * 2 + h2;
            mma16816(acc[2 * cb], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xe), e4m3x2_hi(xe));
            mma16816(acc[2 * cb + 1], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xo), e4m3x2_hi(xo));
          }
        }
      }
    }
  };

  PROF_INIT
  // prologue: Q.K(0) (group A first), softmax(0)
  if (grp == 1) nbar_arrive(3, 256);   // A takes the first turn
  mbar_wait(full + 0, 0);
  nbar_sync(my_turn, 256);
  qk(0);
  nbar_arrive(other_turn, 256);
  nbar_sync(gbar, 128);
  softmax(0);
  nbar_sync(gbar, 128);
  for (int i = 0; i < ntiles; ++i) {
    const bool nx = i + 1 < ntiles;
    if (nx) mbar_wait(full + (i + 1) % ST, ((i + 1) / ST) & 1);
    PROF_T(0)
    nbar_sync(my_turn, 256);
    PROF_T(1)
    if (nx) qk(i + 1);
    pv(i);
    nbar_arrive(other_turn, 256);
    PROF_T(2)
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + i % ST);   // this warp is done with stage i
    nbar_sync(gbar, 128);                          // Sp(i+1) complete; P(i) consumed by the whole group
    PROF_T(3)
    if (nx) softmax(i + 1);
    PROF_T(4)
    nbar_sync(gbar, 128);                          // P(i+1), re(i+1) visible
    PROF_T(5)
  }
  if (grp == 0) nbar_sync(my_turn, 256);   // consume B's final turn token (keeps the barrier counts balanced)
  PROF_END
  l_part += __shfl_xor_sync(0xffffffff, l_part, 1);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 2);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 4);
  if (sc == 0) l_s[sh] = l_part;
  nbar_sync(gbar, 128);
  const float ks_ = PVF8 ? kv_scale * (1.f / 256.f) : kv_scale;
  const float a = ks_ / fmaxf(l_s[g], 1e-30f), b = ks_ / fmaxf(l_s[8 + g], 1e-30f);
  __nv_bfloat16* ot = out + (long long)t * sot + (long long)(16 * grp) * soh;
#pragma unroll
  for (int cb = 0; cb < 8; ++cb) {
    int col = 128 * gw + cb * 16 + 4 * tq;
    __nv_bfloat162 r0[2] = {__floats2bfloat162_rn(acc[2 * cb][0] * a, acc[2 * cb + 1][0] * a),
                            __floats2bfloat162_rn(acc[2 * cb][1] * a, acc[2 * cb + 1][1] * a)};
    __nv_bfloat162 r1[2] = {__floats2bfloat162_rn(acc[2 * cb][2] * b, acc[2 * cb + 1][2] * b),
                            __floats2bfloat162_rn(acc[2 * cb][3] * b, acc[2 * cb + 1][3] * b)};
    *reinterpret_cast<uint2*>(ot + g * soh + col) = *reinterpret_cast<uint2*>(r0);
    *reinterpret_cast<uint2*>(ot + (8 + g) * soh + col) = *reinterpret_cast<uint2*>(r1);
  }
}

template <int ST, bool PVF8, bool HC, int ABL = 0>
int v4_launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx, const void* res,
              long long T, long long R, long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out,
              long long sot, long long soh, float kv_scale, const void* lflag, const void* sflag, int fmt, void* stream) {
  constexpr int SM = v4_smem<ST, PVF8>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(v4_kernel<ST, PVF8, HC, ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  v4_kernel<ST, PVF8, HC, ABL><<<(unsigned)T, V4_NT, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)idx,
      (const int*)res, R, sqt, sqh, sit, topk, sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)lflag,
      (int*)sflag, fmt);
  return (int)cudaGetLastError();
}

// =====================================================================================================================
// v5 decode kernel: v4's ping-pong without loader warps (8 warps -> 255-register cap instead of 168). Each head group
// gathers half the rows of every tile (rows 16 grp .. +15, 8 threads per row) right after its MMA block, i.e. while the
// other group holds the tensor cores; row metadata is prefetched one tile ahead in registers.
constexpr int V5_NT = 256;
template <int ST, bool PVF8>
constexpr int v5_smem() {
  return ST * BN * KS + 2 * 2 * 16 * SPS * 4 + 2 * (PVF8 ? 16 * P8S : 16 * PS * 2) + 2 * 2 * 16 * 4 +
         ((ST * BN + 15) & ~15) + 2 * ST * 8;
}
template <int ST, bool PVF8, bool HC, int ABL = 0, bool CC = false>
// register cap: the overlapped fill copy (offload_attn.cu fill_copy_kernel, 128 threads, <= 72 registers) must still
// fit beside one attention CTA on every SM (256 x 216 + 128 x 72 <= 65536), else the fill starves and the attention
// loaders end up copying the claimed rows themselves (in a serving test, most of the kernel gain was lost at 229 registers).
#ifndef V5_MAXNREG
#define V5_MAXNREG 216
#endif
__global__ void __maxnreg__(V5_MAXNREG)
v5_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ host, const uint8_t* __restrict__ cache,
          const uint8_t* __restrict__ spill, const int* __restrict__ idx, const int* __restrict__ res, long long R,
          long long stride_q_t, long long stride_q_h, long long stride_i_t, int topk, float sm_scale,
          __nv_bfloat16* __restrict__ out, long long sot, long long soh, float kv_scale, int* lflag, int* sflag,
          int fmt) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Ks = smem;                                                    // [ST][BN][KS]
  float* SpAll = reinterpret_cast<float*>(Ks + ST * BN * KS);            // [2 groups][2 kh][16][SPS]
  constexpr int PB = PVF8 ? 16 * P8S : 16 * PS * 2;
  uint8_t* PpAll = reinterpret_cast<uint8_t*>(SpAll + 2 * 2 * 16 * SPS);  // [2 groups][PB]
  float* reAll = reinterpret_cast<float*>(PpAll + 2 * PB);               // [2 groups][16] rescale, [2][16] row sums
  float* lAll = reAll + 2 * 16;
  uint8_t* vflag = reinterpret_cast<uint8_t*>(lAll + 2 * 16);            // [ST][BN]
  uint64_t* full = reinterpret_cast<uint64_t*>(vflag + ((ST * BN + 15) & ~15));
  uint64_t* empty = full + ST;

  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int ntiles = topk / BN;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(full + s, CC ? V5_NT : 2 * V5_NT);   // per thread: one cp.async arrival + one plain arrival (CC: plain only)
      mbar_init(empty + s, 8);    // one per compute warp
    }
  }
  __syncthreads();

  // ================================ compute warps: group grp (heads 16 grp ..), warp gw in the group
  const int grp = warp >> 2, gw = warp & 3, gtid = tid & 127;
  const int kh = gw & 1, nh = gw >> 1;
  float* Sp = SpAll + grp * 2 * 16 * SPS;
  uint8_t* Pp = PpAll + grp * PB;
  float* re_s = reAll + grp * 16;
  float* l_s = lAll + grp * 16;
  const int gbar = 1 + grp;                         // group barrier (128 threads)
  const int my_turn = 3 + grp, other_turn = 4 - grp;   // MMA-block turn barriers (256 threads)

  // ---- row gather of this group's half tile: rows 16 grp + (gtid >> 3), 16 B chunks gtid & 7 + 8 k
  const HostFmt hf = host_fmt(fmt);
  const long long HD = HC ? hf.rb : D;
  const int* idx_t = idx + (long long)t * stride_i_t;
  const int* res_t = res + (long long)t * topk;
  const int lrow = 16 * grp + (gtid >> 3), lsub = gtid & 7;
  int mh = -1, mrr = 0;   // metadata of the next tile this thread issues
  auto meta = [&](int j) {
    mh = -1;
    mrr = 0;
    if (j < ntiles) {
      mh = idx_t[j * BN + lrow];
      mrr = res_t[j * BN + lrow];
    }
  };
  // CC (compact GPU rows): load(j) issues tile j's compact bytes (row slot: lo plane [0, 256), sign plane [256, 320),
  // exponents [320, 336), rope at its final place [512, 576); 16 B cp.async chunks over the row's 8 lanes), then
  // waits for tile j - 1's bytes (cp.async group), decodes this thread's 64 latent values in place and arrives on its
  // full barrier. Needs ST >= 4 (the consumer of tile j - 1 waits two loads later).
  auto cc_issue = [&](int j) {
    if (j < ntiles) {
      const int s = j % ST;
      if (j >= ST) mbar_wait(empty + s, ((j / ST) - 1) & 1);
      const int h = mh;
      int r = mrr;
      const bool v = (h >= 0) && ((long long)h < R);
      const uint8_t* src = host + (long long)(v ? h : 0) * CC_RB;
      if (v) {
        if (r >= 0) {
          if (r >= FRESH) {
            r -= FRESH;
            if (lflag) fetch_or_wait_raw<8>(lflag + r, src, const_cast<uint8_t*>(cache) + (long long)r * CC_RB, lsub);
          }
          src = cache + (long long)r * CC_RB;
        } else if (r <= -2) {
          if (sflag) fetch_or_wait_raw<8>(sflag + (-r - 2), src, const_cast<uint8_t*>(spill) + (long long)(-r - 2) * CC_RB, lsub);
          src = spill + (long long)(-r - 2) * CC_RB;
        }
      }
      const uint32_t d0 = smem_u32(Ks + s * BN * KS + lrow * KS);
#pragma unroll
      for (int c0 = 0; c0 < 24; c0 += 8) {   // 16 B chunks c (0..24): [0, 336) in place, rope chunks 21..24 -> [512, 576)
        const int c = c0 + lsub;
        cp_async16(d0 + (c < 21 ? 16 * c : 512 + 16 * (c - 21)), src + 16 * c, v);
      }
      if (lsub == 0) cp_async16(d0 + 512 + 48, src + 16 * 24, v);
      if (lsub == 0) vflag[s * BN + lrow] = v;
      meta(j + 1);
    }
    cp_commit();
  };
  auto cc_decode = [&](int jd) {   // tile jd (its cp.async group is the second most recent one)
    if (jd < 0 || jd >= ntiles) return;
    if constexpr (ABL == 256) {   // (bench ablation: no wait, no decode)
      mbar_arrive(full + jd % ST);
      return;
    }
    cp_wait<1>();
    __syncwarp();   // the row's bytes were gathered by all 8 of its lanes
    if constexpr (ABL == 128) {   // (bench ablation: wait, no decode)
      __threadfence_block();
      mbar_arrive(full + jd % ST);
      return;
    }
    uint8_t* row = Ks + (jd % ST) * BN * KS + lrow * KS;
    // lane lsub decodes 16-value chunks c = lsub + 8 m (m = 0..3): per store instruction the row's 8 lanes write 128
    // contiguous bytes (bank-conflict free); sources: lo bytes [8 c, +8), sign bits 16 c.. (2 bytes), exponent c / 2
    uint2 lo[4];
    uint32_t sg[4];
    int kk[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int c = lsub + 8 * m;
      lo[m] = *reinterpret_cast<const uint2*>(row + 8 * c);
      sg[m] = *reinterpret_cast<const uint16_t*>(row + 256 + 2 * c);
      kk[m] = (int)*reinterpret_cast<const int8_t*>(row + 320 + (c >> 1));
    }
    __syncwarp();   // every lane of the warp has read its sources before any lane overwrites them
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      uint32_t T[4];
      cc_table(kk[m], T);
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t w = q ? lo[m].y : lo[m].x;
        const uint32_t sel = w & 0x77777777u;
        const uint32_t mrg = ((w >> 1) & 0x44444444u) | 0x32103210u;
        const uint32_t s8 = sg[m] >> (8 * q);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t sl = h ? sel >> 16 : sel, mg = h ? mrg >> 16 : mrg;
          const uint32_t a = prmt(T[0], T[1], sl), b = prmt(T[2], T[3], sl);
          const uint32_t sb = (((h ? s8 >> 4 : s8) & 15u) * 0x10204080u) & 0x80808080u;
          o[2 * q + h] = prmt(a, b, mg) | sb;
        }
      }
      *reinterpret_cast<uint4*>(row + 16 * (lsub + 8 * m)) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    __threadfence_block();
    mbar_arrive(full + jd % ST);
  };
  auto load = [&](int j) {   // tile j -> stage j % ST (after both groups released its previous tile)
    if constexpr (CC) {
      cc_issue(j);
      cc_decode(j - 1);
      return;
    }
    if (j >= ntiles) return;
    const int s = j % ST;
    if (j >= ST) mbar_wait(empty + s, ((j / ST) - 1) & 1);
    const int h = mh;
    int r = mrr;
    const bool v = (h >= 0) && ((long long)h < R);
    bool hc = false;
    const uint8_t* src = cache;
    if (v) {
      if (r >= 0) {
        if (r >= FRESH) {
          r -= FRESH;
          if (lflag) fetch_or_wait<8, HC>(lflag + r, host + (long long)h * HD, const_cast<uint8_t*>(cache) + (long long)r * D, lsub, hf);
        }
        src = cache + (long long)r * D;
      } else if (r <= -2) {
        if (sflag)
          fetch_or_wait<8, HC>(sflag + (-r - 2), host + (long long)h * HD, const_cast<uint8_t*>(spill) + (long long)(-r - 2) * D, lsub, hf);
        src = spill + (long long)(-r - 2) * D;
      } else {
        hc = HC;
        src = host + (long long)h * HD;
      }
    }
    uint8_t* dst = Ks + s * BN * KS + lrow * KS;
    if (HC && hc) {
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const int c = lsub + 8 * k;
        if (c < D / 16) *reinterpret_cast<int4*>(dst + c * 16) = hf_chunk(src, c, hf);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const int c = lsub + 8 * k;
        if (c < D / 16) cp_async16(smem_u32(dst + c * 16), src + c * 16, v);
      }
    }
    if (lsub == 0) vflag[s * BN + lrow] = v;
    mbar_cp_arrive(full + s);
    __threadfence_block();
    mbar_arrive(full + s);
    meta(j + 1);
  };
  meta(0);
#pragma unroll 1
  for (int j = 0; j < ST - 1; ++j) load(j);

  uint32_t qf[18][4];
  {
    const __nv_bfloat16* qt = q + (long long)t * stride_q_t;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const __nv_bfloat16* qr = qt + (long long)(16 * grp + 8 * hh + g) * stride_q_h + 4 * tq;
#pragma unroll
      for (int j = 0; j < 18; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(qr + 16 * (18 * kh + j));
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        qf[j][hh] = pack_h2(a.x, a.y);
        qf[j][2 + hh] = pack_h2(b.x, b.y);
      }
    }
  }
  float acc[16][4];
#pragma unroll
  for (int b = 0; b < 16; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const int sh = gtid >> 3, sc = gtid & 7;
  float m_run = -1.0e30f, l_part = 0.f;

  // partial S of tile j (stage j % ST) over k-half kh, rows 16 nh .. +15 -> Sp
  auto qk = [&](int j) {
    const uint32_t kt_base = smem_u32(Ks + (j % ST) * BN * KS);
    float s4[4][4];   // 4 independent accumulator chains (n8 block, k parity): one warp per SMSP issues the MMAs
#pragma unroll
    for (int b = 0; b < 4; ++b) s4[b][0] = s4[b][1] = s4[b][2] = s4[b][3] = 0.f;
    const int m4 = lane >> 3;
    const uint32_t kbase = kt_base + (16 * nh + 8 * (m4 >> 1) + (lane & 7)) * KS + 16 * (18 * kh + (m4 & 1));
#pragma unroll
    for (int jj = 0; jj < 18; jj += 2) {
      uint32_t kb[4];
      ldsm_x4(kbase + 16 * jj, kb[0], kb[1], kb[2], kb[3]);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const uint32_t b0 = e4m3x2_lo(kb[m]), b1 = e4m3x2_hi(kb[m]);
        const int kk = jj + (m & 1);
        mma16816(s4[m], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
      }
    }
    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int c = 0; c < 4; ++c) s[nb][c] = s4[2 * nb][c] + s4[2 * nb + 1][c];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      const int row = 16 * nh + 8 * nb + 2 * tq;
      *reinterpret_cast<float2*>(Sp + (kh * 16 + g) * SPS + row) = make_float2(s[nb][0], s[nb][1]);
      *reinterpret_cast<float2*>(Sp + (kh * 16 + 8 + g) * SPS + row) = make_float2(s[nb][2], s[nb][3]);
    }
  };
  // softmax of tile j: Sp -> P, re_s (group-local)
  auto softmax = [&](int j) {
    float4 a = *reinterpret_cast<const float4*>(Sp + sh * SPS + 4 * sc);
    const float4 b = *reinterpret_cast<const float4*>(Sp + (16 + sh) * SPS + 4 * sc);
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    const uint32_t vf = *reinterpret_cast<const uint32_t*>(vflag + (j % ST) * BN + 4 * sc);
    float sv[4] = {a.x * sm_scale, a.y * sm_scale, a.z * sm_scale, a.w * sm_scale};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!((vf >> (8 * i)) & 0xFF)) sv[i] = -1.0e30f;
    float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
    const float mn = fmaxf(m_run, mx);
    const float re = exp2f(m_run - mn);
    m_run = mn;
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = ((vf >> (8 * i)) & 0xFF) ? exp2f(sv[i] - mn) : 0.f;
    l_part = l_part * re + (p[0] + p[1]) + (p[2] + p[3]);
    if (sc == 0) re_s[sh] = re;
    if (PVF8) {
      const int r0 = 4 * sc;
      uint8_t* pr = Pp + sh * P8S;
      *reinterpret_cast<uint16_t*>(pr + p8pos(r0)) = (uint16_t)pack_e4m3x2(p[0] * 256.f, p[1] * 256.f);
      *reinterpret_cast<uint16_t*>(pr + p8pos(r0 + 2)) = (uint16_t)pack_e4m3x2(p[2] * 256.f, p[3] * 256.f);
    } else {
      __half* pr = reinterpret_cast<__half*>(Pp) + sh * PS + 4 * sc;
      *reinterpret_cast<uint2*>(pr) = make_uint2(pack_h2(p[0], p[1]), pack_h2(p[2], p[3]));
    }
  };
  // O = O * re + P V for tile j (stage j % ST), columns [128 gw, +128)
  auto pv = [&](int j) {
    const uint32_t kt_base = smem_u32(Ks + (j % ST) * BN * KS);
    const float r0s = re_s[g], r1s = re_s[8 + g];
#pragma unroll
    for (int nb = 0; nb < 16; ++nb) {
      acc[nb][0] *= r0s;
      acc[nb][1] *= r0s;
      acc[nb][2] *= r1s;
      acc[nb][3] *= r1s;
    }
    if (PVF8) {
      const uint8_t* p0 = Pp + g * P8S + 4 * tq;
      const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(p0), pa1 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S),
                     pa2 = *reinterpret_cast<const uint32_t*>(p0 + 16),
                     pa3 = *reinterpret_cast<const uint32_t*>(p0 + 8 * P8S + 16);
      const uint32_t vaddr = kt_base + lane * KS + 128 * gw;
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vaddr + 16 * cb, r0, r1, r2, r3);
        mma16832f8(acc[2 * cb], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x6420), prmt(r2, r3, 0x6420));
        mma16832f8(acc[2 * cb + 1], pa0, pa1, pa2, pa3, prmt(r0, r1, 0x7531), prmt(r2, r3, 0x7531));
      }
    } else {
      const uint32_t ps_base = smem_u32(Pp);
#pragma unroll
      for (int kc = 0; kc < BN / 16; ++kc) {
        uint32_t pa[4];
        ldsm_x4(ps_base + ((lane & 15) * PS + kc * 16 + (lane >> 4) * 8) * 2, pa[0], pa[1], pa[2], pa[3]);
        const uint32_t vaddr = kt_base + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + 128 * gw +
                               (lane >> 4) * 16;
#pragma unroll
        for (int cb2 = 0; cb2 < 4; ++cb2) {
          uint32_t r0, r1, r2, r3;
          ldsm_x4_t(vaddr + cb2 * 32, r0, r1, r2, r3);
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t lo = h2 ? r2 : r0, hi = h2 ? r3 : r1;
            uint32_t xe = prmt(lo, hi, 0x6420), xo = prmt(lo, hi, 0x7531);
            int cb = cb2 * 2 + h2;
            mma16816(acc[2 * cb], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xe), e4m3x2_hi(xe));
            mma16816(acc[2 * cb + 1], pa[0], pa[1], pa[2], pa[3], e4m3x2_lo(xo), e4m3x2_hi(xo));
          }
        }
      }
    }
  };

  PROF_INIT
  // prologue: Q.K(0) (group A first), softmax(0)
  if (grp == 1) nbar_arrive(3, 256);   // A takes the first turn
  mbar_wait(full + 0, 0);
  nbar_sync(my_turn, 256);
  qk(0);
  nbar_arrive(other_turn, 256);
  nbar_sync(gbar, 128);
  softmax(0);
  nbar_sync(gbar, 128);
  for (int i = 0; i < ntiles; ++i) {
    const bool nx = i + 1 < ntiles;
    if (nx) mbar_wait(full + (i + 1) % ST, ((i + 1) / ST) & 1);
    PROF_T(0)
    nbar_sync(my_turn, 256);
    PROF_T(1)
    if (nx) qk(i + 1);
    pv(i);
    nbar_arrive(other_turn, 256);
    PROF_T(2)
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + i % ST);   // this warp is done with stage i
    load(i + ST - 1);                              // while the other group runs its MMA block
    nbar_sync(gbar, 128);                          // Sp(i+1) complete; P(i) consumed by the whole group
    PROF_T(3)
    if (nx) softmax(i + 1);
    PROF_T(4)
    nbar_sync(gbar, 128);                          // P(i+1), re(i+1) visible
    PROF_T(5)
  }
  if (grp == 0) nbar_sync(my_turn, 256);   // consume B's final turn token (keeps the barrier counts balanced)
  PROF_END
  l_part += __shfl_xor_sync(0xffffffff, l_part, 1);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 2);
  l_part += __shfl_xor_sync(0xffffffff, l_part, 4);
  if (sc == 0) l_s[sh] = l_part;
  nbar_sync(gbar, 128);
  const float ks_ = PVF8 ? kv_scale * (1.f / 256.f) : kv_scale;
  const float a = ks_ / fmaxf(l_s[g], 1e-30f), b = ks_ / fmaxf(l_s[8 + g], 1e-30f);
  __nv_bfloat16* ot = out + (long long)t * sot + (long long)(16 * grp) * soh;
#pragma unroll
  for (int cb = 0; cb < 8; ++cb) {
    int col = 128 * gw + cb * 16 + 4 * tq;
    __nv_bfloat162 r0[2] = {__floats2bfloat162_rn(acc[2 * cb][0] * a, acc[2 * cb + 1][0] * a),
                            __floats2bfloat162_rn(acc[2 * cb][1] * a, acc[2 * cb + 1][1] * a)};
    __nv_bfloat162 r1[2] = {__floats2bfloat162_rn(acc[2 * cb][2] * b, acc[2 * cb + 1][2] * b),
                            __floats2bfloat162_rn(acc[2 * cb][3] * b, acc[2 * cb + 1][3] * b)};
    *reinterpret_cast<uint2*>(ot + g * soh + col) = *reinterpret_cast<uint2*>(r0);
    *reinterpret_cast<uint2*>(ot + (8 + g) * soh + col) = *reinterpret_cast<uint2*>(r1);
  }
}

template <int ST, bool PVF8, bool HC, int ABL = 0, bool CC = false>
int v5_launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx, const void* res,
              long long T, long long R, long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out,
              long long sot, long long soh, float kv_scale, const void* lflag, const void* sflag, int fmt, void* stream) {
  constexpr int SM = v5_smem<ST, PVF8>();
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(v5_kernel<ST, PVF8, HC, ABL, CC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  v5_kernel<ST, PVF8, HC, ABL, CC><<<(unsigned)T, V5_NT, SM, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const uint8_t*)host, (const uint8_t*)cache, (const uint8_t*)spill, (const int*)idx,
      (const int*)res, R, sqt, sqh, sit, topk, sm_scale, (__nv_bfloat16*)out, sot, soh, kv_scale, (int*)lflag,
      (int*)sflag, fmt);
  return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

// v2 decode (split 1, normalized bf16 out). cfg: ST (2 | 3) | 16 * PVF8 | 32 * v3 (head-split, 2 CTAs/SM) | 64 * v4 (warp-specialized ping-pong, ST 3 | 4) | 128 * v5 (ping-pong, self-loading groups) | ABL << 8. fmt: compact host row format (0 = fp8 rows).
// res must be non-null (resolved locations); topk <= 2048, multiple of 32. lflag / sflag: overlapped-fill claims.
int oa2_v2_launch(const void* q, const void* host, const void* cache, const void* spill, const void* idx,
                  const void* res, long long T, long long R, long long sqt, long long sqh, long long sit, int topk,
                  float sm_scale, void* out, long long sot, long long soh, float kv_scale, const void* lflag,
                  const void* sflag, int fmt, int cfg, void* stream) {
  if (res == nullptr || topk > V2_TOPK || topk % BN) return -2;
  const int st = cfg & 15;
  const bool f8 = (cfg >> 4) & 1;
  const int abl = cfg >> 8;
#define V2_ARGS q, host, cache, spill, idx, res, T, R, sqt, sqh, sit, topk, sm_scale, out, sot, soh, kv_scale, lflag, sflag, fmt, stream
  const bool v3 = (cfg >> 5) & 1, v4 = (cfg >> 6) & 1;
  const bool v5 = (cfg >> 7) & 1;
  if (v5 && ((fmt >> 16) & 1)) {   // ARC_OFFLOAD_CCACHE: compact e2m2b32 GPU rows, ST 4 only
    if ((fmt & 0xFFFF) != (2 | (32 << 4)) || st != 4) return -5;
    if (abl == 128) return f8 ? v5_launch<4, true, true, 128, true>(V2_ARGS) : v5_launch<4, false, true, 128, true>(V2_ARGS);
    if (abl == 256) return f8 ? v5_launch<4, true, true, 256, true>(V2_ARGS) : v5_launch<4, false, true, 256, true>(V2_ARGS);
    if (abl == 64) return f8 ? v5_launch<4, true, true, 64, true>(V2_ARGS) : v5_launch<4, false, true, 64, true>(V2_ARGS);
    if (abl) return -5;
    return f8 ? v5_launch<4, true, true, 0, true>(V2_ARGS) : v5_launch<4, false, true, 0, true>(V2_ARGS);
  }
  if ((fmt >> 16) & 1) return -5;
  if (v5) {   // ST 3 | 4
    if (abl == 64) return f8 ? v5_launch<4, true, false, 64>(V2_ARGS) : v5_launch<4, false, false, 64>(V2_ARGS);
    if (abl) return -3;
    if (fmt) {
      if (st == 3) return f8 ? v5_launch<3, true, true>(V2_ARGS) : v5_launch<3, false, true>(V2_ARGS);
      return f8 ? v5_launch<4, true, true>(V2_ARGS) : v5_launch<4, false, true>(V2_ARGS);
    }
    if (st == 3) return f8 ? v5_launch<3, true, false>(V2_ARGS) : v5_launch<3, false, false>(V2_ARGS);
    return f8 ? v5_launch<4, true, false>(V2_ARGS) : v5_launch<4, false, false>(V2_ARGS);
  }
  if (v4) {   // ST 3 | 4
    if (abl == 64) return f8 ? v4_launch<4, true, false, 64>(V2_ARGS) : v4_launch<4, false, false, 64>(V2_ARGS);
    if (abl) return -3;
    if (fmt) {
      if (st == 3) return f8 ? v4_launch<3, true, true>(V2_ARGS) : v4_launch<3, false, true>(V2_ARGS);
      return f8 ? v4_launch<4, true, true>(V2_ARGS) : v4_launch<4, false, true>(V2_ARGS);
    }
    if (st == 3) return f8 ? v4_launch<3, true, false>(V2_ARGS) : v4_launch<3, false, false>(V2_ARGS);
    return f8 ? v4_launch<4, true, false>(V2_ARGS) : v4_launch<4, false, false>(V2_ARGS);
  }
  if (v3) {
    if (abl) {
      switch (abl | (f8 ? 128 : 0)) {
        case 1: return v3_launch<2, false, false, 1>(V2_ARGS);
        case 8: return v3_launch<2, false, false, 8>(V2_ARGS);
        case 16: return v3_launch<2, false, false, 16>(V2_ARGS);
        case 64: return v3_launch<2, false, false, 64>(V2_ARGS);
        case 129: return v3_launch<2, true, false, 1>(V2_ARGS);
        case 136: return v3_launch<2, true, false, 8>(V2_ARGS);
        case 144: return v3_launch<2, true, false, 16>(V2_ARGS);
        case 192: return v3_launch<2, true, false, 64>(V2_ARGS);
        default: return -3;
      }
    }
    if (fmt) {
      if (st == 3) return f8 ? v3_launch<3, true, true>(V2_ARGS) : v3_launch<3, false, true>(V2_ARGS);
      return f8 ? v3_launch<2, true, true>(V2_ARGS) : v3_launch<2, false, true>(V2_ARGS);
    }
    if (st == 3) return f8 ? v3_launch<3, true, false>(V2_ARGS) : v3_launch<3, false, false>(V2_ARGS);
    return f8 ? v3_launch<2, true, false>(V2_ARGS) : v3_launch<2, false, false>(V2_ARGS);
  }
  if (abl) {   // bench ablations (ST 3, fp8 host rows only)
    switch (abl | (f8 ? 128 : 0)) {
      case 64: return v2_launch<3, false, false, 64>(V2_ARGS);
      case 192: return v2_launch<3, true, false, 64>(V2_ARGS);
      case 1: return v2_launch<3, false, false, 1>(V2_ARGS);
      case 8: return v2_launch<3, false, false, 8>(V2_ARGS);
      case 16: return v2_launch<3, false, false, 16>(V2_ARGS);
      case 24: return v2_launch<3, false, false, 24>(V2_ARGS);
      case 129: return v2_launch<3, true, false, 1>(V2_ARGS);
      case 136: return v2_launch<3, true, false, 8>(V2_ARGS);
      case 144: return v2_launch<3, true, false, 16>(V2_ARGS);
      case 152: return v2_launch<3, true, false, 24>(V2_ARGS);
      default: return -3;
    }
  }
  if (fmt) {
    if (st == 3) return f8 ? v2_launch<3, true, true>(V2_ARGS) : v2_launch<3, false, true>(V2_ARGS);
    return f8 ? v2_launch<2, true, true>(V2_ARGS) : v2_launch<2, false, true>(V2_ARGS);
  }
  if (st == 3) return f8 ? v2_launch<3, true, false>(V2_ARGS) : v2_launch<3, false, false>(V2_ARGS);
  return f8 ? v2_launch<2, true, false>(V2_ARGS) : v2_launch<2, false, false>(V2_ARGS);
#undef V2_ARGS
}

int oa2_prof(void* host_out, int reset) {   // g_prof -> host (16 u64); reset: zero it
  cudaError_t e = cudaMemcpyFromSymbol(host_out, g_prof, sizeof(g_prof));
  if (e == cudaSuccess && reset) {
    unsigned long long z[16] = {0};
    e = cudaMemcpyToSymbol(g_prof, z, sizeof(z));
  }
  return (int)e;
}

int oa2_mb_launch(int kind, int iters, int blocks, void* out, void* clk, void* stream) {
  mb_kernel<<<blocks, 128, 0, (cudaStream_t)stream>>>(kind, iters, (float*)out, (long long*)clk);
  return (int)cudaGetLastError();
}

// cfg = ST | NW << 4 (NW 4 | 8); abl = ABL bits
int oa2_abl_launch(const void* q, const void* cache, const void* idx, const void* res, long long T, long long R,
                   long long sqt, long long sqh, long long sit, int topk, float sm_scale, void* out, long long sot,
                   long long soh, float kv_scale, int cfg, int abl, void* stream) {
#define ABL_CASE(A)                                                                                                    \
  case A:                                                                                                              \
    return cfg == 0x42 ? abl_launch<2, 4, A>(q, cache, idx, res, T, R, sqt, sqh, sit, topk, sm_scale, out, sot, soh,    \
                                             kv_scale, stream)                                                         \
                       : abl_launch<2, 8, A>(q, cache, idx, res, T, R, sqt, sqh, sit, topk, sm_scale, out, sot, soh,    \
                                             kv_scale, stream);
  switch (abl) {
    ABL_CASE(0)
    ABL_CASE(1)
    ABL_CASE(2)
    ABL_CASE(3)
    ABL_CASE(4)
    ABL_CASE(8)
    ABL_CASE(16)
    ABL_CASE(24)
    ABL_CASE(32)
    ABL_CASE(28)
    default:
      return -1;
  }
#undef ABL_CASE
}

}  // extern "C"
