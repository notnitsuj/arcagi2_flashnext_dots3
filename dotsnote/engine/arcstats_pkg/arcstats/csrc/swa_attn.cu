// Dots3 sliding-window (MLA-latent MQA) decode attention, single pass, all-FP8 tensor cores (sm_89+).
// Timing-only ablations (wrong results; dots3/build_swa_probe.sh): SWA_ABLATE_MMA / _QK / _PV / _TR / _LOAD.
//
// One CTA = one decode token t (request r = t / QL, query j = t % QL): its 16 heads (M = 16) against its own window
// [max(0, qpos - WIN + 1), qpos], streamed in tiles of BN = 32 cache rows (1088 B FP8: 1024 latent + 64 rope) with
// cp.async straight from the paged cache into an XOR-swizzled shared buffer (2 stages). The QL sibling CTAs of a
// request are adjacent in the grid, so their (almost identical) windows are read from DRAM once and hit L2 after.
// NW = 8 or 16 warps. Q.K: warp w owns latent slice ds = w % 8 (dims [128 ds, +128); slices 6, 7 also one 32-dim rope
// chunk each) for n-tiles (keys) 8 / NW * 4 .. of the tile; P.V: warp w owns latent columns [1024 / NW * w, +1024 / NW).
//   Q.K^T: mma.m16n8k32 e4m3 x e4m3 -> f32 with the K bytes as the B operand as they are (no conversion). q is split
//          into hi + lo e4m3 terms per (row, slice) power-of-two scale (QLO: two MMAs, ~bf16 accuracy). Partial S of
//          the 8 slices are summed through shared memory.
//   softmax: 16 rows x 32 keys (online max / sum in fp32, exp2), 16 or 32 lanes per row.
//   P.V:   mma.m16n8k32 e4m3 again: P (scaled by 256) as hi + lo e4m3 (PLO), V bytes as the B operand after a 4x4
//          byte transpose (prmt) of row chunks: no FP8 -> FP16 conversion anywhere.
// Output [T, 16, 1024] bf16 = softmax(q.k * sm_scale) . v * kv_scale. Empty windows (qpos < 0) give zeros.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int H = 16, D = 1088, BN = 32, NDS = 8;   // NDS: 128-dim latent slices (Q.K split)
constexpr int SRS = BN + 8;      // partial-S row stride (floats): conflict-free 8-byte stores
constexpr int PRS = BN + 16;     // P row stride (bytes)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ void mma_fp8(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef SWA_ABLATE_MMA   // timing ablation only: keep operands live, drop the tensor-core work
  c[0] += __uint_as_float((a[0] ^ b0) & 0x3f800000u); c[1] += __uint_as_float((a[1] ^ b1) & 0x3f800000u);
  c[2] += __uint_as_float((a[2] ^ b0) & 0x3f800000u); c[3] += __uint_as_float((a[3] ^ b1) & 0x3f800000u);
  return;
#endif
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void lds128(uint32_t addr, uint32_t* r) {
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void lds64(uint32_t addr, uint32_t& a, uint32_t& b) {
  asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];\n" : "=r"(a), "=r"(b) : "r"(addr));
}

// e4m3 of x (saturating), and the e4m3 residual x - float(hi)
__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ float e4m3f(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}
// 4 floats -> hi word and lo word (byte i = element i)
template <bool LO>
__device__ __forceinline__ void quant4(const float* x, uint32_t& hi, uint32_t& lo) {
  hi = lo = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t h = f2e4m3(x[i]);
    hi |= uint32_t(h) << (8 * i);
    if (LO) lo |= uint32_t(f2e4m3((x[i] - e4m3f(h)) * 16.f)) << (8 * i);   // x16: exact for bf16 x (see QK)
  }
}

// 4x4 byte transpose: out[c] byte j = in[j] byte c
__device__ __forceinline__ void tr4(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t* o) {
  uint32_t x0 = prmt(a0, a1, 0x5140), x1 = prmt(a0, a1, 0x7362);
  uint32_t y0 = prmt(a2, a3, 0x5140), y1 = prmt(a2, a3, 0x7362);
  o[0] = prmt(x0, y0, 0x5410);
  o[1] = prmt(x0, y0, 0x7632);
  o[2] = prmt(x1, y1, 0x5410);
  o[3] = prmt(x1, y1, 0x7632);
}

// shared-memory byte offset of 16-byte chunk c (0..67) of tile row kr
__device__ __forceinline__ int swz(int kr, int c) {
  int p = c < 64 ? ((c & ~7) | ((c ^ kr) & 7)) : (64 + ((c ^ kr) & 3));
  return kr * D + p * 16;
}


constexpr int smem_bytes(int st) { return st * BN * D + NDS * H * SRS * 4 + 2 * H * PRS + 3 * H * 4 + st * BN; }

template <int ST, int NW, bool QLO, bool PLO>
__global__ void __launch_bounds__(NW * 32, 1)
swa_attn_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv, const int* __restrict__ bt,
                const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, long long sq_t, long long sq_h,
                long long so_t, long long so_h, long long blk_stride, int bt_stride, int ql, int win, int page,
                float sm_scale_log2, float kv_scale) {
  constexpr int NT = NW * 32;
  constexpr int NTW = 4 * NDS / NW;        // Q.K n-tiles per warp
  constexpr int CW = 1024 / NW;            // P.V latent columns per warp
  constexpr int NPV = CW / 8;              // P.V n-tiles per warp = columns per thread per B row chunk
  constexpr int KPT = BN * H / NT;         // softmax keys per thread (2 | 1)
  constexpr int LPR = BN / KPT;            // softmax lanes per row (16 | 32)
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;                                                        // [ST][BN][D] swizzled
  float* Sp = reinterpret_cast<float*>(Ks + ST * BN * D);                    // [NDS][H][SRS] partial S
  uint8_t* Ph = reinterpret_cast<uint8_t*>(Sp + NDS * H * SRS);              // [H][PRS] P hi (key-permuted)
  uint8_t* Pl = Ph + H * PRS;                                                // [H][PRS] P lo
  float* alpha = reinterpret_cast<float*>(Pl + H * PRS);                     // [H] rescale of this tile
  float* invl = alpha + H;                                                   // [H] final 1 / l
  uint8_t* vflag = reinterpret_cast<uint8_t*>(invl + H);                     // [ST][BN] row has a valid page

  const int t = blockIdx.x;                  // decode token; blockIdx.y = 16-head group (H = 16 per rank at TP4)
  const int r = t / ql, jq = t % ql;
  q += (long long)blockIdx.y * H * sq_h;
  out += (long long)blockIdx.y * H * so_h;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int ds = w % NDS, nt0 = (w / NDS) * NTW;
  const int qpos = seq_lens[r] - ql + jq;
  const int start = max(qpos - win + 1, 0);
  const int nk = qpos >= 0 ? qpos - start + 1 : 0;
  const int ntiles = (nk + BN - 1) / BN;
  const int* btr = bt + (long long)r * bt_stride;

  // ---- loader: BN rows x 68 chunks of 16 B over NT threads
  constexpr int TPR = NT / BN;                       // threads per row (8 | 16)
  constexpr int CPT = (D / 16 + TPR - 1) / TPR;      // chunks per thread
  const int lrow = tid / TPR, lsub = tid % TPR;
  auto issue = [&](int tile) {
    if (tile < ntiles) {
      int k = tile * BN + lrow;
      bool v = k < nk;
      const uint8_t* src = kv;
      if (v) {
        int pos = start + k;
        int pg = btr[pos / page];
        v = pg >= 0;
        if (v) src = kv + (long long)pg * blk_stride + (long long)(pos % page) * D;
      }
      if (lsub == 0) vflag[(tile % ST) * BN + lrow] = v;
      const uint32_t dst = smem_u32(Ks + (tile % ST) * BN * D);
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        int c = lsub + TPR * i;
#ifndef SWA_ABLATE_LOAD
        if (c < D / 16) cp_async16(dst + swz(lrow, c), src + c * 16, v);
#endif
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) issue(s);

  // ---- q fragments: rows g, g + 8; latent words i = 0..7 <-> dims 128 ds + 64 (i >> 2) + 16 tq + 4 (i & 3) + 0..3;
  //      rope (slices 6, 7, chunk rc = ds - 6): words 8, 9 <-> dims 1024 + 32 rc + 8 tq + 4 (i - 8) + 0..3
  const bool rope = ds >= NDS - 2;
  const int rc = ds - (NDS - 2);
  uint32_t qa[5][4], qlw[5][4];
  float qinv[2];
  {
    float qv[2][40];
    float amax[2] = {0.f, 0.f};
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const __nv_bfloat16* qr = q + (long long)t * sq_t + (g + 8 * h2) * sq_h;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4* src = reinterpret_cast<const uint4*>(qr + 128 * ds + 64 * half + 16 * tq);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint4 x = src[u];
          const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float2 f = __bfloat1622float2(b[e]);
            qv[h2][16 * half + 8 * u + 2 * e] = f.x;
            qv[h2][16 * half + 8 * u + 2 * e + 1] = f.y;
          }
        }
      }
      if (rope) {
        uint4 x = *reinterpret_cast<const uint4*>(qr + 1024 + 32 * rc + 8 * tq);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 f = __bfloat1622float2(b[e]);
          qv[h2][32 + 2 * e] = f.x;
          qv[h2][32 + 2 * e + 1] = f.y;
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) qv[h2][32 + e] = 0.f;
      }
#pragma unroll
      for (int e = 0; e < 40; ++e) amax[h2] = fmaxf(amax[h2], fabsf(qv[h2][e]));
      amax[h2] = fmaxf(amax[h2], __shfl_xor_sync(0xffffffff, amax[h2], 1));
      amax[h2] = fmaxf(amax[h2], __shfl_xor_sync(0xffffffff, amax[h2], 2));
    }
    // power-of-two scale per (row, slice): amax -> (224, 448]
    float qsc[2];
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      int ex = amax[h2] > 0.f ? (int)floorf(log2f(448.f / amax[h2])) : 0;
      ex = max(min(ex, 100), -100);
      qsc[h2] = exp2f((float)ex);
      qinv[h2] = exp2f((float)-ex);
    }
    // A fragments: chunk m (0..3) latent, 4 = rope. a0 = row g word 2m, a1 = row g+8 word 2m, a2/a3 = word 2m+1
#pragma unroll
    for (int m = 0; m < 5; ++m)
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        int h2 = x & 1, word = 2 * m + (x >> 1);
        float f[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) f[e] = qv[h2][4 * word + e] * qsc[h2];
        quant4<QLO>(f, qa[m][x], qlw[m][x]);
      }
  }

  float o[NPV][4];
#pragma unroll
  for (int n = 0; n < NPV; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  // softmax role: row sr = tid / LPR, keys sk .. sk + KPT - 1
  const int sr = tid / LPR, sk = KPT * (tid % LPR);
  float m_run = -1.0e30f, l_run = 0.f;

  for (int it = 0; it < ntiles; ++it) {
    cp_wait<ST - 2>();
    __syncthreads();
    issue(it + ST - 1);
    const uint32_t kb = smem_u32(Ks + (it % ST) * BN * D);
    const int kleft = nk - it * BN;            // valid keys in this tile (> 0)

    // ---- partial S (16 x 8 NTW) over this warp's slice
#pragma unroll
    for (int i = 0; i < NTW; ++i) {
      const int nt = nt0 + i;
      float s[4] = {0.f, 0.f, 0.f, 0.f}, sl[4] = {0.f, 0.f, 0.f, 0.f};
      if (nt * 8 < kleft) {                    // warp-uniform: skip n-tiles past the window end
        const int key = nt * 8 + g;
        uint32_t kw[8];
        lds128(kb + swz(key, 8 * ds + tq), kw);
        lds128(kb + swz(key, 8 * ds + 4 + tq), kw + 4);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#ifndef SWA_ABLATE_QK
          mma_fp8(s, qa[m], kw[2 * m], kw[2 * m + 1]);
          if (QLO) mma_fp8(sl, qlw[m], kw[2 * m], kw[2 * m + 1]);
#else
          s[0] += __uint_as_float((qa[m][0] ^ kw[2 * m]) & 0x3f800000u);
#endif
        }
        if (rope) {
          uint32_t r0, r1;
          lds64(kb + swz(key, 64 + 2 * rc + (tq >> 1)) + 8 * (tq & 1), r0, r1);
#ifndef SWA_ABLATE_QK
          mma_fp8(s, qa[4], r0, r1);
          if (QLO) mma_fp8(sl, qlw[4], r0, r1);
#else
          s[1] += __uint_as_float((qa[4][0] ^ r0 ^ r1) & 0x3f800000u);
#endif
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) s[e] += sl[e] * 0.0625f;   // q = hi + lo / 16
      float* sp = Sp + (ds * H + g) * SRS + nt * 8 + 2 * tq;
      *reinterpret_cast<float2*>(sp) = make_float2(s[0] * qinv[0], s[1] * qinv[0]);
      *reinterpret_cast<float2*>(sp + 8 * SRS) = make_float2(s[2] * qinv[1], s[3] * qinv[1]);
    }
    __syncthreads();

    // ---- softmax over the 32 keys of this tile
    {
      float x[KPT];
      if constexpr (KPT == 2) {
        float2 acc = make_float2(0.f, 0.f);
#pragma unroll
        for (int d = 0; d < NDS; ++d) {
          float2 v = *reinterpret_cast<const float2*>(Sp + (d * H + sr) * SRS + sk);
          acc.x += v.x;
          acc.y += v.y;
        }
        x[0] = acc.x;
        x[1] = acc.y;
      } else {
        float acc = 0.f;
#pragma unroll
        for (int d = 0; d < NDS; ++d) acc += Sp[(d * H + sr) * SRS + sk];
        x[0] = acc;
      }
      const uint8_t* vf = vflag + (it % ST) * BN;
      bool kv_[KPT];
      float mx = -1.0e30f;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        kv_[i] = sk + i < kleft && vf[sk + i];
        x[i] = kv_[i] ? x[i] * sm_scale_log2 : -1.0e30f;
        mx = fmaxf(mx, x[i]);
      }
#pragma unroll
      for (int o_ = 1; o_ < LPR; o_ <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o_));
      const float mn = fmaxf(m_run, mx);
      const float al = exp2f(m_run - mn);
      m_run = mn;
      float ps = 0.f;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        const int k = sk + i;
        const float pf = kv_[i] ? exp2f(x[i] - mn) : 0.f;
        // P.V uses p rounded to bf16 (as the Triton path), split exactly into e4m3 hi + lo for p >= ~1e-3; l = sum of
        // the fp32 p
        const float p = __bfloat162float(__float2bfloat16_rn(pf)) * 256.f;
        uint8_t hb = f2e4m3(p);
        float qp = e4m3f(hb);
        // key k -> byte (k & 3) * 8 + (k >> 2) of the permuted row (thread tq reads keys tq + 4 j, j = 0..7)
        const int b = (k & 3) * 8 + (k >> 2);
        Ph[sr * PRS + b] = hb;
        if (PLO) {
          Pl[sr * PRS + b] = f2e4m3(p - qp);
        }
        ps += pf * 256.f;
      }
#pragma unroll
      for (int o_ = 1; o_ < LPR; o_ <<= 1) ps += __shfl_xor_sync(0xffffffff, ps, o_);
      l_run = l_run * al + ps;
      if (tid % LPR == 0) alpha[sr] = al;
    }
    __syncthreads();

    // ---- O (16 x [CW w, +CW)) = O * alpha + P V
    {
      const float a0 = alpha[g], a1 = alpha[g + 8];
#pragma unroll
      for (int n = 0; n < NPV; ++n) {
        o[n][0] *= a0;
        o[n][1] *= a0;
        o[n][2] *= a1;
        o[n][3] *= a1;
      }
      uint32_t pa[4], pl[4];
      lds64(smem_u32(Ph + g * PRS + 8 * tq), pa[0], pa[2]);
      lds64(smem_u32(Ph + (g + 8) * PRS + 8 * tq), pa[1], pa[3]);
      if (PLO) {
        lds64(smem_u32(Pl + g * PRS + 8 * tq), pl[0], pl[2]);
        lds64(smem_u32(Pl + (g + 8) * PRS + 8 * tq), pl[1], pl[3]);
      }
      // B: keys tq + 4 j (j 0..3 -> b0, 4..7 -> b1); columns CW w + NPV g + c, c < NPV (n-tile c, n = g)
      const int col = CW * w + NPV * g;
      uint32_t bb[2][NPV];
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        uint32_t vw[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int key = tq + 4 * (4 * hb + j);
          if constexpr (NPV == 16) lds128(kb + swz(key, col >> 4), vw[j]);
          else lds64(kb + swz(key, col >> 4) + (col & 15), vw[j][0], vw[j][1]);
        }
#pragma unroll
        for (int u = 0; u < NPV / 4; ++u) {
#ifndef SWA_ABLATE_TR
          tr4(vw[0][u], vw[1][u], vw[2][u], vw[3][u], &bb[hb][4 * u]);
#else
          bb[hb][4 * u] = vw[0][u]; bb[hb][4 * u + 1] = vw[1][u]; bb[hb][4 * u + 2] = vw[2][u]; bb[hb][4 * u + 3] = vw[3][u];
#endif
        }
      }
#pragma unroll
      for (int c = 0; c < NPV; ++c) {
#ifndef SWA_ABLATE_PV
        mma_fp8(o[c], pa, bb[0][c], bb[1][c]);
        if (PLO) mma_fp8(o[c], pl, bb[0][c], bb[1][c]);
#else
        o[c][0] += __uint_as_float((pa[0] ^ bb[0][c] ^ bb[1][c]) & 0x3f800000u);
#endif
      }
    }
  }
  cp_wait<0>();

  if (tid % LPR == 0) invl[sr] = l_run > 0.f ? kv_scale / l_run : 0.f;
  __syncthreads();
  // thread (g, tq): n-tile c holds columns CW w + 2 NPV tq + c (c0, c2) and CW w + 2 NPV tq + NPV + c (c1, c3)
#pragma unroll
  for (int h2 = 0; h2 < 2; ++h2) {
    const float iv = invl[g + 8 * h2];
    __nv_bfloat16* orow = out + (long long)t * so_t + (g + 8 * h2) * so_h + CW * w + 2 * NPV * tq;
    uint32_t pk[NPV];
#pragma unroll
    for (int c = 0; c < NPV; c += 2) {
      __nv_bfloat162 lo = __floats2bfloat162_rn(o[c][2 * h2] * iv, o[c + 1][2 * h2] * iv);
      __nv_bfloat162 hi = __floats2bfloat162_rn(o[c][2 * h2 + 1] * iv, o[c + 1][2 * h2 + 1] * iv);
      pk[c / 2] = *reinterpret_cast<uint32_t*>(&lo);
      pk[NPV / 2 + c / 2] = *reinterpret_cast<uint32_t*>(&hi);
    }
    uint4* dst = reinterpret_cast<uint4*>(orow);
#pragma unroll
    for (int u = 0; u < NPV / 4; ++u) dst[u] = make_uint4(pk[4 * u], pk[4 * u + 1], pk[4 * u + 2], pk[4 * u + 3]);
  }
}

template <int NW, bool QLO, bool PLO>
int launch(const void* q, const void* kv, const void* bt, const void* sl, void* out, long long sq_t, long long sq_h,
           long long so_t, long long so_h, long long blk_stride, int bt_stride, int T, int ql, int win, int page,
           float sm_scale_log2, float kv_scale, cudaStream_t st, int hgroups) {
  constexpr int sm = smem_bytes(2);
  auto k = swa_attn_kernel<2, NW, QLO, PLO>;
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) != cudaSuccess) return 2;
  k<<<dim3(T, hgroups), NW * 32, sm, st>>>(static_cast<const __nv_bfloat16*>(q), static_cast<const uint8_t*>(kv),
                            static_cast<const int*>(bt), static_cast<const int*>(sl),
                            static_cast<__nv_bfloat16*>(out), sq_t, sq_h, so_t, so_h, blk_stride, bt_stride, ql, win,
                            page, sm_scale_log2, kv_scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}


// ---------------------------------------------------------------------------------------------------------------------
// Token-pair packed variant (cfg 21..24): one CTA = 2 consecutive MTP tokens of a request x 16 heads (M = 32 rows),
// 16 warps. The pair shares one stream of 32-key tiles over the union of its two windows (each row masks keys outside
// its own window), so every K/V tile is loaded and transposed once for 32 rows instead of 16, and the CTA count halves.
// L4 budget: O (32 x 1024 fp32) = 64 regs/thread at 512 threads; smem = 2 x 34.8 KB tiles + 4 x 32-row partial S.
// q is e4m3 hi only with one power-of-two scale per (row, 256-dim slice pair) (as cfg 3: QLO is not supported here);
// PLO selects P hi/lo (cfg 21) or P hi (cfg 23). Q.K: warp w -> token mt = w / 8, slice pair sp = (w / 2) % 4
// (latent dims [256 sp, +256); sp 3 also both rope chunks), n-tiles 2 (w % 2) .. +1; partial S summed over sp.
constexpr int PK_M = 32, PK_NSP = 4;
constexpr int pk_smem_bytes() { return 2 * BN * D + PK_NSP * PK_M * SRS * 4 + 2 * PK_M * PRS + 3 * PK_M * 4 + 2 * BN; }

template <int NW, bool PLO>
__global__ void __launch_bounds__(NW * 32, 1)
swa_pk2_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv, const int* __restrict__ bt,
               const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, long long sq_t, long long sq_h,
               long long so_t, long long so_h, long long blk_stride, int bt_stride, int ql, int win, int page,
               float sm_scale_log2, float kv_scale) {
  constexpr int ST = 2, NT = NW * 32, CW = 1024 / NW, NPV = CW / 8, KPT = BN * PK_M / NT, LPR = BN / KPT;
  constexpr int NTW = 4 * 8 / NW;            // Q.K n-tiles per warp (4 | 2)
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;                                                        // [ST][BN][D] swizzled
  float* Sp = reinterpret_cast<float*>(Ks + ST * BN * D);                    // [NSP][M][SRS] partial S
  uint8_t* Ph = reinterpret_cast<uint8_t*>(Sp + PK_NSP * PK_M * SRS);        // [M][PRS]
  uint8_t* Pl = Ph + PK_M * PRS;
  float* alpha = reinterpret_cast<float*>(Pl + PK_M * PRS);                  // [M]
  float* invl = alpha + PK_M;                                                // [M]
  uint8_t* vflag = reinterpret_cast<uint8_t*>(invl + PK_M);                  // [ST][BN]

  const int t0 = 2 * blockIdx.x;             // first token of the pair
  const int r = t0 / ql, j0 = t0 % ql;
  q += (long long)blockIdx.y * H * sq_h;
  out += (long long)blockIdx.y * H * so_h;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int qp0 = seq_lens[r] - ql + j0, qp1 = qp0 + 1;
  const int st0 = max(qp0 - win + 1, 0), st1 = max(qp1 - win + 1, 0);
  const int start = st0;                                        // union window [st0, qp1]
  const int nk = qp1 >= 0 ? qp1 - start + 1 : 0;
  const int ntiles = (nk + BN - 1) / BN;
  const int* btr = bt + (long long)r * bt_stride;

  constexpr int TPR = NT / BN;                       // threads per row (8 | 16)
  constexpr int CPT = (D / 16 + TPR - 1) / TPR;      // chunks per thread
  const int lrow = tid / TPR, lsub = tid % TPR;
  auto issue = [&](int tile) {
    if (tile < ntiles) {
      int k = tile * BN + lrow;
      bool v = k < nk;
      const uint8_t* src = kv;
      if (v) {
        int pos = start + k;
        int pg = btr[pos / page];
        v = pg >= 0;
        if (v) src = kv + (long long)pg * blk_stride + (long long)(pos % page) * D;
      }
      if (lsub == 0) vflag[(tile % ST) * BN + lrow] = v;
      const uint32_t dst = smem_u32(Ks + (tile % ST) * BN * D);
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        int c = lsub + TPR * i;
        if (c < D / 16) cp_async16(dst + swz(lrow, c), src + c * 16, v);
      }
    }
    cp_commit();
  };
  issue(0);

  // ---- q fragments of this warp: token mt, slices ds = 2 sp + ss (ss = 0, 1); chunk 4 = rope (slices 6, 7 only)
  // Q.K roles: NW 16 -> token w / 8, slice pair (w / 2) % 4, n-tiles 2 (w % 2) + 0..1; NW 8 -> token w / 4, pair w % 4
  const int mt = NW == 16 ? (w >> 3) : (w >> 2), sp = NW == 16 ? ((w >> 1) & 3) : (w & 3);
  const int nh = NW == 16 ? (w & 1) : 0;
  const int tq_tok = t0 + mt;
  uint32_t qa[2][5][4];
  float qinv[2];
  {
    // q element e (0..39) of (slice ds, head row h2): latent words as the 8-warp kernel, e >= 32: rope (slices 6, 7)
    auto qload = [&](int ds, int h2, int word, float* f) {   // 4 consecutive elements of fragment word
      const __nv_bfloat16* qr = q + (long long)tq_tok * sq_t + (g + 8 * h2) * sq_h;
      const __nv_bfloat16* src;
      if (word < 8) src = qr + 128 * ds + 64 * (word >> 2) + 16 * tq + 4 * (word & 3);
      else if (ds >= NDS - 2) src = qr + 1024 + 32 * (ds - (NDS - 2)) + 8 * tq + 4 * (word - 8);
      else { f[0] = f[1] = f[2] = f[3] = 0.f; return; }
      uint2 x = *reinterpret_cast<const uint2*>(src);
      const __nv_bfloat162* bq = reinterpret_cast<const __nv_bfloat162*>(&x);
      float2 a0 = __bfloat1622float2(bq[0]), a1 = __bfloat1622float2(bq[1]);
      f[0] = a0.x; f[1] = a0.y; f[2] = a1.x; f[3] = a1.y;
    };
    float amax[2] = {0.f, 0.f};
#pragma unroll
    for (int ss = 0; ss < 2; ++ss)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
        for (int word = 0; word < 10; ++word) {
          float f[4];
          qload(2 * sp + ss, h2, word, f);
#pragma unroll
          for (int e = 0; e < 4; ++e) amax[h2] = fmaxf(amax[h2], fabsf(f[e]));
        }
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      amax[h2] = fmaxf(amax[h2], __shfl_xor_sync(0xffffffff, amax[h2], 1));
      amax[h2] = fmaxf(amax[h2], __shfl_xor_sync(0xffffffff, amax[h2], 2));
    }
    float qsc[2];
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      int ex = amax[h2] > 0.f ? (int)floorf(log2f(448.f / amax[h2])) : 0;
      ex = max(min(ex, 100), -100);
      qsc[h2] = exp2f((float)ex);
      qinv[h2] = exp2f((float)-ex);
    }
#pragma unroll
    for (int ss = 0; ss < 2; ++ss)
#pragma unroll
      for (int m = 0; m < 5; ++m)
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          int h2 = x & 1, word = 2 * m + (x >> 1);
          float f[4];
          qload(2 * sp + ss, h2, word, f);
#pragma unroll
          for (int e = 0; e < 4; ++e) f[e] *= qsc[h2];
          uint32_t lo;
          quant4<false>(f, qa[ss][m][x], lo);
        }
  }

  float o[2][NPV][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int n = 0; n < NPV; ++n) o[a][n][0] = o[a][n][1] = o[a][n][2] = o[a][n][3] = 0.f;
  // softmax role: row sr (token sr / 16, head sr % 16), keys sk, sk + 1
  const int sr = tid / LPR, sk = KPT * (tid % LPR);
  const int my_qp = (sr >> 4) ? qp1 : qp0, my_st = (sr >> 4) ? st1 : st0;
  float m_run = -1.0e30f, l_run = 0.f;

  for (int it = 0; it < ntiles; ++it) {
    cp_wait<0>();
    __syncthreads();
    issue(it + 1);
    const uint32_t kb = smem_u32(Ks + (it % ST) * BN * D);
    const int kleft = nk - it * BN;

    // ---- partial S over this warp's slice pair for token mt, n-tiles 2 nh, 2 nh + 1
#pragma unroll
    for (int i = 0; i < NTW; ++i) {
      const int nt = NTW * nh + i;
      float s[4] = {0.f, 0.f, 0.f, 0.f};
      if (nt * 8 < kleft) {
        const int key = nt * 8 + g;
#pragma unroll
        for (int ss = 0; ss < 2; ++ss) {
          const int ds = 2 * sp + ss;
          uint32_t kw[8];
          lds128(kb + swz(key, 8 * ds + tq), kw);
          lds128(kb + swz(key, 8 * ds + 4 + tq), kw + 4);
#pragma unroll
          for (int m = 0; m < 4; ++m) mma_fp8(s, qa[ss][m], kw[2 * m], kw[2 * m + 1]);
          if (ds >= NDS - 2) {
            const int rc = ds - (NDS - 2);
            uint32_t r0, r1;
            lds64(kb + swz(key, 64 + 2 * rc + (tq >> 1)) + 8 * (tq & 1), r0, r1);
            mma_fp8(s, qa[ss][4], r0, r1);
          }
        }
      }
      float* spp = Sp + (sp * PK_M + 16 * mt + g) * SRS + nt * 8 + 2 * tq;
      *reinterpret_cast<float2*>(spp) = make_float2(s[0] * qinv[0], s[1] * qinv[0]);
      *reinterpret_cast<float2*>(spp + 8 * SRS) = make_float2(s[2] * qinv[1], s[3] * qinv[1]);
    }
    __syncthreads();

    // ---- softmax over the 32 keys of this tile (each row masks keys outside its own window)
    {
      float x[KPT];
#pragma unroll
      for (int i = 0; i < KPT; ++i) x[i] = 0.f;
#pragma unroll
      for (int d = 0; d < PK_NSP; ++d)
#pragma unroll
        for (int i = 0; i < KPT; ++i) x[i] += Sp[(d * PK_M + sr) * SRS + sk + i];
      const uint8_t* vf = vflag + (it % ST) * BN;
      bool kv_[KPT];
      float mx = -1.0e30f;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        const int pos = start + it * BN + sk + i;
        kv_[i] = sk + i < kleft && vf[sk + i] && pos >= my_st && pos <= my_qp;
        x[i] = kv_[i] ? x[i] * sm_scale_log2 : -1.0e30f;
        mx = fmaxf(mx, x[i]);
      }
#pragma unroll
      for (int o_ = 1; o_ < LPR; o_ <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o_));
      const float mn = fmaxf(m_run, mx);
      const float al = exp2f(m_run - mn);
      m_run = mn;
      float ps = 0.f;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        const int k = sk + i;
        const float pf = kv_[i] ? exp2f(x[i] - mn) : 0.f;
        const float p = __bfloat162float(__float2bfloat16_rn(pf)) * 256.f;
        uint8_t hb = f2e4m3(p);
        const int b = (k & 3) * 8 + (k >> 2);
        Ph[sr * PRS + b] = hb;
        if (PLO) Pl[sr * PRS + b] = f2e4m3(p - e4m3f(hb));
        ps += pf * 256.f;
      }
#pragma unroll
      for (int o_ = 1; o_ < LPR; o_ <<= 1) ps += __shfl_xor_sync(0xffffffff, ps, o_);
      l_run = l_run * al + ps;
      if (tid % LPR == 0) alpha[sr] = al;
    }
    __syncthreads();

    // ---- O (32 x [64 w, +64)) = O * alpha + P V; V fragments shared by both tokens
    {
      const int col = CW * w + NPV * g;
      uint32_t bb[2][NPV];
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        uint32_t vw[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int key = tq + 4 * (4 * hb + j);
          if constexpr (NPV == 16) lds128(kb + swz(key, col >> 4), vw[j]);
          else lds64(kb + swz(key, col >> 4) + (col & 15), vw[j][0], vw[j][1]);
        }
#pragma unroll
        for (int u = 0; u < NPV / 4; ++u) tr4(vw[0][u], vw[1][u], vw[2][u], vw[3][u], &bb[hb][4 * u]);
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const float a0 = alpha[16 * a + g], a1 = alpha[16 * a + g + 8];
        uint32_t pa[4], pl[4];
        lds64(smem_u32(Ph + (16 * a + g) * PRS + 8 * tq), pa[0], pa[2]);
        lds64(smem_u32(Ph + (16 * a + g + 8) * PRS + 8 * tq), pa[1], pa[3]);
        if (PLO) {
          lds64(smem_u32(Pl + (16 * a + g) * PRS + 8 * tq), pl[0], pl[2]);
          lds64(smem_u32(Pl + (16 * a + g + 8) * PRS + 8 * tq), pl[1], pl[3]);
        }
#pragma unroll
        for (int c = 0; c < NPV; ++c) {
          o[a][c][0] *= a0;
          o[a][c][1] *= a0;
          o[a][c][2] *= a1;
          o[a][c][3] *= a1;
          mma_fp8(o[a][c], pa, bb[0][c], bb[1][c]);
          if (PLO) mma_fp8(o[a][c], pl, bb[0][c], bb[1][c]);
        }
      }
    }
  }
  cp_wait<0>();

  if (tid % LPR == 0) invl[sr] = l_run > 0.f ? kv_scale / l_run : 0.f;
  __syncthreads();
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const float iv = invl[16 * a + g + 8 * h2];
      __nv_bfloat16* orow = out + (long long)(t0 + a) * so_t + (g + 8 * h2) * so_h + CW * w + 2 * NPV * tq;
      uint32_t pk[NPV];
#pragma unroll
      for (int c = 0; c < NPV; c += 2) {
        __nv_bfloat162 lo = __floats2bfloat162_rn(o[a][c][2 * h2] * iv, o[a][c + 1][2 * h2] * iv);
        __nv_bfloat162 hi = __floats2bfloat162_rn(o[a][c][2 * h2 + 1] * iv, o[a][c + 1][2 * h2 + 1] * iv);
        pk[c / 2] = *reinterpret_cast<uint32_t*>(&lo);
        pk[NPV / 2 + c / 2] = *reinterpret_cast<uint32_t*>(&hi);
      }
      uint4* dst = reinterpret_cast<uint4*>(orow);
#pragma unroll
      for (int u = 0; u < NPV / 4; ++u) dst[u] = make_uint4(pk[4 * u], pk[4 * u + 1], pk[4 * u + 2], pk[4 * u + 3]);
    }
}

template <int NW, bool PLO>
int launch_pk2(const void* q, const void* kv, const void* bt, const void* sl, void* out, long long sq_t, long long sq_h,
               long long so_t, long long so_h, long long blk_stride, int bt_stride, int T, int ql, int win, int page,
               float sm_scale_log2, float kv_scale, cudaStream_t st, int hgroups) {
  if (ql % 2 || T % 2) return 5;
  constexpr int sm = pk_smem_bytes();
  auto k = swa_pk2_kernel<NW, PLO>;
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) != cudaSuccess) return 2;
  k<<<dim3(T / 2, hgroups), NW * 32, sm, st>>>(static_cast<const __nv_bfloat16*>(q), static_cast<const uint8_t*>(kv),
                            static_cast<const int*>(bt), static_cast<const int*>(sl),
                            static_cast<__nv_bfloat16*>(out), sq_t, sq_h, so_t, so_h, blk_stride, bt_stride, ql, win,
                            page, sm_scale_log2, kv_scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

}  // namespace

// cfg (2 shared-memory stages; 3 would need ~126 KB > sm_89's 99 KB per CTA):
//   1 = 8 warps, q hi/lo + P hi/lo (~bf16 accuracy)      11 = same with 16 warps (64 latent columns per warp in P.V)
//   2 = 8 warps, q hi/lo + P hi only                      12 = 16 warps
//   3 = 8 warps, q hi + P hi only                         13 = 16 warps
//   4 = 8 warps, q hi only + P hi/lo                      14 = 16 warps
//  21 = token-pair packed (2 MTP tokens x 16 heads per CTA, 16 warps), q hi + P hi/lo; 23 = q hi + P hi; 31 / 33 = 8 warps
extern "C" int swa_attn_launch(const void* q, const void* kv, const void* bt, const void* sl, void* out,
                               long long sq_t, long long sq_h, long long so_t, long long so_h, long long blk_stride,
                               int bt_stride, int T, int H_, int D_, int ql, int win, int page, float sm_scale_log2,
                               float kv_scale, int cfg, void* stream) {
  if (H_ % H || H_ <= 0 || D_ != D || T <= 0) return 1;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
#define ARGS q, kv, bt, sl, out, sq_t, sq_h, so_t, so_h, blk_stride, bt_stride, T, ql, win, page, sm_scale_log2, kv_scale, st, H_ / H
  switch (cfg) {
    case 1: return launch<8, true, true>(ARGS);
    case 2: return launch<8, true, false>(ARGS);
    case 3: return launch<8, false, false>(ARGS);
    case 4: return launch<8, false, true>(ARGS);
    case 11: return launch<16, true, true>(ARGS);
    case 12: return launch<16, true, false>(ARGS);
    case 13: return launch<16, false, false>(ARGS);
    case 14: return launch<16, false, true>(ARGS);
    // token-pair packed (odd query length, e.g. the MTP draft steps: the single-token kernel with the same numerics)
    case 21: return ql % 2 ? launch<8, false, true>(ARGS) : launch_pk2<16, true>(ARGS);
    case 23: return ql % 2 ? launch<8, false, false>(ARGS) : launch_pk2<16, false>(ARGS);
    case 31: return ql % 2 ? launch<8, false, true>(ARGS) : launch_pk2<8, true>(ARGS);
    case 33: return ql % 2 ? launch<8, false, false>(ARGS) : launch_pk2<8, false>(ARGS);
  }
#undef ARGS
  return 4;
}

extern "C" int swa_attn_smem() { return smem_bytes(2); }
