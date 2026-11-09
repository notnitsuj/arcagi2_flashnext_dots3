// CPU-reduced host all-reduce (arcstats/host_ar.py, ARC_HOST_AR_CPU=1): host threads sum the ranks' quantized
// contributions in shared pinned host memory and publish the requantized result, so each GPU reads 1 S instead of the
// two-shot kernel's 1.5 S (S = quantized payload; the binding PCIe direction on Kaggle's pair-shared Gen3 uplinks).
// Build: dots3/build_host_arc.sh -> host_arc.so (AVX-512 path picked at run time, portable scalar fallback).
//
// Segment layout (byte offsets from `base`, all set by the Python side):
//   GF  gpu -> cpu flags, one 64-byte line per (rank, block, sub-chunk): word 0 epoch, word 1 q (quarter length in
//       elements), word 2 B | RB << 8 | OWNQ << 16
//   CF  cpu -> gpu flags, one line per (block, sub-chunk): word 0 epoch
//   IN  per rank, `slot` bytes: the rank's quantized contributions at global element positions p. B = 8: int8 codes at
//       byte p. B = 5/6 ("vertical" packing, G = 128): per group of 128 elements 16 int32 words at byte 4 (p / 8) in the
//       nibble plane, word w holding the low nibble of element 16 e + w at bits [4e, 4e + 4); 16 int16 (B = 6) / int8
//       (B = 5) words at maxn / 2 + (B - 4) (p / 8) holding the high bits of element 16 e + w at bits [(B-4) e, ..).
//       fp32 per-G scales at byte maxn + 4 (p / G)
//   OWN per rank, own_slot bytes: the rank's own quarter as bf16 (OWNQ = 0), at quarter offsets
//   RED one slot (same planes, RB bits): the reduced vector at global positions
// Work unit = (block b, sub-chunk i): for every owner j the elements [j q + c0, j q + c1) with the GPU kernel's slices
// (per = ceil(ceil(q / NB) / G) G, sub = ceil(ceil(per / K) / G) G). Math per element (owner j):
//   acc = own_j (bf16 -> fp32; OWNQ: code * scale), then for k = 0 .. W-1, k != j: acc = fma(code_k, scale_k, acc)
//   scale = max(max |acc| over the G group, 1e-30) / (2^(RB-1) - 1); code = round-half-away(acc * (1 / scale))
// = _host_ar_qb_kernel's owner math up to the division (Triton emits div.full.f32 there): rare 1-code differences.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#if defined(__x86_64__)
#include <immintrin.h>
#define PAUSE() _mm_pause()
#else
#define PAUSE() do {} while (0)
#endif

#define GMAX 512
#define TMAX 128

typedef struct {
    uint8_t *base;
    int W, NB, K, G, FS, nt, simd;
    int64_t maxn, slot, own_slot, in_off, own_off, red_off, gf_off, cf_off;
    volatile int stop;
    int64_t units[TMAX * 8];      // per-thread processed units (stats; padded to separate cache lines)
    int64_t spins[TMAX * 8];      // per-thread spin iterations waiting for the GPUs (stats)
    int cpus[TMAX];               // optional pinning (-1 = none)
    pthread_t th[TMAX];
    struct targ { void *c; int t; } args[TMAX];
    int started;
} ctx_t;

static inline volatile int32_t *gf(ctx_t *c, int r, int b, int i) {
    return (volatile int32_t *)(c->base + c->gf_off) + ((int64_t)(r * c->NB + b) * c->K + i) * c->FS;
}
static inline volatile int32_t *cf(ctx_t *c, int b, int i) {
    return (volatile int32_t *)(c->base + c->cf_off) + ((int64_t)b * c->K + i) * c->FS;
}

// ---------------------------------------------------------------- scalar reference (G a multiple of 128)
static inline void unpack(const uint8_t *s, int64_t maxn, int B, int64_t p, int G, int32_t *restrict out) {
    if (B == 8) {
        const int8_t *q = (const int8_t *)s + p;
        for (int e = 0; e < G; e++) out[e] = q[e];
        return;
    }
    const int hb = B - 4, off = 1 << (B - 1);
    for (int g = 0; g < G; g += 128) {
        const uint32_t *lo = (const uint32_t *)s + (p + g) / 8;
        for (int w = 0; w < 16; w++) {
            uint32_t L = lo[w];
            uint32_t H = B == 6 ? ((const uint16_t *)(s + maxn / 2))[(p + g) / 8 + w]
                                : ((const uint8_t *)(s + maxn / 2))[(p + g) / 8 + w];
            for (int e = 0; e < 8; e++)
                out[g + 16 * e + w] = (int32_t)(((L >> (4 * e)) & 15u) | (((H >> (hb * e)) & ((1u << hb) - 1)) << 4)) - off;
        }
    }
}

static inline void pack(uint8_t *s, int64_t maxn, int B, int64_t p, int G, const int32_t *restrict code) {
    if (B == 8) {
        int8_t *q = (int8_t *)s + p;
        for (int e = 0; e < G; e++) q[e] = (int8_t)code[e];
        return;
    }
    const int hb = B - 4, off = 1 << (B - 1);
    for (int g = 0; g < G; g += 128) {
        for (int w = 0; w < 16; w++) {
            uint32_t L = 0, H = 0;
            for (int e = 0; e < 8; e++) {
                uint32_t u = (uint32_t)(code[g + 16 * e + w] + off);
                L |= (u & 15u) << (4 * e);
                H |= ((u >> 4) & ((1u << hb) - 1)) << (hb * e);
            }
            ((uint32_t *)s)[(p + g) / 8 + w] = L;
            if (B == 6) ((uint16_t *)(s + maxn / 2))[(p + g) / 8 + w] = (uint16_t)H;
            else ((uint8_t *)(s + maxn / 2))[(p + g) / 8 + w] = (uint8_t)H;
        }
    }
}

static inline float bf16f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void group_scalar(ctx_t *c, int j, int64_t q, int B, int RB, int ownq, int64_t o) {
    const int G = c->G, W = c->W;
    float acc[GMAX];
    int32_t cd[GMAX];
    int64_t p = (int64_t)j * q + o;
    if (ownq) {
        const uint8_t *s = c->base + c->in_off + (int64_t)j * c->slot;
        unpack(s, c->maxn, B, p, G, cd);
        float sc = ((const float *)(s + c->maxn))[p / G];
        for (int e = 0; e < G; e++) acc[e] = (float)cd[e] * sc;
    } else {
        const uint16_t *ow = (const uint16_t *)(c->base + c->own_off + (int64_t)j * c->own_slot) + o;
        for (int e = 0; e < G; e++) acc[e] = bf16f(ow[e]);
    }
    for (int k = 0; k < W; k++) {
        if (k == j) continue;
        const uint8_t *s = c->base + c->in_off + (int64_t)k * c->slot;
        unpack(s, c->maxn, B, p, G, cd);
        float sc = ((const float *)(s + c->maxn))[p / G];
        for (int e = 0; e < G; e++) acc[e] = fmaf((float)cd[e], sc, acc[e]);
    }
    float amax = 0.f;
    for (int e = 0; e < G; e++) amax = fmaxf(amax, fabsf(acc[e]));
    float sc = fmaxf(amax, 1e-30f) / (float)((1 << (RB - 1)) - 1), inv = 1.0f / sc;
    for (int e = 0; e < G; e++) {
        float r = acc[e] * inv;
        cd[e] = (int32_t)(r + copysignf(0.5f, r));   // trunc(r +- 0.5) = round half away from zero
    }
    uint8_t *red = c->base + c->red_off;
    pack(red, c->maxn, RB, p, G, cd);
    ((float *)(red + c->maxn))[p / G] = sc;
}

// ---------------------------------------------------------------- AVX-512 (G = 128)
#if defined(__x86_64__)
#define T512 __attribute__((target("avx512f,fma")))
T512 static inline void deq512(const uint8_t *s, int64_t maxn, int B, int64_t p, int first, __m512 acc[8]) {
    const __m512 vs = _mm512_set1_ps(((const float *)(s + maxn))[p / 128]);
    if (B == 8) {
        const int8_t *q = (const int8_t *)s + p;
        for (int v = 0; v < 8; v++) {
            __m512 f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(q + 16 * v))));
            acc[v] = first ? _mm512_mul_ps(f, vs) : _mm512_fmadd_ps(f, vs, acc[v]);
        }
        return;
    }
    const int hb = B - 4;
    __m512i L = _mm512_loadu_si512((const void *)((const uint32_t *)s + p / 8));
    __m512i H = B == 6 ? _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)((const uint16_t *)(s + maxn / 2) + p / 8)))
                       : _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)((const uint8_t *)(s + maxn / 2) + p / 8)));
    H = _mm512_slli_epi32(H, 4);
    const __m512i m15 = _mm512_set1_epi32(15), mh = _mm512_set1_epi32(((1 << hb) - 1) << 4),
                  off = _mm512_set1_epi32(1 << (B - 1));
    for (int e = 0; e < 8; e++) {
        __m512i lo = _mm512_and_si512(_mm512_srl_epi32(L, _mm_cvtsi32_si128(4 * e)), m15);
        __m512i hi = _mm512_and_si512(_mm512_srl_epi32(H, _mm_cvtsi32_si128(hb * e)), mh);
        __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_or_si512(lo, hi), off));
        acc[e] = first ? _mm512_mul_ps(f, vs) : _mm512_fmadd_ps(f, vs, acc[e]);
    }
}

T512 static void group512(ctx_t *c, int j, int64_t q, int B, int RB, int ownq, int64_t o) {
    const int W = c->W;
    __m512 acc[8];
    int64_t p = (int64_t)j * q + o;
    if (ownq) {
        deq512(c->base + c->in_off + (int64_t)j * c->slot, c->maxn, B, p, 1, acc);
    } else {
        const uint16_t *ow = (const uint16_t *)(c->base + c->own_off + (int64_t)j * c->own_slot) + o;
        for (int v = 0; v < 8; v++)
            acc[v] = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(ow + 16 * v))), 16));
    }
    for (int k = 0; k < W; k++)
        if (k != j) deq512(c->base + c->in_off + (int64_t)k * c->slot, c->maxn, B, p, 0, acc);
    __m512 m = _mm512_abs_ps(acc[0]);
    for (int v = 1; v < 8; v++) m = _mm512_max_ps(m, _mm512_abs_ps(acc[v]));
    float amax = _mm512_reduce_max_ps(m);
    float sc = fmaxf(amax, 1e-30f) / (float)((1 << (RB - 1)) - 1), inv = 1.0f / sc;
    const __m512 vinv = _mm512_set1_ps(inv), half = _mm512_set1_ps(0.5f);
    const __m512i sgn = _mm512_set1_epi32((int)0x80000000);
    uint8_t *red = c->base + c->red_off;
    __m512i cd[8];
    for (int v = 0; v < 8; v++) {
        __m512 r = _mm512_mul_ps(acc[v], vinv);
        __m512 h = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(half),
                                                       _mm512_and_si512(_mm512_castps_si512(r), sgn)));
        cd[v] = _mm512_cvttps_epi32(_mm512_add_ps(r, h));
    }
    if (RB == 8) {
        for (int v = 0; v < 8; v++) _mm_storeu_si128((__m128i *)((int8_t *)red + p + 16 * v), _mm512_cvtepi32_epi8(cd[v]));
    } else {
        const int hb = RB - 4;
        const __m512i off = _mm512_set1_epi32(1 << (RB - 1)), m15 = _mm512_set1_epi32(15),
                      mh = _mm512_set1_epi32((1 << hb) - 1);
        __m512i L = _mm512_setzero_si512(), H = _mm512_setzero_si512();
        for (int e = 0; e < 8; e++) {
            __m512i u = _mm512_add_epi32(cd[e], off);
            L = _mm512_or_si512(L, _mm512_sll_epi32(_mm512_and_si512(u, m15), _mm_cvtsi32_si128(4 * e)));
            H = _mm512_or_si512(H, _mm512_sll_epi32(_mm512_and_si512(_mm512_srli_epi32(u, 4), mh), _mm_cvtsi32_si128(hb * e)));
        }
        _mm512_storeu_si512((void *)((uint32_t *)red + p / 8), L);
        if (RB == 6) _mm256_storeu_si256((__m256i *)((uint16_t *)(red + c->maxn / 2) + p / 8), _mm512_cvtepi32_epi16(H));
        else _mm_storeu_si128((__m128i *)((uint8_t *)(red + c->maxn / 2) + p / 8), _mm512_cvtepi32_epi8(H));
    }
    ((float *)(red + c->maxn))[p / 128] = sc;
}

#define T256 __attribute__((target("avx2,fma")))
T256 static inline void deq256(const uint8_t *s, int64_t maxn, int B, int64_t p, int first, __m256 acc[16]) {
    const __m256 vs = _mm256_set1_ps(((const float *)(s + maxn))[p / 128]);
    if (B == 8) {
        const int8_t *q = (const int8_t *)s + p;
        for (int v = 0; v < 16; v++) {
            __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q + 8 * v))));
            acc[v] = first ? _mm256_mul_ps(f, vs) : _mm256_fmadd_ps(f, vs, acc[v]);
        }
        return;
    }
    const int hb = B - 4;
    const uint32_t *lw = (const uint32_t *)s + p / 8;
    __m256i L[2], H[2];
    for (int h = 0; h < 2; h++) {
        L[h] = _mm256_loadu_si256((const __m256i *)(lw + 8 * h));
        H[h] = B == 6 ? _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)((const uint16_t *)(s + maxn / 2) + p / 8 + 8 * h)))
                      : _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)((const uint8_t *)(s + maxn / 2) + p / 8 + 8 * h)));
        H[h] = _mm256_slli_epi32(H[h], 4);
    }
    const __m256i m15 = _mm256_set1_epi32(15), mh = _mm256_set1_epi32(((1 << hb) - 1) << 4),
                  off = _mm256_set1_epi32(1 << (B - 1));
    for (int e = 0; e < 8; e++)
        for (int h = 0; h < 2; h++) {
            __m256i lo = _mm256_and_si256(_mm256_srl_epi32(L[h], _mm_cvtsi32_si128(4 * e)), m15);
            __m256i hi = _mm256_and_si256(_mm256_srl_epi32(H[h], _mm_cvtsi32_si128(hb * e)), mh);
            __m256 f = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_or_si256(lo, hi), off));
            acc[2 * e + h] = first ? _mm256_mul_ps(f, vs) : _mm256_fmadd_ps(f, vs, acc[2 * e + h]);
        }
}

T256 static void group256(ctx_t *c, int j, int64_t q, int B, int RB, int ownq, int64_t o) {
    const int W = c->W;
    __m256 acc[16];
    int64_t p = (int64_t)j * q + o;
    if (ownq) {
        deq256(c->base + c->in_off + (int64_t)j * c->slot, c->maxn, B, p, 1, acc);
    } else {
        const uint16_t *ow = (const uint16_t *)(c->base + c->own_off + (int64_t)j * c->own_slot) + o;
        for (int v = 0; v < 16; v++)
            acc[v] = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(ow + 8 * v))), 16));
    }
    for (int k = 0; k < W; k++)
        if (k != j) deq256(c->base + c->in_off + (int64_t)k * c->slot, c->maxn, B, p, 0, acc);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m = _mm256_and_ps(acc[0], absm);
    for (int v = 1; v < 16; v++) m = _mm256_max_ps(m, _mm256_and_ps(acc[v], absm));
    float t8[8];
    _mm256_storeu_ps(t8, m);
    float amax = 0.f;
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, t8[i]);
    float sc = fmaxf(amax, 1e-30f) / (float)((1 << (RB - 1)) - 1), inv = 1.0f / sc;
    const __m256 vinv = _mm256_set1_ps(inv), half = _mm256_set1_ps(0.5f);
    const __m256 sgn = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
    uint8_t *red = c->base + c->red_off;
    int32_t cd[128];
    for (int v = 0; v < 16; v++) {
        __m256 r = _mm256_mul_ps(acc[v], vinv);
        __m256 h = _mm256_or_ps(half, _mm256_and_ps(r, sgn));
        _mm256_storeu_si256((__m256i *)(cd + 8 * v), _mm256_cvttps_epi32(_mm256_add_ps(r, h)));
    }
    pack(red, c->maxn, RB, p, 128, cd);
    ((float *)(red + c->maxn))[p / 128] = sc;
}
#endif

static void unit(ctx_t *c, int b, int i, int64_t q, int B, int RB, int ownq) {
    const int G = c->G;
    int64_t per = ((q + c->NB - 1) / c->NB + G - 1) / G * G;
    int64_t sub = ((per + c->K - 1) / c->K + G - 1) / G * G;
    int64_t lo = (int64_t)b * per, hi = lo + per < q ? lo + per : q;
    int64_t c0 = lo + (int64_t)i * sub, c1 = c0 + sub < hi ? c0 + sub : hi;
    for (int j = 0; j < c->W; j++)
        for (int64_t o = c0; o < c1; o += G) {
#if defined(__x86_64__)
            if (c->simd == 2) { group512(c, j, q, B, RB, ownq, o); continue; }
            if (c->simd == 1) { group256(c, j, q, B, RB, ownq, o); continue; }
#endif
            group_scalar(c, j, q, B, RB, ownq, o);
        }
}

static void *worker(void *arg) {
    struct targ *a = (struct targ *)arg;
    ctx_t *c = (ctx_t *)a->c;
    int t = a->t;
    if (c->cpus[t] >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(c->cpus[t], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    int32_t e = 0;
    while (!c->stop) {
        e++;
        for (int i = 0; i < c->K && !c->stop; i++) {
            for (int b = t; b < c->NB && !c->stop; b += c->nt) {
                for (int r = 0; r < c->W; r++) {
                    volatile int32_t *f = gf(c, r, b, i);
                    while (__atomic_load_n(f, __ATOMIC_ACQUIRE) < e) {
                        if (c->stop) return NULL;
                        PAUSE();
                        c->spins[t * 8]++;
                    }
                }
                volatile int32_t *h = gf(c, 0, b, i);
                int64_t q = h[1];
                int32_t m = h[2];
                unit(c, b, i, q, m & 255, (m >> 8) & 255, (m >> 16) & 1);
                __atomic_store_n(cf(c, b, i), e, __ATOMIC_RELEASE);
                c->units[t * 8]++;
            }
        }
    }
    return NULL;
}

void *arc_create(void *base, int W, int NB, int K, int G, int FS, int nt, int64_t maxn, int64_t slot, int64_t own_slot,
                 int64_t in_off, int64_t own_off, int64_t red_off, int64_t gf_off, int64_t cf_off, const int *cpus,
                 int simd) {
    if (nt < 1 || nt > TMAX || G > GMAX || G % 128) return NULL;
    ctx_t *c = (ctx_t *)calloc(1, sizeof(ctx_t));
    c->base = (uint8_t *)base;
    c->W = W; c->NB = NB; c->K = K; c->G = G; c->FS = FS; c->nt = nt;
    c->maxn = maxn; c->slot = slot; c->own_slot = own_slot;
    c->in_off = in_off; c->own_off = own_off; c->red_off = red_off; c->gf_off = gf_off; c->cf_off = cf_off;
    for (int t = 0; t < nt; t++) c->cpus[t] = cpus ? cpus[t] : -1;
#if defined(__x86_64__)
    // simd: 0 scalar, 1 AVX2, 2 AVX-512 (requested level capped by the CPU)
    if (G == 128 && simd >= 2 && __builtin_cpu_supports("avx512f")) c->simd = 2;
    else if (G == 128 && simd >= 1 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) c->simd = 1;
#endif
    return c;
}

int arc_simd(void *h) { return ((ctx_t *)h)->simd; }

int arc_start(void *h) {
    ctx_t *c = (ctx_t *)h;
    for (int t = 0; t < c->nt; t++) {
        c->args[t].c = c;
        c->args[t].t = t;
        if (pthread_create(&c->th[t], NULL, worker, &c->args[t])) return -1;
    }
    c->started = 1;
    return 0;
}

void arc_stop(void *h) {
    ctx_t *c = (ctx_t *)h;
    c->stop = 1;
    if (c->started)
        for (int t = 0; t < c->nt; t++) pthread_join(c->th[t], NULL);
    c->started = 0;
}

int64_t arc_stat(void *h, int which) {
    ctx_t *c = (ctx_t *)h;
    int64_t s = 0;
    for (int t = 0; t < c->nt; t++) s += which ? c->spins[t * 8] : c->units[t * 8];
    return s;
}

// throughput / exactness test without GPUs: reduce every unit of one all-reduce on the calling thread
void arc_reduce_all(void *h, int64_t q, int B, int RB, int ownq) {
    ctx_t *c = (ctx_t *)h;
    for (int i = 0; i < c->K; i++)
        for (int b = 0; b < c->NB; b++) unit(c, b, i, q, B, RB, ownq);
}
