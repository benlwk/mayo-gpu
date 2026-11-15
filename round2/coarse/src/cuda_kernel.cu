#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include "../include/cuda_kernel.cuh"
#include "../include/fips202.cuh"
#include "../include/inputs.cuh"
#include "../include/parameters.cuh"
#include "../include/mayo.cuh"

#define MAYO_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAYO_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define PK_PRF         AES_128_CTR

typedef struct {
    uint64_t p[(142 * (142 + 1) / 2 * 9) + (142 * 17 * 9)];
    uint8_t O[142 * 17];
} host_dynamic_sk_t;

// Generalised hybrid layout
#define MAYO_HYBRID_GROUPS     (M_MAX / 64)
#define MAYO_BULK_ELEMS        (64 * MAYO_HYBRID_GROUPS)
#define MAYO_HYBRID_TAIL_WORDS ((M_MAX - MAYO_BULK_ELEMS + 15) / 16)
#define MAYO_HYBRID_MVEC_WORDS (4 * MAYO_HYBRID_GROUPS + MAYO_HYBRID_TAIL_WORDS)
#define MAYO_BIN_COLS          16 // rows per chunk of the bins accumulator

static void bitslice_m_vec_hybrid_host(const uint64_t *mvec, uint64_t *h, int m, int m_vec_limbs)
{
    (void)m_vec_limbs;
    for (int g = 0; g < MAYO_HYBRID_GROUPS; g++) {
        for (int p = 0; p < 4; p++) {
            uint64_t pl = 0;
            for (int i = 0; i < 64; i++) {
                const int idx = g * 64 + i;
                if (idx >= m) {
                    break;
                }
                const unsigned e = (unsigned)((mvec[idx / 16] >> (4 * (idx % 16))) & 0xf);
                pl |= (uint64_t)((e >> p) & 1) << i;
            }
            h[4 * g + p] = pl;
        }
    }
    for (int tw = 0; tw < MAYO_HYBRID_TAIL_WORDS; tw++) {
        uint64_t tail = 0;
        for (int i = 0; i < 16; i++) {
            const int idx = MAYO_BULK_ELEMS + 16 * tw + i;
            if (idx >= m) {
                break;
            }
            const unsigned e = (unsigned)((mvec[idx / 16] >> (4 * (idx % 16))) & 0xf);
            tail |= (uint64_t)e << (4 * i);
        }
        h[4 * MAYO_HYBRID_GROUPS + tw] = tail;
    }
}

static void bitslice_m_vec_host(const uint64_t *mvec, uint64_t *planes, int m, int m_vec_limbs,
                                int plane_words)
{
    (void)m_vec_limbs;
    for (int p = 0; p < 4; p++) {
        for (int w = 0; w < plane_words; w++) {
            uint64_t pl = 0;
            for (int i = 0; i < 64; i++) {
                const int idx = w * 64 + i;
                if (idx >= m) {
                    break;
                }
                const unsigned e = (unsigned)((mvec[idx / 16] >> (4 * (idx % 16))) & 0xf);
                pl |= (uint64_t)((e >> p) & 1) << i;
            }
            planes[p * plane_words + w] = pl;
        }
    }
}

static void upload_bitsliced_key(const mayo_params_t *p, const sk_t *expanded_sk, uint64_t **d_P1_bs,
                                 uint64_t **d_L_bs)
{
    *d_P1_bs = NULL;
    *d_L_bs = NULL;
    const int m_vec_limbs = PARAM_m_vec_limbs(p);
#if defined(MAYO_BITSLICE_HYBRID)

    const int plane_words = 1;
    const int mvec_bs = MAYO_HYBRID_MVEC_WORDS;
#else
    const int plane_words = (PARAM_m(p) + 63) / 64;
    const int mvec_bs = 4 * plane_words;
#endif
    const int p1_mvecs = PARAM_P1_limbs(p) / m_vec_limbs;
    const int l_mvecs = PARAM_v(p) * PARAM_o(p);

    uint64_t *P1_bs = new uint64_t[(size_t)p1_mvecs * mvec_bs];
    uint64_t *L_bs = new uint64_t[(size_t)l_mvecs * mvec_bs];
    for (int k = 0; k < p1_mvecs; k++) {
#if defined(MAYO_BITSLICE_HYBRID)
        bitslice_m_vec_hybrid_host(expanded_sk->p + (size_t)k * m_vec_limbs, P1_bs + (size_t)k * mvec_bs,
                                   PARAM_m(p), m_vec_limbs);
#else
        bitslice_m_vec_host(expanded_sk->p + (size_t)k * m_vec_limbs, P1_bs + (size_t)k * mvec_bs, PARAM_m(p),
                            m_vec_limbs, plane_words);
#endif
    }
    for (int k = 0; k < l_mvecs; k++) {
#if defined(MAYO_BITSLICE_HYBRID)
        bitslice_m_vec_hybrid_host(expanded_sk->p + PARAM_P1_limbs(p) + (size_t)k * m_vec_limbs,
                                   L_bs + (size_t)k * mvec_bs, PARAM_m(p), m_vec_limbs);
#else
        bitslice_m_vec_host(expanded_sk->p + PARAM_P1_limbs(p) + (size_t)k * m_vec_limbs,
                            L_bs + (size_t)k * mvec_bs, PARAM_m(p), m_vec_limbs, plane_words);
#endif
    }

    cudaMalloc(d_P1_bs, (size_t)p1_mvecs * mvec_bs * sizeof(uint64_t));
    cudaMalloc(d_L_bs, (size_t)l_mvecs * mvec_bs * sizeof(uint64_t));
    cudaMemcpy(*d_P1_bs, P1_bs, (size_t)p1_mvecs * mvec_bs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_L_bs, L_bs, (size_t)l_mvecs * mvec_bs * sizeof(uint64_t), cudaMemcpyHostToDevice);

    delete[] P1_bs;
    delete[] L_bs;
}

static void upload_bitsliced_pk(const mayo_params_t *p, const uint64_t *expanded_pk, uint64_t **d_P1_bs,
                                uint64_t **d_P2_bs, uint64_t **d_P3_bs)
{
    *d_P1_bs = NULL;
    *d_P2_bs = NULL;
    *d_P3_bs = NULL;
    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    const int mvec_bs = MAYO_HYBRID_MVEC_WORDS; // hybrid: 4 bulk planes + 1 tail word
    const int p1_mvecs = PARAM_P1_limbs(p) / m_vec_limbs;
    const int p2_mvecs = PARAM_P2_limbs(p) / m_vec_limbs;
    const int p3_mvecs = PARAM_P3_limbs(p) / m_vec_limbs;

    const uint64_t *P1 = expanded_pk;
    const uint64_t *P2 = P1 + PARAM_P1_limbs(p);
    const uint64_t *P3 = P2 + PARAM_P2_limbs(p);

    uint64_t *P1_bs = new uint64_t[(size_t)p1_mvecs * mvec_bs];
    uint64_t *P2_bs = new uint64_t[(size_t)p2_mvecs * mvec_bs];
    uint64_t *P3_bs = new uint64_t[(size_t)p3_mvecs * mvec_bs];
    for (int k = 0; k < p1_mvecs; k++) {
        bitslice_m_vec_hybrid_host(P1 + (size_t)k * m_vec_limbs, P1_bs + (size_t)k * mvec_bs, PARAM_m(p),
                                   m_vec_limbs);
    }
    for (int k = 0; k < p2_mvecs; k++) {
        bitslice_m_vec_hybrid_host(P2 + (size_t)k * m_vec_limbs, P2_bs + (size_t)k * mvec_bs, PARAM_m(p),
                                   m_vec_limbs);
    }
    for (int k = 0; k < p3_mvecs; k++) {
        bitslice_m_vec_hybrid_host(P3 + (size_t)k * m_vec_limbs, P3_bs + (size_t)k * mvec_bs, PARAM_m(p),
                                   m_vec_limbs);
    }

    cudaMalloc(d_P1_bs, (size_t)p1_mvecs * mvec_bs * sizeof(uint64_t));
    cudaMalloc(d_P2_bs, (size_t)p2_mvecs * mvec_bs * sizeof(uint64_t));
    cudaMalloc(d_P3_bs, (size_t)p3_mvecs * mvec_bs * sizeof(uint64_t));
    cudaMemcpy(*d_P1_bs, P1_bs, (size_t)p1_mvecs * mvec_bs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_P2_bs, P2_bs, (size_t)p2_mvecs * mvec_bs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_P3_bs, P3_bs, (size_t)p3_mvecs * mvec_bs * sizeof(uint64_t), cudaMemcpyHostToDevice);

    delete[] P1_bs;
    delete[] P2_bs;
    delete[] P3_bs;
}

// Real MAYO signer entry point
extern "C" int crypto_sign(unsigned char *sm, size_t *smlen, const unsigned char *m, size_t mlen,
                           const unsigned char *sk);
extern "C" int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);
extern "C" int crypto_sign_open(unsigned char *m, size_t *mlen, const unsigned char *sm, size_t smlen,
                                const unsigned char *pk);
extern "C" int crypto_sign_verify(const unsigned char *sig, size_t siglen, const unsigned char *m,
                                  size_t mlen, const unsigned char *pk);

// MAYO-C can expand the compact secret key once on the CPU
extern "C" int mayo_expand_sk(const mayo_params_t *p, const unsigned char *csk, host_dynamic_sk_t *esk);
extern "C" int mayo_expand_pk(const mayo_params_t *p, const unsigned char *cpk, uint64_t *pk);
extern "C" const mayo_params_t MAYO_VARIANT;

// Convert two gettimeofday() timestamps into milliseconds
static double elapsed_ms(const timeval &start, const timeval &stop)
{
    return (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_usec - start.tv_usec) / 1000.0;
}

static int crypto_sign_quiet(unsigned char *sm, size_t *smlen, const unsigned char *m, size_t mlen,
                             const unsigned char *sk)
{
    fflush(stdout);
    int saved_stdout = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (saved_stdout >= 0 && devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
    }

    int ret = crypto_sign(sm, smlen, m, mlen, sk);

    fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }
    if (devnull >= 0) {
        close(devnull);
    }
    return ret;
}

__device__ __constant__ const unsigned char f_tail_64[] = {8, 0, 2, 8};
__device__ __constant__ const unsigned char f_tail_78[] = {8, 1, 1, 0};
__device__ __constant__ const unsigned char f_tail_108[] = {8, 0, 1, 7};
__device__ __constant__ const unsigned char f_tail_142[] = {4, 0, 8, 1};

// 2
#if defined(MAYO_VARIANT)
__device__ const unsigned char *f_tail = PARAM_NAME(f_tail_arr);
#else
#error "no MAYO variant selected"
#endif

#undef PARAM_f_tail
#define PARAM_f_tail(p) f_tail

__device__ __forceinline__ void decode_device(const unsigned char *m, unsigned char *mdec, int mdeclen)
{
    int i;
    for (i = 0; i < mdeclen / 2; ++i) {
        mdec[i * 2] = m[i] & 0xf;
        mdec[i * 2 + 1] = m[i] >> 4;
    }
    if (mdeclen % 2 == 1) {
        mdec[i * 2] = m[i] & 0x0f;
    }
}

__device__ __forceinline__ void encode_device(const unsigned char *m, unsigned char *menc, int mlen)
{
    int i;
    for (i = 0; i < mlen / 2; ++i) {
        menc[i] = m[i * 2] | (m[i * 2 + 1] << 4);
    }
    if (mlen % 2 == 1) {
        menc[i] = m[i * 2];
    }
}

__device__ unsigned char mul_f_device(unsigned char a, unsigned char b)
{
    unsigned char p = 0;
    p = (a & 1) * b;
    p ^= (a & 2) * b;
    p ^= (a & 4) * b;
    p ^= (a & 8) * b;

    unsigned char top_p = p & 0xf0;
    return (p ^ (top_p >> 4) ^ (top_p >> 3)) & 0x0f;
}

__device__ __forceinline__ uint64_t mul_fx8_device(unsigned char a, uint64_t b)
{
    uint64_t p = 0;
    p = (a & 1) * b;
    p ^= (a & 2) * b;
    p ^= (a & 4) * b;
    p ^= (a & 8) * b;

    uint64_t top_p = p & 0xf0f0f0f0f0f0f0f0ULL;
    return (p ^ (top_p >> 4) ^ (top_p >> 3)) & 0x0f0f0f0f0f0f0f0fULL;
}

__device__ __forceinline__ void m_vec_add_device(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        acc[i] ^= in[i];
    }
}

__device__ __forceinline__ uint64_t gf16v_mul_u64_device(uint64_t a, unsigned char b)
{
    uint64_t p = 0;
#ifdef MAYO_GF16_MASK
    uint64_t b0 = -((uint64_t)(b & 1));
    p = a & b0;
#else
    p = (b & 1) * a;
#endif
    p ^= ((b >> 1) & 1) *
         (((a ^ (a & 0x8888888888888888ULL)) << 1) ^ (((a & 0x8888888888888888ULL) >> 3) * 3));

    uint64_t a2 = ((a ^ (a & 0x8888888888888888ULL)) << 1) ^ (((a & 0x8888888888888888ULL) >> 3) * 3);
    uint64_t a4 = ((a2 ^ (a2 & 0x8888888888888888ULL)) << 1) ^ (((a2 & 0x8888888888888888ULL) >> 3) * 3);
    uint64_t a8 = ((a4 ^ (a4 & 0x8888888888888888ULL)) << 1) ^ (((a4 & 0x8888888888888888ULL) >> 3) * 3);

    p ^= ((b >> 2) & 1) * a4;
    p ^= ((b >> 3) & 1) * a8;
    return p;
}

__device__ __forceinline__ void m_vec_mul_add_device(int m_vec_limbs, const uint64_t *in, unsigned char a,
                                                     uint64_t *acc)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        acc[i] ^= gf16v_mul_u64_device(in[i], a);
    }
}

#if defined(MAYO_BITSLICE) || defined(MAYO_BITSLICE_SPS) || defined(MAYO_BITSLICE_HYBRID) ||                 \
    defined(MAYO_BITSLICE_SOLVE)

__device__ __forceinline__ void m_vec_mul_add_bs_device(const uint64_t *in, unsigned char b, uint64_t *acc,
                                                        int plane_words)
{
    // scalar bits broadcast to all-ones/all-zeros masks
    const uint64_t b0 = -((uint64_t)(b & 1));
    const uint64_t b1 = -((uint64_t)((b >> 1) & 1));
    const uint64_t b2 = -((uint64_t)((b >> 2) & 1));
    const uint64_t b3 = -((uint64_t)((b >> 3) & 1));

    for (int w = 0; w < plane_words; w++) {
        const uint64_t a0 = in[w];
        const uint64_t a1 = in[w + plane_words];
        const uint64_t a2 = in[w + 2 * plane_words];
        const uint64_t a3 = in[w + 3 * plane_words];

        const uint64_t c0 = a0 & b0;
        const uint64_t c1 = (a0 & b1) ^ (a1 & b0);
        const uint64_t c2 = (a0 & b2) ^ (a1 & b1) ^ (a2 & b0);
        const uint64_t c3 = (a0 & b3) ^ (a1 & b2) ^ (a2 & b1) ^ (a3 & b0);
        const uint64_t c4 = (a1 & b3) ^ (a2 & b2) ^ (a3 & b1);
        const uint64_t c5 = (a2 & b3) ^ (a3 & b2);
        const uint64_t c6 = a3 & b3;

        acc[w] ^= c0 ^ c4;
        acc[w + plane_words] ^= c1 ^ c4 ^ c5;
        acc[w + 2 * plane_words] ^= c2 ^ c5 ^ c6;
        acc[w + 3 * plane_words] ^= c3 ^ c6;
    }
}

__device__ void unbitslice_m_vec_device(const uint64_t *planes, uint64_t *mvec, int m, int m_vec_limbs,
                                        int plane_words)
{
    for (int w = 0; w < m_vec_limbs; w++) {
        uint64_t word = 0;
        for (int i = 0; i < 16; i++) {
            const int idx = w * 16 + i;
            if (idx >= m) {
                break;
            }
            const uint64_t pw = idx / 64;
            const uint64_t bit = idx % 64;
            const uint64_t e = ((planes[pw] >> bit) & 1) | (((planes[plane_words + pw] >> bit) & 1) << 1) |
                               (((planes[2 * plane_words + pw] >> bit) & 1) << 2) |
                               (((planes[3 * plane_words + pw] >> bit) & 1) << 3);
            word |= e << (4 * i);
        }
        mvec[w] = word;
    }
}

__device__ __forceinline__ void m_vec_mul_add_hybrid_device(const uint64_t *in, unsigned char b,
                                                            uint64_t *acc)
{
    const uint64_t b0 = -((uint64_t)(b & 1));
    const uint64_t b1 = -((uint64_t)((b >> 1) & 1));
    const uint64_t b2 = -((uint64_t)((b >> 2) & 1));
    const uint64_t b3 = -((uint64_t)((b >> 3) & 1));

#pragma unroll
    for (int g = 0; g < MAYO_HYBRID_GROUPS; g++) {
        const uint64_t a0 = in[4 * g + 0];
        const uint64_t a1 = in[4 * g + 1];
        const uint64_t a2 = in[4 * g + 2];
        const uint64_t a3 = in[4 * g + 3];

        const uint64_t c0 = a0 & b0;
        const uint64_t c1 = (a0 & b1) ^ (a1 & b0);
        const uint64_t c2 = (a0 & b2) ^ (a1 & b1) ^ (a2 & b0);
        const uint64_t c3 = (a0 & b3) ^ (a1 & b2) ^ (a2 & b1) ^ (a3 & b0);
        const uint64_t c4 = (a1 & b3) ^ (a2 & b2) ^ (a3 & b1);
        const uint64_t c5 = (a2 & b3) ^ (a3 & b2);
        const uint64_t c6 = a3 & b3;

        acc[4 * g + 0] ^= c0 ^ c4;
        acc[4 * g + 1] ^= c1 ^ c4 ^ c5;
        acc[4 * g + 2] ^= c2 ^ c5 ^ c6;
        acc[4 * g + 3] ^= c3 ^ c6;
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_HYBRID_TAIL_WORDS; tw++) {
        acc[4 * MAYO_HYBRID_GROUPS + tw] ^= gf16v_mul_u64_device(in[4 * MAYO_HYBRID_GROUPS + tw], b);
    }
}

// Hybrid (4 bulk planes + 1 tail word) -> nibble m_vec.
__device__ void unbitslice_m_vec_hybrid_device(const uint64_t *h, uint64_t *mvec, int m, int m_vec_limbs)
{
    (void)m_vec_limbs;
#pragma unroll
    for (int g = 0; g < MAYO_HYBRID_GROUPS; g++) {
        for (int w = 0; w < 4; w++) {
            uint64_t word = 0;
            for (int i = 0; i < 16; i++) {
                const int idx = w * 16 + i;
                const uint64_t e = ((h[4 * g + 0] >> idx) & 1) | (((h[4 * g + 1] >> idx) & 1) << 1) |
                                   (((h[4 * g + 2] >> idx) & 1) << 2) | (((h[4 * g + 3] >> idx) & 1) << 3);
                word |= e << (4 * i);
            }
            mvec[4 * g + w] = word;
        }
    }

#pragma unroll
    for (int tw = 0; tw < MAYO_HYBRID_TAIL_WORDS; tw++) {
        const int first = MAYO_BULK_ELEMS + 16 * tw;
        int valid = m - first;
        if (valid > 16) {
            valid = 16;
        }
        if (valid <= 0) {
            break;
        }
        const uint64_t mask = (valid >= 16) ? ~0ULL : ((1ULL << (4 * valid)) - 1);
        mvec[4 * MAYO_HYBRID_GROUPS + tw] = h[4 * MAYO_HYBRID_GROUPS + tw] & mask;
    }
}
#endif // MAYO_BITSLICE

#ifdef MAYO_BITSLICE_SPS

__device__ __forceinline__ void m_vec_add_h(const uint64_t *in, uint64_t *acc)
{
    for (int w = 0; w < MAYO_HYBRID_MVEC_WORDS; w++) {
        acc[w] ^= in[w];
    }
}

__device__ __forceinline__ void m_vec_mul_add_x_h(const uint64_t *in, uint64_t *acc)
{
#pragma unroll
    for (int g = 0; g < MAYO_HYBRID_GROUPS; g++) {
        const uint64_t a0 = in[4 * g], a1 = in[4 * g + 1], a2 = in[4 * g + 2], a3 = in[4 * g + 3];
        acc[4 * g + 0] ^= a3;
        acc[4 * g + 1] ^= a0 ^ a3;
        acc[4 * g + 2] ^= a1;
        acc[4 * g + 3] ^= a2;
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_HYBRID_TAIL_WORDS; tw++) {
        const uint64_t v = in[4 * MAYO_HYBRID_GROUPS + tw];
        const uint64_t t = v & 0x8888888888888888ULL;
        acc[4 * MAYO_HYBRID_GROUPS + tw] ^= ((v ^ t) << 1) ^ ((t >> 3) * 3);
    }
}

__device__ __forceinline__ void m_vec_mul_add_x_inv_h(const uint64_t *in, uint64_t *acc)
{
#pragma unroll
    for (int g = 0; g < MAYO_HYBRID_GROUPS; g++) {
        const uint64_t a0 = in[4 * g], a1 = in[4 * g + 1], a2 = in[4 * g + 2], a3 = in[4 * g + 3];
        acc[4 * g + 0] ^= a0 ^ a1;
        acc[4 * g + 1] ^= a2;
        acc[4 * g + 2] ^= a3;
        acc[4 * g + 3] ^= a0;
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_HYBRID_TAIL_WORDS; tw++) {
        const uint64_t v = in[4 * MAYO_HYBRID_GROUPS + tw];
        const uint64_t t = v & 0x1111111111111111ULL;
        acc[4 * MAYO_HYBRID_GROUPS + tw] ^= ((v ^ t) >> 1) ^ (t * 9);
    }
}

__device__ void m_vec_multiply_bins_h(uint64_t *bins, uint64_t *out)
{
    m_vec_mul_add_x_inv_h(bins + 5 * MAYO_HYBRID_MVEC_WORDS, bins + 10 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 11 * MAYO_HYBRID_MVEC_WORDS, bins + 12 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 10 * MAYO_HYBRID_MVEC_WORDS, bins + 7 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 12 * MAYO_HYBRID_MVEC_WORDS, bins + 6 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 7 * MAYO_HYBRID_MVEC_WORDS, bins + 14 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 6 * MAYO_HYBRID_MVEC_WORDS, bins + 3 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 14 * MAYO_HYBRID_MVEC_WORDS, bins + 15 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 3 * MAYO_HYBRID_MVEC_WORDS, bins + 8 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 15 * MAYO_HYBRID_MVEC_WORDS, bins + 13 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 8 * MAYO_HYBRID_MVEC_WORDS, bins + 4 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 13 * MAYO_HYBRID_MVEC_WORDS, bins + 9 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 4 * MAYO_HYBRID_MVEC_WORDS, bins + 2 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_inv_h(bins + 9 * MAYO_HYBRID_MVEC_WORDS, bins + 1 * MAYO_HYBRID_MVEC_WORDS);
    m_vec_mul_add_x_h(bins + 2 * MAYO_HYBRID_MVEC_WORDS, bins + 1 * MAYO_HYBRID_MVEC_WORDS);
    for (int w = 0; w < MAYO_HYBRID_MVEC_WORDS; w++) {
        out[w] = bins[MAYO_HYBRID_MVEC_WORDS + w];
    }
}

__device__ void m_calculate_PS_SPS_device_h(const mayo_params_t *p, const uint64_t *P1_bs,
                                            const uint64_t *P2_bs, const uint64_t *P3_bs,
                                            const unsigned char *S, uint64_t *SPS)
{
    const int param_v = PARAM_v(p);
    const int param_o = PARAM_o(p);
    const int param_k = PARAM_k(p);
    const int param_n = param_o + param_v;
    const int mvb = MAYO_HYBRID_MVEC_WORDS;

    uint64_t PS[N_MAX * MAYO_HYBRID_MVEC_WORDS];

    for (int col = 0; col < param_k; col++) {
        const unsigned char *s = S + col * param_n;
        int P1_used = 0, P3_used = 0;
        for (int row = 0; row < param_n; row++) {
            uint64_t acc[MAYO_HYBRID_MVEC_WORDS] = {0};
            if (row < param_v) {
                for (int j = row; j < param_v; j++) {
                    m_vec_mul_add_hybrid_device(P1_bs + (P1_used++) * mvb, s[j], acc);
                }
                for (int j = 0; j < param_o; j++) {
                    m_vec_mul_add_hybrid_device(P2_bs + (row * param_o + j) * mvb, s[param_v + j], acc);
                }
            } else {
                for (int j = row; j < param_n; j++) {
                    m_vec_mul_add_hybrid_device(P3_bs + (P3_used++) * mvb, s[j], acc);
                }
            }
            for (int w = 0; w < mvb; w++) {
                PS[row * mvb + w] = acc[w];
            }
        }
        for (int row = 0; row < param_k; row++) {
            uint64_t acc[MAYO_HYBRID_MVEC_WORDS] = {0};
            for (int j = 0; j < param_n; j++) {
                m_vec_mul_add_hybrid_device(PS + j * mvb, S[row * param_n + j], acc);
            }
            for (int w = 0; w < mvb; w++) {
                SPS[(row * param_k + col) * mvb + w] = acc[w];
            }
        }
    }
}

#endif // MAYO_BITSLICE_SPS

__device__ __forceinline__ void m_vec_copy_device(int m_vec_limbs, const uint64_t *in, uint64_t *out)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        out[i] = in[i];
    }
}

__device__ __forceinline__ void m_vec_mul_add_x_device(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    const uint64_t mask_msb = 0x8888888888888888ULL;
    for (int i = 0; i < m_vec_limbs; i++) {
        uint64_t t = in[i] & mask_msb;
        acc[i] ^= ((in[i] ^ t) << 1) ^ ((t >> 3) * 3);
    }
}

__device__ __forceinline__ void m_vec_mul_add_x_inv_device(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    const uint64_t mask_lsb = 0x1111111111111111ULL;
    for (int i = 0; i < m_vec_limbs; i++) {
        uint64_t t = in[i] & mask_lsb;
        acc[i] ^= ((in[i] ^ t) >> 1) ^ (t * 9);
    }
}

__device__ void m_vec_multiply_bins_device(int m_vec_limbs, uint64_t *bins, uint64_t *out)
{
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 5 * m_vec_limbs, bins + 10 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 11 * m_vec_limbs, bins + 12 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 10 * m_vec_limbs, bins + 7 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 12 * m_vec_limbs, bins + 6 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 7 * m_vec_limbs, bins + 14 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 6 * m_vec_limbs, bins + 3 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 14 * m_vec_limbs, bins + 15 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 3 * m_vec_limbs, bins + 8 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 15 * m_vec_limbs, bins + 13 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 8 * m_vec_limbs, bins + 4 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 13 * m_vec_limbs, bins + 9 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 4 * m_vec_limbs, bins + 2 * m_vec_limbs);
    m_vec_mul_add_x_inv_device(m_vec_limbs, bins + 9 * m_vec_limbs, bins + 1 * m_vec_limbs);
    m_vec_mul_add_x_device(m_vec_limbs, bins + 2 * m_vec_limbs, bins + 1 * m_vec_limbs);
    m_vec_copy_device(m_vec_limbs, bins + m_vec_limbs, out);
}

__device__ void mat_mul_device(const unsigned char *A, const unsigned char *B, unsigned char *C, int rows,
                               int cols, int k)
{
    for (int j = 0; j < k; ++j) {
        for (int i0 = 0; i0 < rows; i0 += 16) {
            const int nrows = MAYO_MIN(16, rows - i0);
            uint64_t acc_word = 0;
            for (int l = 0; l < cols; ++l) {
                const unsigned char scalar = B[l * k + j] & 0x0f;
                uint64_t packed = 0;
                for (int r = 0; r < 16; ++r) {
                    uint64_t nib = 0;
                    if (r < nrows) {
                        nib = A[(i0 + r) * cols + l] & 0x0f;
                    }
                    packed |= nib << (4 * r);
                }
                acc_word ^= gf16v_mul_u64_device(packed, scalar);
            }
            for (int r = 0; r < nrows; ++r) {
                C[(i0 + r) * k + j] = (unsigned char)((acc_word >> (4 * r)) & 0x0f);
            }
        }
    }
}

__device__ void mat_add_device(const unsigned char *A, const unsigned char *B, unsigned char *C, int rows,
                               int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] ^ B[i];
    }
}

__device__ void PK_PRF_device(unsigned char *out, size_t outlen, const unsigned char *in, size_t inlen)
{
    shake256_device(out, outlen, in, inlen);
}

__device__ void mul_add_m_upper_triangular_mat_x_mat_device(int m_vec_limbs, const uint64_t *bs_mat,
                                                            const unsigned char *mat, uint64_t *acc,
                                                            int bs_mat_rows, int bs_mat_cols, int mat_cols,
                                                            int triangular)
{
    int bs_mat_entries_used = 0;
    for (int r = 0; r < bs_mat_rows; r++) {
        for (int c = triangular * r; c < bs_mat_cols; c++) {
            for (int k = 0; k < mat_cols; k++) {
                m_vec_mul_add_device(m_vec_limbs, bs_mat + m_vec_limbs * bs_mat_entries_used,
                                     mat[c * mat_cols + k], acc + m_vec_limbs * (r * mat_cols + k));
            }
            bs_mat_entries_used++;
        }
    }
}

__device__ void mul_add_m_upper_triangular_mat_x_mat_trans_device(int m_vec_limbs, const uint64_t *bs_mat,
                                                                  const unsigned char *mat, uint64_t *acc,
                                                                  int bs_mat_rows, int bs_mat_cols,
                                                                  int mat_rows, int triangular)
{
    int bs_mat_entries_used = 0;
    for (int r = 0; r < bs_mat_rows; r++) {
        for (int c = triangular * r; c < bs_mat_cols; c++) {
            for (int k = 0; k < mat_rows; k++) {
                m_vec_mul_add_device(m_vec_limbs, bs_mat + m_vec_limbs * bs_mat_entries_used,
                                     mat[k * bs_mat_cols + c], acc + m_vec_limbs * (r * mat_rows + k));
            }
            bs_mat_entries_used++;
        }
    }
}

__device__ void mul_add_mat_x_m_mat_device(int m_vec_limbs, const unsigned char *mat, const uint64_t *bs_mat,
                                           uint64_t *acc, int mat_rows, int mat_cols, int bs_mat_cols)
{
    for (int r = 0; r < mat_rows; r++) {
        for (int c = 0; c < mat_cols; c++) {
            for (int k = 0; k < bs_mat_cols; k++) {
                m_vec_mul_add_device(m_vec_limbs, bs_mat + m_vec_limbs * (c * bs_mat_cols + k),
                                     mat[r * mat_cols + c], acc + m_vec_limbs * (r * bs_mat_cols + k));
            }
        }
    }
}

__device__ void mul_add_mat_trans_x_m_mat_device(int m_vec_limbs, const unsigned char *mat,
                                                 const uint64_t *bs_mat, uint64_t *acc, int mat_rows,
                                                 int mat_cols, int bs_mat_cols)
{
    for (int r = 0; r < mat_cols; r++) {
        for (int c = 0; c < mat_rows; c++) {
            for (int k = 0; k < bs_mat_cols; k++) {
                m_vec_mul_add_device(m_vec_limbs, bs_mat + m_vec_limbs * (c * bs_mat_cols + k),
                                     mat[c * mat_cols + r], acc + m_vec_limbs * (r * bs_mat_cols + k));
            }
        }
    }
}

__device__ void P1P1t_times_O_device(const mayo_params_t *p, const uint64_t *P1, const unsigned char *O,
                                     uint64_t *L)
{
    const int param_o = PARAM_o(p);
    const int param_v = PARAM_v(p);
    const int m_vec_limbs = PARAM_m_vec_limbs(p);

    int bs_mat_entries_used = 0;
    for (int r = 0; r < param_v; r++) {
        for (int c = r; c < param_v; c++) {
            if (c == r) {
                bs_mat_entries_used++;
                continue;
            }
            for (int k = 0; k < param_o; k++) {
                m_vec_mul_add_device(m_vec_limbs, P1 + m_vec_limbs * bs_mat_entries_used, O[c * param_o + k],
                                     L + m_vec_limbs * (r * param_o + k));
                m_vec_mul_add_device(m_vec_limbs, P1 + m_vec_limbs * bs_mat_entries_used, O[r * param_o + k],
                                     L + m_vec_limbs * (c * param_o + k));
            }
            bs_mat_entries_used++;
        }
    }
}

__device__ void P1_times_Vt_device(const mayo_params_t *p, const uint64_t *P1, const unsigned char *V,
                                   uint64_t *acc)
{
    mul_add_m_upper_triangular_mat_x_mat_trans_device(PARAM_m_vec_limbs(p), P1, V, acc, PARAM_v(p),
                                                      PARAM_v(p), PARAM_k(p), 1);
}

__device__ void compute_M_and_VPV_device(const mayo_params_t *p, const unsigned char *V, const uint64_t *L,
                                         const uint64_t *P1, uint64_t *Mtmp, uint64_t *A,
                                         const uint64_t *L_bs, const uint64_t *P1_bs)
{
    const int param_k = PARAM_k(p);
    const int param_v = PARAM_v(p);
    const int param_o = PARAM_o(p);
    const int m_vec_limbs = PARAM_m_vec_limbs(p);

#if defined(MAYO_BITSLICE_HYBRID)

    {
        const int mvec_bs = MAYO_HYBRID_MVEC_WORDS; // 5 words per hybrid m_vec

        uint64_t Mtmp_bs[K_MAX * O_MAX * MAYO_HYBRID_MVEC_WORDS] = {0};
        uint64_t A_bs[K_MAX * K_MAX * MAYO_HYBRID_MVEC_WORDS] = {0};

        // Mtmp = V * L
        for (int r = 0; r < param_k; r++) {
            for (int c = 0; c < param_v; c++) {
                const unsigned char s = V[r * param_v + c];
                for (int kk = 0; kk < param_o; kk++) {
                    m_vec_mul_add_hybrid_device(L_bs + (c * param_o + kk) * mvec_bs, s,
                                                Mtmp_bs + (r * param_o + kk) * mvec_bs);
                }
            }
        }

        // Row-blocked Pv = P1 * V^t folded into A = V * Pv
#ifndef MAYO_PV_ROWS
#define MAYO_PV_ROWS 16
#endif
        {
            int bs_mat_entries_used = 0;
            for (int r0 = 0; r0 < param_v; r0 += MAYO_PV_ROWS) {
                const int r1 = (r0 + MAYO_PV_ROWS < param_v) ? r0 + MAYO_PV_ROWS : param_v;
                uint64_t Pv_blk[MAYO_PV_ROWS * K_MAX * MAYO_HYBRID_MVEC_WORDS] = {0};
                for (int r = r0; r < r1; r++) {
                    for (int c = r; c < param_v; c++) {
                        for (int kk = 0; kk < param_k; kk++) {
                            m_vec_mul_add_hybrid_device(P1_bs + bs_mat_entries_used * mvec_bs,
                                                        V[kk * param_v + c],
                                                        Pv_blk + ((r - r0) * param_k + kk) * mvec_bs);
                        }
                        bs_mat_entries_used++;
                    }
                }
                for (int rr = 0; rr < param_k; rr++) {
                    for (int r = r0; r < r1; r++) {
                        const unsigned char s = V[rr * param_v + r];
                        for (int kk = 0; kk < param_k; kk++) {
                            m_vec_mul_add_hybrid_device(Pv_blk + ((r - r0) * param_k + kk) * mvec_bs, s,
                                                        A_bs + (rr * param_k + kk) * mvec_bs);
                        }
                    }
                }
            }
        }

        // un-bit-slice outputs (Mtmp: k*o m_vecs, A: k*k m_vecs)
        for (int i = 0; i < param_k * param_o; i++) {
            unbitslice_m_vec_hybrid_device(Mtmp_bs + i * mvec_bs, Mtmp + i * m_vec_limbs, PARAM_m(p),
                                           m_vec_limbs);
        }
        for (int i = 0; i < param_k * param_k; i++) {
            unbitslice_m_vec_hybrid_device(A_bs + i * mvec_bs, A + i * m_vec_limbs, PARAM_m(p), m_vec_limbs);
        }
    }
#else
    mul_add_mat_x_m_mat_device(m_vec_limbs, V, L, Mtmp, param_k, param_v, param_o);

#ifndef MAYO_PV_ROWS
#define MAYO_PV_ROWS 16
#endif
    {
        int entries_used = 0;
        for (int r0 = 0; r0 < param_v; r0 += MAYO_PV_ROWS) {
            const int r1 = (r0 + MAYO_PV_ROWS < param_v) ? r0 + MAYO_PV_ROWS : param_v;
            uint64_t Pv_blk[MAYO_PV_ROWS * K_MAX * M_VEC_LIMBS_MAX] = {0};
            for (int r = r0; r < r1; r++) {
                for (int c = r; c < param_v; c++) {
                    for (int kk = 0; kk < param_k; kk++) {
                        m_vec_mul_add_device(m_vec_limbs, P1 + entries_used * m_vec_limbs,
                                             V[kk * param_v + c],
                                             Pv_blk + ((r - r0) * param_k + kk) * m_vec_limbs);
                    }
                    entries_used++;
                }
            }
            for (int rr = 0; rr < param_k; rr++) {
                for (int r = r0; r < r1; r++) {
                    const unsigned char s = V[rr * param_v + r];
                    for (int kk = 0; kk < param_k; kk++) {
                        m_vec_mul_add_device(m_vec_limbs, Pv_blk + ((r - r0) * param_k + kk) * m_vec_limbs, s,
                                             A + (rr * param_k + kk) * m_vec_limbs);
                    }
                }
            }
        }
    }
#endif
}

__device__ void compute_P3_device(const mayo_params_t *p, const uint64_t *P1, const uint64_t *P2,
                                  const unsigned char *O, uint64_t *P3)
{
    uint64_t P1O_plus_P2[V_MAX * O_MAX * M_VEC_LIMBS_MAX] = {0};
    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    const int p2_limbs = PARAM_v(p) * PARAM_o(p) * m_vec_limbs;

    for (int i = 0; i < p2_limbs; i++) {
        P1O_plus_P2[i] = P2[i];
    }
    mul_add_m_upper_triangular_mat_x_mat_device(m_vec_limbs, P1, O, P1O_plus_P2, PARAM_v(p), PARAM_v(p),
                                                PARAM_o(p), 1);
    mul_add_mat_trans_x_m_mat_device(m_vec_limbs, O, P1O_plus_P2, P3, PARAM_v(p), PARAM_o(p), PARAM_o(p));
}

__device__ void m_upper_device(const mayo_params_t *p, const uint64_t *in, uint64_t *out, int size)
{
    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    int m_vecs_stored = 0;
    for (int r = 0; r < size; r++) {
        for (int c = r; c < size; c++) {
            m_vec_copy_device(m_vec_limbs, in + m_vec_limbs * (r * size + c),
                              out + m_vec_limbs * m_vecs_stored);
            if (r != c) {
                m_vec_add_device(m_vec_limbs, in + m_vec_limbs * (c * size + r),
                                 out + m_vec_limbs * m_vecs_stored);
            }
            m_vecs_stored++;
        }
    }
}

__device__ __forceinline__ uint64_t ct_eq_mask_device(int a, int b)
{
    const uint64_t d = (uint64_t)(uint32_t)(a ^ b);
    return (uint64_t)(-(int64_t)((d - 1) >> 63));
}
// all-ones if a > b, else 0; no branch
__device__ __forceinline__ uint64_t ct_gt_mask_device(int a, int b)
{
    return (uint64_t)(((int64_t)b - (int64_t)a) >> 63);
}

__device__ __forceinline__ unsigned char inverse_f_device(unsigned char a)
{
    // GF(16) inverse
    unsigned char a2 = mul_f_device(a, a);
    unsigned char a4 = mul_f_device(a2, a2);
    unsigned char a8 = mul_f_device(a4, a4);
    return mul_f_device(mul_f_device(a8, a4), a2);
}

__device__ int sample_solution_device(const mayo_params_t *p, unsigned char *A, const unsigned char *y,
                                      const unsigned char *r, unsigned char *x, int k, int o, int m,
                                      int A_cols)
{
    (void)p;
    const int variables = k * o;

    // Start from the random vector r
    for (int i = 0; i < k * o; i++) {
        x[i] = r[i];
    }

    // Compute A*r and put y - A*r into the augmented column
    unsigned char Ar[M_MAX] = {0};
    for (int row = 0; row < m; row++) {
        A[row * A_cols + variables] = 0;
    }
    mat_mul_device(A, r, Ar, m, A_cols, 1);
    for (int row = 0; row < m; row++) {
        A[row * A_cols + variables] = y[row] ^ Ar[row];
    }

    {
        const int row_len = (A_cols + 15) / 16;
        uint64_t packed_A[((K_MAX * O_MAX + 1 + 15) / 16) * M_MAX] = {0};

        for (int row = 0; row < m; row++) {
            for (int w = 0; w < row_len; w++) {
                uint64_t word = 0;
                for (int r = 0; r < 16; r++) {
                    const int col = 16 * w + r;
                    if (col < A_cols) {
                        word |= (uint64_t)(A[row * A_cols + col] & 0x0f) << (4 * r);
                    }
                }
                packed_A[row * row_len + w] = word;
            }
        }

        uint64_t pivot_word[(K_MAX * O_MAX + 1 + 15) / 16] = {0};
        uint64_t pivot_norm[(K_MAX * O_MAX + 1 + 15) / 16] = {0};
        // Constant-time elimination with the masks of MAYO-C EF()
        int pivot_row = 0;
        for (int pivot_col = 0; pivot_col < A_cols; pivot_col++) {
            const int pivot_row_lower_bound = MAYO_MAX(0, pivot_col + m - A_cols);
            const int pivot_row_upper_bound = MAYO_MIN(m - 1, pivot_col);
            const int search_upper_bound = MAYO_MIN(m - 1, pivot_row_upper_bound + 32);

            for (int w = 0; w < row_len; w++) {
                pivot_word[w] = 0;
                pivot_norm[w] = 0;
            }

            unsigned char pivot = 0;
            uint64_t pivot_is_zero = ~0ULL;
            for (int row = pivot_row_lower_bound; row <= search_upper_bound; row++) {
                const uint64_t take =
                    ct_eq_mask_device(row, pivot_row) | (ct_gt_mask_device(row, pivot_row) & pivot_is_zero);
                for (int w = 0; w < row_len; w++) {
                    pivot_word[w] ^= take & packed_A[row * row_len + w];
                }
                pivot = (unsigned char)((pivot_word[pivot_col >> 4] >> (4 * (pivot_col & 15))) & 0x0f);
                pivot_is_zero = ct_eq_mask_device(pivot, 0);
            }

            const unsigned char inv = inverse_f_device(pivot);
            for (int w = 0; w < row_len; w++) {
                pivot_norm[w] = gf16v_mul_u64_device(pivot_word[w], inv);
            }

            for (int row = pivot_row_lower_bound; row <= pivot_row_upper_bound; row++) {
                const uint64_t do_copy = ct_eq_mask_device(row, pivot_row) & ~pivot_is_zero;
                for (int w = 0; w < row_len; w++) {
                    packed_A[row * row_len + w] =
                        (packed_A[row * row_len + w] & ~do_copy) | (pivot_norm[w] & do_copy);
                }
            }

            for (int row = pivot_row_lower_bound; row < m; row++) {
                const unsigned char factor =
                    (unsigned char)(ct_gt_mask_device(row, pivot_row) &
                                    ((packed_A[row * row_len + (pivot_col >> 4)] >> (4 * (pivot_col & 15))) &
                                     0x0f));
                for (int w = 0; w < row_len; w++) {
                    packed_A[row * row_len + w] ^= gf16v_mul_u64_device(pivot_norm[w], factor);
                }
            }

            pivot_row += (int)(1 & ~pivot_is_zero);
        }

        for (int row = 0; row < m; row++) {
            for (int w = 0; w < row_len; w++) {
                const uint64_t word = packed_A[row * row_len + w];
                for (int r = 0; r < 16; r++) {
                    const int col = 16 * w + r;
                    if (col < A_cols) {
                        A[row * A_cols + col] = (unsigned char)((word >> (4 * r)) & 0x0f);
                    }
                }
            }
        }
    }

    unsigned char full_rank = 0;
    for (int col = 0; col < A_cols - 1; col++) {
        full_rank |= A[(m - 1) * A_cols + col];
    }
    if (full_rank == 0) {
        return 0;
    }

    // Constant-time back-substitution (MAYO-C sample_solution)
    for (int row = m - 1; row >= 0; row--) {
        unsigned char finished = 0;
        const int col_upper_bound = MAYO_MIN(row + (32 / (m - row)), variables);
        for (int col = row; col <= col_upper_bound; col++) {
            const unsigned char nz =
                (unsigned char)(-(int)((((unsigned)A[row * A_cols + col] & 0x0f) + 0x0f) >> 4));
            const unsigned char correct = nz & (unsigned char)~finished;
            const unsigned char u = correct & (A[row * A_cols + A_cols - 1] & 0x0f);
            x[col] ^= u;
            for (int i = 0; i < row; i++) {
                A[i * A_cols + A_cols - 1] ^= mul_f_device(u, A[i * A_cols + col]);
            }
            finished |= correct;
        }
    }
    return 1;
}

__device__ void eval_public_map_device(const mayo_params_t *p, const unsigned char *s, const uint64_t *P1,
                                       const uint64_t *P2, const uint64_t *P3, unsigned char *eval,
                                       const uint64_t *P1_bs, const uint64_t *P2_bs, const uint64_t *P3_bs);

__device__ int mayo_verify_device(const mayo_params_t *p, const unsigned char *m, size_t mlen,
                                  const unsigned char *sig, const uint64_t *expanded_pk,
                                  const uint64_t *P1_bs, const uint64_t *P2_bs, const uint64_t *P3_bs)
{
    unsigned char tEnc[M_BYTES_MAX];
    unsigned char t[M_MAX];
    unsigned char y[2 * M_MAX] = {0}; // extra space for reduction mod f(X)
    unsigned char s[K_MAX * N_MAX];
    unsigned char tmp[DIGEST_BYTES_MAX + SALT_BYTES_MAX];

    const int param_m = PARAM_m(p);
    const int param_n = PARAM_n(p);
    const int param_k = PARAM_k(p);
    const int param_m_bytes = PARAM_m_bytes(p);
    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int param_digest_bytes = PARAM_digest_bytes(p);
    const int param_salt_bytes = PARAM_salt_bytes(p);

    const uint64_t *P1 = expanded_pk;
    const uint64_t *P2 = P1 + PARAM_P1_limbs(p);
    const uint64_t *P3 = P2 + PARAM_P2_limbs(p);

    // hash m
    shake256_device(tmp, param_digest_bytes, m, mlen);

    // compute t = shake256(digest || salt), then decode nibbles
    memcpy(tmp + param_digest_bytes, sig + param_sig_bytes - param_salt_bytes, param_salt_bytes);
    shake256_device(tEnc, param_m_bytes, tmp, param_digest_bytes + param_salt_bytes);
    decode_device(tEnc, t, param_m);

    // decode s from the signature
    decode_device(sig, s, param_k * param_n);

    // y = eval_public_map(s, P1, P2, P3)
    eval_public_map_device(p, s, P1, P2, P3, y, P1_bs, P2_bs, P3_bs);

    int diff = 0;
    for (int i = 0; i < param_m; i++) {
        diff |= (y[i] ^ t[i]);
    }
    if (diff == 0) {
        return MAYO_OK; // good signature
    }
    return MAYO_ERR; // bad signature
}

__device__ void compute_rhs_device(const mayo_params_t *p, uint64_t *vPv, const unsigned char *t,
                                   unsigned char *y)
{
    const size_t top_pos = ((PARAM_m(p) - 1) % 16) * 4;
    const size_t m_vec_limbs = PARAM_m_vec_limbs(p);

    if (PARAM_m(p) % 16 != 0) {
        uint64_t mask = 1;
        mask <<= ((PARAM_m(p) % 16) * 4);
        mask -= 1;
        for (int i = 0; i < PARAM_k(p) * PARAM_k(p); i++) {
            vPv[i * m_vec_limbs + m_vec_limbs - 1] &= mask;
        }
    }

    const uint64_t packed_tail = (uint64_t)PARAM_f_tail(p)[0] | ((uint64_t)PARAM_f_tail(p)[1] << 4) |
                                 ((uint64_t)PARAM_f_tail(p)[2] << 8) | ((uint64_t)PARAM_f_tail(p)[3] << 12);

    uint64_t temp[M_VEC_LIMBS_MAX] = {0};
    unsigned char *temp_bytes = (unsigned char *)temp;
    for (int i = PARAM_k(p) - 1; i >= 0; i--) {
        for (int j = i; j < PARAM_k(p); j++) {
            unsigned char top = (temp[m_vec_limbs - 1] >> top_pos) % 16;
            temp[m_vec_limbs - 1] <<= 4;
            for (int k = m_vec_limbs - 2; k >= 0; k--) {
                temp[k + 1] ^= temp[k] >> 60;
                temp[k] <<= 4;
            }
            {
                // one vector multiply yields all 4 products (nibbles 0..3)
                const uint64_t prod = gf16v_mul_u64_device(packed_tail, top);
                temp_bytes[0] ^= (unsigned char)(prod & 0xFF);
                temp_bytes[1] ^= (unsigned char)((prod >> 8) & 0xFF);
            }
            for (size_t k = 0; k < m_vec_limbs; k++) {
                temp[k] ^= vPv[(i * PARAM_k(p) + j) * m_vec_limbs + k] ^
                           ((i != j) * vPv[(j * PARAM_k(p) + i) * m_vec_limbs + k]);
            }
        }
    }

    for (int i = 0; i < PARAM_m(p); i += 2) {
        y[i] = t[i] ^ (temp_bytes[i / 2] & 0xF);
        y[i + 1] = t[i + 1] ^ (temp_bytes[i / 2] >> 4);
    }
}

__device__ void transpose_16x16_nibbles_device(uint64_t *M)
{
    const uint64_t even_nibbles = 0x0f0f0f0f0f0f0f0f;
    const uint64_t even_bytes = 0x00ff00ff00ff00ff;
    const uint64_t even_2bytes = 0x0000ffff0000ffff;
    const uint64_t even_half = 0x00000000ffffffff;

    for (size_t i = 0; i < 16; i += 2) {
        uint64_t t = ((M[i] >> 4) ^ M[i + 1]) & even_nibbles;
        M[i] ^= t << 4;
        M[i + 1] ^= t;
    }
    for (size_t i = 0; i < 16; i += 4) {
        uint64_t t0 = ((M[i] >> 8) ^ M[i + 2]) & even_bytes;
        uint64_t t1 = ((M[i + 1] >> 8) ^ M[i + 3]) & even_bytes;
        M[i] ^= (t0 << 8);
        M[i + 1] ^= (t1 << 8);
        M[i + 2] ^= t0;
        M[i + 3] ^= t1;
    }
    for (size_t i = 0; i < 4; i++) {
        uint64_t t0 = ((M[i] >> 16) ^ M[i + 4]) & even_2bytes;
        uint64_t t1 = ((M[i + 8] >> 16) ^ M[i + 12]) & even_2bytes;
        M[i] ^= t0 << 16;
        M[i + 8] ^= t1 << 16;
        M[i + 4] ^= t0;
        M[i + 12] ^= t1;
    }
    for (size_t i = 0; i < 8; i++) {
        uint64_t t = ((M[i] >> 32) ^ M[i + 8]) & even_half;
        M[i] ^= t << 32;
        M[i + 8] ^= t;
    }
}

#ifndef MAYO_M_OVER_8
#define MAYO_M_OVER_8 ((M_MAX + 7) / 8)
#endif

__device__ void compute_A_device(const mayo_params_t *p, uint64_t *VtL, unsigned char *A_out)
{
    int bits_to_shift = 0;
    int words_to_shift = 0;
    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    uint64_t A[(((O_MAX * K_MAX + 15) / 16) * 16) * MAYO_M_OVER_8] = {0};
    size_t A_width = ((PARAM_o(p) * PARAM_k(p) + 15) / 16) * 16;
    const uint64_t *Mi, *Mj;

    if (PARAM_m(p) % 16 != 0) {
        uint64_t mask = 1;
        mask <<= (PARAM_m(p) % 16) * 4;
        mask -= 1;
        for (int i = 0; i < PARAM_o(p) * PARAM_k(p); i++) {
            VtL[i * m_vec_limbs + m_vec_limbs - 1] &= mask;
        }
    }

    for (int i = 0; i <= PARAM_k(p) - 1; ++i) {
        for (int j = PARAM_k(p) - 1; j >= i; --j) {
            Mj = VtL + j * m_vec_limbs * PARAM_o(p);
            for (int c = 0; c < PARAM_o(p); c++) {
                for (int k = 0; k < m_vec_limbs; k++) {
                    A[PARAM_o(p) * i + c + (k + words_to_shift) * A_width] ^= Mj[k + c * m_vec_limbs]
                                                                              << bits_to_shift;
                    if (bits_to_shift > 0) {
                        A[PARAM_o(p) * i + c + (k + words_to_shift + 1) * A_width] ^=
                            Mj[k + c * m_vec_limbs] >> (64 - bits_to_shift);
                    }
                }
            }
            if (i != j) {
                Mi = VtL + i * m_vec_limbs * PARAM_o(p);
                for (int c = 0; c < PARAM_o(p); c++) {
                    for (int k = 0; k < m_vec_limbs; k++) {
                        A[PARAM_o(p) * j + c + (k + words_to_shift) * A_width] ^= Mi[k + c * m_vec_limbs]
                                                                                  << bits_to_shift;
                        if (bits_to_shift > 0) {
                            A[PARAM_o(p) * j + c + (k + words_to_shift + 1) * A_width] ^=
                                Mi[k + c * m_vec_limbs] >> (64 - bits_to_shift);
                        }
                    }
                }
            }
            bits_to_shift += 4;
            if (bits_to_shift == 64) {
                words_to_shift++;
                bits_to_shift = 0;
            }
        }
    }

    for (size_t c = 0; c < A_width * ((PARAM_m(p) + (PARAM_k(p) + 1) * PARAM_k(p) / 2 + 15) / 16); c += 16) {
        transpose_16x16_nibbles_device(A + c);
    }

    unsigned char tab[F_TAIL_LEN * 4] = {0};
    for (size_t i = 0; i < F_TAIL_LEN; i++) {
        tab[4 * i] = mul_f_device(PARAM_f_tail(p)[i], 1);
        tab[4 * i + 1] = mul_f_device(PARAM_f_tail(p)[i], 2);
        tab[4 * i + 2] = mul_f_device(PARAM_f_tail(p)[i], 4);
        tab[4 * i + 3] = mul_f_device(PARAM_f_tail(p)[i], 8);
    }

    uint64_t low_bit_in_nibble = 0x1111111111111111;
    for (size_t c = 0; c < A_width; c += 16) {
        for (int r = PARAM_m(p); r < PARAM_m(p) + (PARAM_k(p) + 1) * PARAM_k(p) / 2; r++) {
            size_t pos = (r / 16) * A_width + c + (r % 16);
            uint64_t t0 = A[pos] & low_bit_in_nibble;
            uint64_t t1 = (A[pos] >> 1) & low_bit_in_nibble;
            uint64_t t2 = (A[pos] >> 2) & low_bit_in_nibble;
            uint64_t t3 = (A[pos] >> 3) & low_bit_in_nibble;
            for (size_t t = 0; t < F_TAIL_LEN; t++) {
                A[((r + t - PARAM_m(p)) / 16) * A_width + c + ((r + t - PARAM_m(p)) % 16)] ^=
                    t0 * tab[4 * t + 0] ^ t1 * tab[4 * t + 1] ^ t2 * tab[4 * t + 2] ^ t3 * tab[4 * t + 3];
            }
        }
    }

    for (int r = 0; r < PARAM_m(p); r += 16) {
        for (int c = 0; c < PARAM_A_cols(p) - 1; c += 16) {
            for (int i = 0; i + r < PARAM_m(p); i++) {
                decode_device((unsigned char *)&A[r * A_width / 16 + c + i],
                              A_out + PARAM_A_cols(p) * (r + i) + c, MAYO_MIN(16, PARAM_A_cols(p) - 1 - c));
            }
        }
    }
}

__device__ void unpack_m_vecs_device(const unsigned char *in, uint64_t *out, int vecs, int m)
{
    const int m_vec_limbs = (m + 15) / 16;
    unsigned char *_out = (unsigned char *)out;
    uint64_t tmp[(M_MAX + 15) / 16] = {0};
    for (int i = vecs - 1; i >= 0; i--) {
        memcpy(tmp, in + i * m / 2, m / 2);
        memcpy(_out + i * m_vec_limbs * sizeof(uint64_t), tmp, m_vec_limbs * sizeof(uint64_t));
    }
}

__device__ void pack_m_vecs_device(const uint64_t *in, unsigned char *out, int vecs, int m)
{
    const int m_vec_limbs = (m + 15) / 16;
    unsigned char *_in = (unsigned char *)in;
    for (int i = 0; i < vecs; i++) {
        memcpy(out + (i * m / 2), _in + i * m_vec_limbs * sizeof(uint64_t), m / 2);
    }
}

__device__ void expand_P1_P2_device(const mayo_params_t *p, uint64_t *P, const unsigned char *seed_pk)
{
    PK_PRF_device((unsigned char *)P, PARAM_P1_bytes(p) + PARAM_P2_bytes(p), seed_pk, PARAM_pk_seed_bytes(p));
    unpack_m_vecs_device((unsigned char *)P, P,
                         (PARAM_P1_limbs(p) + PARAM_P2_limbs(p)) / PARAM_m_vec_limbs(p), PARAM_m(p));
}

__device__ void m_calculate_PS_SPS_device(const mayo_params_t *p, const uint64_t *P1, const uint64_t *P2,
                                          const uint64_t *P3, const unsigned char *S, uint64_t *SPS)
{
    const int param_m = PARAM_m(p);
    const int param_v = PARAM_v(p);
    const int param_o = PARAM_o(p);
    const int param_k = PARAM_k(p);
    const int param_n = param_o + param_v;
    const int m_vec_limbs = (param_m + 15) / 16;

    uint64_t PS[N_MAX * M_VEC_LIMBS_MAX];

    for (int col = 0; col < param_k; col++) {
        const unsigned char *s = S + col * param_n;
        int P1_used = 0, P3_used = 0;
        for (int row = 0; row < param_n; row++) {
            uint64_t acc[M_VEC_LIMBS_MAX] = {0};
            if (row < param_v) {
                for (int j = row; j < param_v; j++) {
                    m_vec_mul_add_device(m_vec_limbs, P1 + (P1_used++) * m_vec_limbs, s[j], acc);
                }
                for (int j = 0; j < param_o; j++) {
                    m_vec_mul_add_device(m_vec_limbs, P2 + (row * param_o + j) * m_vec_limbs, s[param_v + j],
                                         acc);
                }
            } else {
                for (int j = row; j < param_n; j++) {
                    m_vec_mul_add_device(m_vec_limbs, P3 + (P3_used++) * m_vec_limbs, s[j], acc);
                }
            }
            for (int w = 0; w < m_vec_limbs; w++) {
                PS[row * m_vec_limbs + w] = acc[w];
            }
        }
        for (int row = 0; row < param_k; row++) {
            uint64_t acc[M_VEC_LIMBS_MAX] = {0};
            for (int j = 0; j < param_n; j++) {
                m_vec_mul_add_device(m_vec_limbs, PS + j * m_vec_limbs, S[row * param_n + j], acc);
            }
            for (int w = 0; w < m_vec_limbs; w++) {
                SPS[(row * param_k + col) * m_vec_limbs + w] = acc[w];
            }
        }
    }
}

__device__ void eval_public_map_device(const mayo_params_t *p, const unsigned char *s, const uint64_t *P1,
                                       const uint64_t *P2, const uint64_t *P3, unsigned char *eval,
                                       const uint64_t *P1_bs, const uint64_t *P2_bs, const uint64_t *P3_bs)
{
    uint64_t SPS[K_MAX * K_MAX * M_VEC_LIMBS_MAX] = {0};
#ifdef MAYO_BITSLICE_SPS

    uint64_t SPS_bs[K_MAX * K_MAX * MAYO_HYBRID_MVEC_WORDS] = {0};
    m_calculate_PS_SPS_device_h(p, P1_bs, P2_bs, P3_bs, s, SPS_bs);
    {
        const int m_vec_limbs = (PARAM_m(p) + 15) / 16;
        for (int i = 0; i < PARAM_k(p) * PARAM_k(p); i++) {
            unbitslice_m_vec_hybrid_device(SPS_bs + i * MAYO_HYBRID_MVEC_WORDS, SPS + i * m_vec_limbs,
                                           PARAM_m(p), m_vec_limbs);
        }
    }
#else
    m_calculate_PS_SPS_device(p, P1, P2, P3, s, SPS);
#endif
    unsigned char zero[M_MAX] = {0};
    compute_rhs_device(p, SPS, zero, eval);
}

__device__ int mayo_keypair_device(const mayo_params_t *p, unsigned char *pk, unsigned char *sk)
{
    int ret = mayo_keypair_compact_device(p, pk, sk);
    return ret;
}

__device__ int mayo_expand_sk_device(const mayo_params_t *p, const unsigned char *csk, sk_t *sk)
{
    int ret = MAYO_OK;
    unsigned char S[PK_SEED_BYTES_MAX + O_BYTES_MAX];
    uint64_t *P = sk->p;
    unsigned char *O = sk->O;

    const int param_o = PARAM_o(p);
    const int param_v = PARAM_v(p);
    const int param_O_bytes = PARAM_O_bytes(p);
    const int param_pk_seed_bytes = PARAM_pk_seed_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);

    shake256_device(S, param_pk_seed_bytes + param_O_bytes, csk, param_sk_seed_bytes);
    decode_device(S + param_pk_seed_bytes, O, param_v * param_o);

    expand_P1_P2_device(p, P, S);

    uint64_t *P2 = P + PARAM_P1_limbs(p);
    uint64_t *P1 = P;
    uint64_t *L = P2;

    P1P1t_times_O_device(p, P1, O, L);
    memset(S, 0, PK_SEED_BYTES_MAX + O_BYTES_MAX);

    return ret;
}

#define MAYO_RETRY 2

__device__ int mayo_sign_signature_device(const mayo_params_t *p, unsigned char *sig, size_t *siglen,
                                          const unsigned char *m, size_t mlen, const unsigned char *csk,
                                          const sk_t *expanded_sk, const uint64_t *P1_bs,
                                          const uint64_t *L_bs, int ctr_lo, int ctr_hi)
{
    int ret = MAYO_OK;
    unsigned char tenc[M_BYTES_MAX], t[M_MAX];
    unsigned char y[M_MAX];
    unsigned char salt[SALT_BYTES_MAX];
    unsigned char V[K_MAX * V_BYTES_MAX + R_BYTES_MAX], Vdec[V_MAX * K_MAX];
    unsigned char A[((M_MAX + 7) / 8 * 8) * (K_MAX * O_MAX + 1)] = {0};
    unsigned char x[K_MAX * N_MAX];
    unsigned char r[K_MAX * O_MAX + 1] = {0};
    unsigned char s[K_MAX * N_MAX];

    unsigned char Ox[V_MAX];
    unsigned char tmp[DIGEST_BYTES_MAX + SALT_BYTES_MAX + SK_SEED_BYTES_MAX + 1];
    unsigned char *ctrbyte;
    unsigned char *vi;
    int sol_found = 0;

    const int param_m = PARAM_m(p);
    const int param_n = PARAM_n(p);
    const int param_o = PARAM_o(p);
    const int param_k = PARAM_k(p);
    const int param_v = PARAM_v(p);
    const int param_m_bytes = PARAM_m_bytes(p);
    const int param_v_bytes = PARAM_v_bytes(p);
    const int param_r_bytes = PARAM_r_bytes(p);
    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int param_A_cols = PARAM_A_cols(p);
    const int param_digest_bytes = PARAM_digest_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);
    const int param_salt_bytes = PARAM_salt_bytes(p);

    shake256_device(tmp, param_digest_bytes, m, mlen);

    const uint64_t *P1 = expanded_sk->p;
    const uint64_t *L = P1 + PARAM_P1_limbs(p);
    uint64_t Mtmp[K_MAX * O_MAX * M_VEC_LIMBS_MAX] = {0};

    for (int i = 0; i < param_salt_bytes; i++) {
        tmp[param_digest_bytes + i] = 1;
    }

    memcpy(tmp + param_digest_bytes + param_salt_bytes, csk, param_sk_seed_bytes);
    shake256_device(salt, param_salt_bytes, tmp, param_digest_bytes + param_salt_bytes + param_sk_seed_bytes);

    memcpy(tmp + param_digest_bytes, salt, param_salt_bytes);
    ctrbyte = tmp + param_digest_bytes + param_salt_bytes + param_sk_seed_bytes;

    shake256_device(tenc, param_m_bytes, tmp, param_digest_bytes + param_salt_bytes);
    decode_device(tenc, t, param_m);

    for (int ctr = ctr_lo; ctr <= ctr_hi; ++ctr) {
        *ctrbyte = (unsigned char)ctr;

        shake256_device(V, param_k * param_v_bytes + param_r_bytes, tmp,
                        param_digest_bytes + param_salt_bytes + param_sk_seed_bytes + 1);

        for (int i = 0; i <= param_k - 1; ++i) {
            decode_device(V + i * param_v_bytes, Vdec + i * param_v, param_v);
        }

        compute_M_and_VPV_device(p, Vdec, L, P1, Mtmp, (uint64_t *)A, L_bs, P1_bs);
        compute_rhs_device(p, (uint64_t *)A, t, y);
        compute_A_device(p, Mtmp, A);

        for (int i = 0; i < param_m; i++) {
            A[(1 + i) * (param_k * param_o + 1) - 1] = 0;
        }

        decode_device(V + param_k * param_v_bytes, r, param_k * param_o);

        if (sample_solution_device(p, A, y, r, x, param_k, param_o, param_m, param_A_cols)) {
            sol_found = 1;
            break;
        } else {
            memset(Mtmp, 0, sizeof(Mtmp));
            memset(A, 0, sizeof(A));
        }
    }

    if (!sol_found) {
        ret = (ctr_hi < 255) ? MAYO_RETRY : MAYO_ERR;
        goto err;
    }

    for (int i = 0; i <= param_k - 1; ++i) {
        vi = Vdec + i * (param_n - param_o);
        mat_mul_device(expanded_sk->O, x + i * param_o, Ox, param_n - param_o, param_o, 1);
        mat_add_device(vi, Ox, s + i * param_n, param_n - param_o, 1);
        memcpy(s + i * param_n + (param_n - param_o), x + i * param_o, param_o);
    }

    encode_device(s, sig, param_n * param_k);
    memcpy(sig + param_sig_bytes - param_salt_bytes, salt, param_salt_bytes);
    *siglen = param_sig_bytes;

err:
    memset(V, 0, sizeof(V));
    memset(Vdec, 0, sizeof(Vdec));
    memset(A, 0, sizeof(A));
    memset(r, 0, sizeof(r));
    memset(Ox, 0, sizeof(Ox));
    memset(tmp, 0, sizeof(tmp));
    memset(Mtmp, 0, sizeof(Mtmp));

    return ret;
}

// One signing pass
__global__ void mayo_sign_kernel(const mayo_params_t *p, unsigned char *sm_all, size_t *smlen_all,
                                 const unsigned char *m_all, size_t mlen, const unsigned char *csk,
                                 const sk_t *expanded_sk, const uint64_t *P1_bs, const uint64_t *L_bs,
                                 int *ret_val_all, int sm_stride, int batch_size, const int *queue_in,
                                 const int *count_in, int *queue_out, int *count_out, int ctr_lo, int ctr_hi)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = tid;
    if (queue_in) {
        if (tid >= *count_in) {
            return;
        }
        slot = queue_in[tid];
    } else if (tid >= batch_size) {
        return;
    }

    const int param_sig_bytes = PARAM_sig_bytes(p);
    unsigned char *my_sm = sm_all + (size_t)slot * sm_stride;
    const unsigned char *my_m = m_all + (size_t)slot * mlen;
    for (size_t i = 0; i < mlen; i++) {
        my_sm[param_sig_bytes + i] = my_m[i];
    }

    size_t my_siglen = 0;
    int ret = mayo_sign_signature_device(p, my_sm, &my_siglen, my_sm + param_sig_bytes, mlen, csk,
                                         expanded_sk, P1_bs, L_bs, ctr_lo, ctr_hi);
    if (ret == MAYO_RETRY) {
        queue_out[atomicAdd(count_out, 1)] = slot;
        ret_val_all[slot] = MAYO_ERR;
        return;
    }
    if (ret != MAYO_OK || my_siglen != (size_t)param_sig_bytes) {
        for (size_t i = 0; i < param_sig_bytes + mlen; i++) {
            my_sm[i] = 0;
        }
        ret_val_all[slot] = MAYO_ERR;
        return;
    }
    smlen_all[slot] = my_siglen + mlen;
    ret_val_all[slot] = ret;
}

#define SIGN_PASSES 4
static int *d_sign_queue = NULL, *d_sign_count = NULL;
static int sign_queue_cap = 0;

static void launch_sign(const mayo_params_t *d_params, unsigned char *d_sm_all, size_t *d_smlen_all,
                        const unsigned char *d_m_all, size_t msg_bytes, const unsigned char *d_csk,
                        const sk_t *d_expanded_sk, const uint64_t *d_P1_bs, const uint64_t *d_L_bs,
                        int *d_ret_all, int sm_stride, int batch)
{
    if (batch > sign_queue_cap) {
        cudaFree(d_sign_queue);
        cudaFree(d_sign_count);
        sign_queue_cap = batch > BATCH ? batch : BATCH;
        cudaMalloc(&d_sign_queue, 2 * (size_t)sign_queue_cap * sizeof(int));
        cudaMalloc(&d_sign_count, SIGN_PASSES * sizeof(int));
    }
    cudaMemsetAsync(d_sign_count, 0, SIGN_PASSES * sizeof(int), 0);
    const int grid = (batch + BLOCK - 1) / BLOCK;
    int *queue[2] = {d_sign_queue, d_sign_queue + sign_queue_cap};
    for (int pass = 0; pass < SIGN_PASSES; pass++) {
        const int last = (pass == SIGN_PASSES - 1);
        mayo_sign_kernel<<<grid, BLOCK>>>(
            d_params, d_sm_all, d_smlen_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs, d_L_bs,
            d_ret_all, sm_stride, batch, pass ? queue[(pass - 1) & 1] : NULL,
            pass ? d_sign_count + pass - 1 : NULL, last ? NULL : queue[pass & 1],
            last ? NULL : d_sign_count + pass, pass, last ? 255 : pass);
    }
}

// Stream mode
__global__ void mayo_sign_stream_kernel(const mayo_params_t *p, unsigned char *sm_all,
                                        const unsigned char *m_all, size_t mlen, const unsigned char *csk,
                                        const sk_t *expanded_sk, const uint64_t *P1_bs, const uint64_t *L_bs,
                                        int sm_stride, int n_new, int n_carry, int cap,
                                        const unsigned char *cm_in, const unsigned char *cctr_in,
                                        unsigned char *csm_out, int *cret_out, unsigned char *cm_out,
                                        unsigned char *cctr_out, int *count_out, int *done)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_carry + n_new) {
        return;
    }

    const int sig_bytes = PARAM_sig_bytes(p);
    const unsigned char *my_m;
    unsigned char *my_sm;
    int ctr = 0;
    if (tid < n_carry) {
        my_m = cm_in + (size_t)tid * mlen;
        my_sm = csm_out + (size_t)tid * sm_stride;
        ctr = cctr_in[tid];
    } else {
        my_m = m_all + (size_t)(tid - n_carry) * mlen;
        my_sm = sm_all + (size_t)(tid - n_carry) * sm_stride;
    }
    for (size_t i = 0; i < mlen; i++) {
        my_sm[sig_bytes + i] = my_m[i];
    }

    size_t siglen = 0;
    int ret = mayo_sign_signature_device(p, my_sm, &siglen, my_sm + sig_bytes, mlen, csk, expanded_sk, P1_bs,
                                         L_bs, ctr, ctr);
    if (ret == MAYO_RETRY) {
        int k = atomicAdd(count_out, 1);
        if (k < cap) {
            for (size_t i = 0; i < mlen; i++) {
                cm_out[(size_t)k * mlen + i] = my_m[i];
            }
            cctr_out[k] = (unsigned char)(ctr + 1);
        }
        ret = MAYO_ERR;
    } else if (ret == MAYO_OK && siglen == (size_t)sig_bytes) {
        atomicAdd(done, 1);
    } else {
        ret = MAYO_ERR;
    }
    if (tid < n_carry) {
        cret_out[tid] = ret;
    }
}

static unsigned char *s_cm[2], *s_cctr[2], *s_csm;
static int *s_cret, *s_count, *s_done;
static int s_cap = 0;

static void stream_alloc(int cap, size_t msg_bytes, int sm_stride)
{
    if (cap <= s_cap) {
        return;
    }
    for (int i = 0; i < 2; i++) {
        cudaFree(s_cm[i]);
        cudaFree(s_cctr[i]);
        cudaMalloc(&s_cm[i], (size_t)cap * msg_bytes);
        cudaMalloc(&s_cctr[i], cap);
    }
    cudaFree(s_csm);
    cudaFree(s_cret);
    cudaFree(s_count);
    cudaFree(s_done);
    cudaMalloc(&s_csm, (size_t)cap * sm_stride);
    cudaMalloc(&s_cret, cap * sizeof(int));
    cudaMalloc(&s_count, 2 * sizeof(int));
    cudaMalloc(&s_done, sizeof(int));
    s_cap = cap;
}

// One stream launch
static int stream_launch(const mayo_params_t *d_params, unsigned char *d_sm_all, const unsigned char *d_m_all,
                         size_t msg_bytes, const unsigned char *d_csk, const sk_t *d_expanded_sk,
                         const uint64_t *d_P1_bs, const uint64_t *d_L_bs, int sm_stride, int n_new,
                         int n_carry, int *cur, unsigned char *h_sm, unsigned char *h_csm, bool copies)
{
    const int nxt = *cur ^ 1;
    const int n = n_carry + n_new;
    cudaMemsetAsync(s_count + nxt, 0, sizeof(int), 0);
    if (n > 0) {
        mayo_sign_stream_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs, d_L_bs, sm_stride, n_new,
            n_carry, s_cap, s_cm[*cur], s_cctr[*cur], s_csm, s_cret, s_cm[nxt], s_cctr[nxt], s_count + nxt,
            s_done);
    }
    if (copies) {
        if (n_new) {
            cudaMemcpyAsync(h_sm, d_sm_all, (size_t)n_new * sm_stride, cudaMemcpyDeviceToHost, 0);
        }
        if (n_carry) {
            cudaMemcpyAsync(h_csm, s_csm, (size_t)n_carry * sm_stride, cudaMemcpyDeviceToHost, 0);
        }
    }
    int carried = 0;
    cudaMemcpy(&carried, s_count + nxt, sizeof(int), cudaMemcpyDeviceToHost);
    *cur = nxt;
    return carried < s_cap ? carried : s_cap;
}

__global__ void mayo_open_kernel(const mayo_params_t *p, unsigned char *m, size_t *mlen,
                                 const unsigned char *sm, size_t smlen, const uint64_t *expanded_pk,
                                 const uint64_t *P1_bs, const uint64_t *P2_bs, const uint64_t *P3_bs,
                                 int *ret_val)
{
    const int param_sig_bytes = PARAM_sig_bytes(p);
    if (smlen < (size_t)param_sig_bytes) {
        *ret_val = MAYO_ERR;
        return;
    }

    int result = mayo_verify_device(p, sm + param_sig_bytes, smlen - param_sig_bytes, sm, expanded_pk, P1_bs,
                                    P2_bs, P3_bs);

    if (result == MAYO_OK) {
        *mlen = smlen - param_sig_bytes;
        for (size_t i = 0; i < *mlen; i++) {
            m[i] = sm[param_sig_bytes + i];
        }
    }
    *ret_val = result;
}

// Batched GPU signature verification
__global__ void mayo_verify_kernel(const mayo_params_t *p, const unsigned char *sm_all, size_t smlen,
                                   const uint64_t *expanded_pk, const uint64_t *pk_P1_bs,
                                   const uint64_t *pk_P2_bs, const uint64_t *pk_P3_bs, int *ret_val_all,
                                   int sm_stride, int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) {
        return;
    }

    const int param_sig_bytes = PARAM_sig_bytes(p);
    const unsigned char *my_sm = sm_all + tid * sm_stride;
    const size_t my_mlen = smlen - (size_t)param_sig_bytes;

    if (smlen < (size_t)param_sig_bytes) {
        ret_val_all[tid] = MAYO_ERR;
        return;
    }

    ret_val_all[tid] = mayo_verify_device(p, my_sm + param_sig_bytes, my_mlen, my_sm, expanded_pk, pk_P1_bs,
                                          pk_P2_bs, pk_P3_bs);
}

__global__ void mayo_keypair_compact_kernel(const mayo_params_t *p, unsigned char *cpk, unsigned char *csk,
                                            int *ret_val)
{
    int ret = MAYO_OK;
    unsigned char *seed_sk = csk;
    unsigned char S[PK_SEED_BYTES_MAX + O_BYTES_MAX];
    uint64_t P[P1_LIMBS_MAX + P2_LIMBS_MAX];
    uint64_t P3[O_MAX * O_MAX * M_VEC_LIMBS_MAX] = {0};

    unsigned char *seed_pk;
    unsigned char O[(V_MAX)*O_MAX];

    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    const int param_m = PARAM_m(p);
    const int param_v = PARAM_v(p);
    const int param_o = PARAM_o(p);
    const int param_O_bytes = PARAM_O_bytes(p);
    const int param_P1_limbs = PARAM_P1_limbs(p);
    const int param_P3_limbs = PARAM_P3_limbs(p);
    const int param_pk_seed_bytes = PARAM_pk_seed_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);

    uint64_t *P1 = P;
    uint64_t *P2 = P + param_P1_limbs;

    for (int i = 0; i < param_sk_seed_bytes; i++) {
        seed_sk[i] = 1;
    }

    shake256_device(S, param_pk_seed_bytes + param_O_bytes, seed_sk, param_sk_seed_bytes);
    seed_pk = S;
    decode_device(S + param_pk_seed_bytes, O, param_v * param_o);

    expand_P1_P2_device(p, P, seed_pk);
    compute_P3_device(p, P1, P2, O, P3);

    for (int i = 0; i < param_pk_seed_bytes; i++) {
        cpk[i] = seed_pk[i];
    }

    uint64_t P3_upper[P3_LIMBS_MAX];
    m_upper_device(p, P3, P3_upper, param_o);
    pack_m_vecs_device(P3_upper, cpk + param_pk_seed_bytes, param_P3_limbs / m_vec_limbs, param_m);

    *ret_val = ret;
}

// Benchmark batch-size / iteration knobs
#define MAX_BATCHES 16

static int bench_batch_list(int *out, int max_out)
{
    const char *env = getenv("MAYO_BENCH_BATCHES");
    int n = 0;
    if (env && *env) {
        char buf[256];
        strncpy(buf, env, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *tok = strtok(buf, ",");
        while (tok && n < max_out) {
            const int b = atoi(tok);
            if (b > 0 && b <= BATCH) {
                out[n++] = b;
            }
            tok = strtok(NULL, ",");
        }
    }
    if (n == 0) {
        static const int def[] = {1, 16, 256, 4096, 16384, 131072};
        n = (int)(sizeof(def) / sizeof(def[0]));
        if (n > max_out) {
            n = max_out;
        }
        for (int i = 0; i < n; i++) {
            out[i] = def[i];
        }
    }
    return n;
}

static int bench_iterations(const char *env_name, int default_n)
{
    const char *env = getenv(env_name);
    if (env && *env) {
        const int v = atoi(env);
        if (v > 0) {
            return v;
        }
    }
    return default_n;
}

static int distinct_msgs_enabled(void)
{
    const char *env = getenv("MAYO_DISTINCT_MSGS");
    return (env != NULL && atoi(env) > 0);
}

// Fill one message buffer
static void fill_distinct_message(unsigned char *out, const unsigned char *base, int msg_bytes, int index)
{
    memcpy(out, base, (size_t)msg_bytes);
    if (index == 0) {
        return; // element 0 keeps the exact base message
    }
    uint32_t x = (uint32_t)(index + 1) * 0x9E3779B9u;
    for (int j = 0; j < 4; j++) {
        out[j] ^= (unsigned char)(x >> (8 * j));
    }
    out[4] ^= (unsigned char)(index >> 8);
    out[5] ^= (unsigned char)(index >> 16);
}

static void mayo_sign_impl(const mayo_params_t *p, const uint8_t *msg, const uint8_t *pk, const uint8_t *sk)
{
    // The public key is not needed for signing itself
    (void)pk;

    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int msg_bytes = 32;
    const int sm_stride = param_sig_bytes + msg_bytes;
    unsigned char sm[PARAM_NAME(sig_bytes) + 32] = {0};
    size_t smlen = sizeof(sm);
    const int NUM_ITERATIONS = bench_iterations("MAYO_BENCH_ITERS", 10);
    const bool include_copies =
        (getenv("MAYO_BENCH_INCLUDE_COPIES") != NULL && atoi(getenv("MAYO_BENCH_INCLUDE_COPIES")) != 0);
    const bool stream_mode = (getenv("MAYO_BENCH_STREAM") != NULL && atoi(getenv("MAYO_BENCH_STREAM")) != 0);
    printf("[SIGN TIMING] include_copies=%d stream=%d\n", include_copies ? 1 : 0, stream_mode ? 1 : 0);
    double iteration_time = 0.0;

    int ret = crypto_sign_keypair((unsigned char *)pk, (unsigned char *)sk);
    if (ret != MAYO_OK) {
        printf("[ERROR] crypto_sign_keypair failed: ret=%d\n", ret);
        return;
    }

    // First run
    ret = crypto_sign(sm, &smlen, msg, msg_bytes, sk);
    if (ret != MAYO_OK || smlen != (size_t)sm_stride) {
        printf("[ERROR] real crypto_sign failed: ret=%d smlen=%zu expected=%d\n", ret, smlen, sm_stride);
        return;
    }
    unsigned char cpu_opened_msg[32] = {0};
    size_t cpu_opened_mlen = sizeof(cpu_opened_msg);
    int cpu_verify_ret = crypto_sign_open(cpu_opened_msg, &cpu_opened_mlen, sm, smlen, pk);
    bool cpu_signature_valid = (cpu_verify_ret == MAYO_OK) && (cpu_opened_mlen == (size_t)msg_bytes) &&
                               (memcmp(cpu_opened_msg, msg, msg_bytes) == 0);
    printf("[CPU CHECK] signature verifies_with_pk=%s\n", cpu_signature_valid ? "yes" : "no");

    host_dynamic_sk_t host_dynamic_sk;
    sk_t host_expanded_sk;
    memset(&host_dynamic_sk, 0, sizeof(host_dynamic_sk));
    memset(&host_expanded_sk, 0, sizeof(host_expanded_sk));
    ret = mayo_expand_sk(p, sk, &host_dynamic_sk);
    if (ret != MAYO_OK) {
        printf("[ERROR] mayo_expand_sk failed: ret=%d\n", ret);
        return;
    }
    memcpy(host_expanded_sk.p, host_dynamic_sk.p, (PARAM_P1_limbs(p) + PARAM_P2_limbs(p)) * sizeof(uint64_t));
    memcpy(host_expanded_sk.O, host_dynamic_sk.O, PARAM_v(p) * PARAM_o(p));

    mayo_params_t *d_params = NULL;
    sk_t *d_expanded_sk = NULL;
    uint64_t *d_P1_bs = NULL;
    uint64_t *d_L_bs = NULL;
    unsigned char *d_csk = NULL;
    unsigned char *d_m_all = NULL;
    unsigned char *d_sm_all = NULL;
    size_t *d_smlen_all = NULL;
    int *d_ret_all = NULL;

    unsigned char *h_m_all = new unsigned char[BATCH * msg_bytes];
    unsigned char *h_sm_all = new unsigned char[BATCH * sm_stride];
    size_t *h_smlen_all = new size_t[BATCH];
    int *h_ret_all = new int[BATCH];
    if (include_copies) {
        cudaHostRegister(h_m_all, (size_t)BATCH * msg_bytes, cudaHostRegisterDefault);
        cudaHostRegister(h_sm_all, (size_t)BATCH * sm_stride, cudaHostRegisterDefault);
    }

    const bool distinct_msgs = distinct_msgs_enabled();
    for (int b = 0; b < BATCH; b++) {
        fill_distinct_message(h_m_all + b * msg_bytes, msg, msg_bytes, distinct_msgs ? b : 0);
        memset(h_sm_all + b * sm_stride, 0, sm_stride);
        h_smlen_all[b] = 0;
        h_ret_all[b] = MAYO_ERR;
    }

    cudaError_t cuerr = cudaSuccess;
    cuerr = cudaMalloc(&d_params, sizeof(mayo_params_t));
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_expanded_sk, sizeof(sk_t));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_csk, PARAM_csk_bytes(p));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_m_all, BATCH * msg_bytes);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_sm_all, BATCH * sm_stride);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_smlen_all, BATCH * sizeof(size_t));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_ret_all, BATCH * sizeof(int));
    }
    if (cuerr != cudaSuccess) {
        printf("[ERROR] cudaMalloc failed: %s\n", cudaGetErrorString(cuerr));
        delete[] h_m_all;
        delete[] h_sm_all;
        delete[] h_smlen_all;
        delete[] h_ret_all;
        cudaFree(d_params);
        cudaFree(d_expanded_sk);
        cudaFree(d_csk);
        cudaFree(d_m_all);
        cudaFree(d_sm_all);
        cudaFree(d_smlen_all);
        cudaFree(d_ret_all);
        return;
    }

    cudaMemcpy(d_params, p, sizeof(mayo_params_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expanded_sk, &host_expanded_sk, sizeof(sk_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csk, sk, PARAM_csk_bytes(p), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_all, h_m_all, BATCH * msg_bytes, cudaMemcpyHostToDevice);

    upload_bitsliced_key(p, &host_expanded_sk, &d_P1_bs, &d_L_bs);

    // Correctness launch before timing
    launch_sign(d_params, d_sm_all, d_smlen_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs, d_L_bs,
                d_ret_all, sm_stride, BATCH);
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        printf("[ERROR] GPU signing launch failed: %s\n", cudaGetErrorString(cuerr));
        delete[] h_m_all;
        delete[] h_sm_all;
        delete[] h_smlen_all;
        delete[] h_ret_all;
        cudaFree(d_params);
        cudaFree(d_expanded_sk);
        cudaFree(d_P1_bs);
        cudaFree(d_L_bs);
        cudaFree(d_csk);
        cudaFree(d_m_all);
        cudaFree(d_sm_all);
        cudaFree(d_smlen_all);
        cudaFree(d_ret_all);
        return;
    }
    cudaMemcpy(h_sm_all, d_sm_all, BATCH * sm_stride, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smlen_all, d_smlen_all, BATCH * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ret_all, d_ret_all, BATCH * sizeof(int), cudaMemcpyDeviceToHost);

    bool gpu_matches_cpu = (h_ret_all[0] == MAYO_OK) && (h_smlen_all[0] == (size_t)sm_stride) &&
                           (memcmp(h_sm_all, sm, sm_stride) == 0);
    int first_diff = -1;
    for (int i = 0; i < sm_stride; i++) {
        if (h_sm_all[i] != sm[i]) {
            first_diff = i;
            break;
        }
    }
    unsigned char opened_msg[32] = {0};
    size_t opened_mlen = sizeof(opened_msg);
    int verify_ret = crypto_sign_open(opened_msg, &opened_mlen, h_sm_all, h_smlen_all[0], pk);
    bool gpu_signature_valid = (verify_ret == MAYO_OK) && (opened_mlen == (size_t)msg_bytes) &&
                               (memcmp(opened_msg, msg, msg_bytes) == 0);
    printf("[GPU CHECK] batch[0] ret=%d smlen=%zu match_cpu=%s\n", h_ret_all[0], h_smlen_all[0],
           gpu_matches_cpu ? "yes" : "no");
    printf("[GPU CHECK] gpu_signature verifies_with_cpu_verifier=%s\n", gpu_signature_valid ? "yes" : "no");
    if (first_diff >= 0) {
        printf("[GPU CHECK] first_diff=%d cpu=%02x gpu=%02x\n", first_diff, sm[first_diff],
               h_sm_all[first_diff]);
    }

    std::ofstream outfile("benchmark_results.txt");
    if (outfile.is_open()) {
        outfile << "batch_size,iteration,ms,ops_per_sec\n";
    }

    int batches[MAX_BATCHES];
    const int num_batches = bench_batch_list(batches, MAX_BATCHES);

    printf("\n=== STARTING SIGNING THROUGHPUT BENCHMARK ===\n");
    printf("   batch |   avg ms       |   avg ops/sec  |   last ms      |   last ops/sec\n");

    for (int bi = 0; bi < num_batches; bi++) {
        const int active_batch = batches[bi];
        const int active_grid = (active_batch + BLOCK - 1) / BLOCK;
        double total_time_ms = 0.0;
        double last_time_ms = 0.0;
        double last_ops_per_sec = 0.0;

        if (stream_mode) {
            stream_alloc(active_batch + 1024, msg_bytes, sm_stride);
            unsigned char *h_csm = new unsigned char[(size_t)s_cap * sm_stride];
            int cur = 0, n_carry = 0;

            cudaMemcpy(d_m_all, h_m_all, (size_t)active_batch * msg_bytes, cudaMemcpyHostToDevice);
            int c0 = stream_launch(d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                                   d_L_bs, sm_stride, active_batch, 0, &cur, h_sm_all, h_csm, false);
            n_carry = stream_launch(d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                                    d_L_bs, sm_stride, active_batch, c0, &cur, h_sm_all, h_csm, false);
            int n_chk = c0 < 32 ? c0 : 32, chk_ok = 0, chk_same = 0, chk_signed = 0;
            if (n_chk > 0) {
                int h_cret[32];
                cudaMemcpy(h_csm, s_csm, (size_t)n_chk * sm_stride, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_cret, s_cret, n_chk * sizeof(int), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n_chk; i++) {
                    if (h_cret[i] != MAYO_OK) {
                        continue;
                    }
                    chk_signed++;
                    unsigned char *gs = h_csm + (size_t)i * sm_stride;
                    unsigned char om[32], cs[PARAM_NAME(sig_bytes) + 32];
                    size_t ol = sizeof(om), cl = sizeof(cs);
                    if (crypto_sign_open(om, &ol, gs, sm_stride, pk) == MAYO_OK && ol == (size_t)msg_bytes) {
                        chk_ok++;
                    }
                    if (crypto_sign_quiet(cs, &cl, gs + param_sig_bytes, msg_bytes, sk) == MAYO_OK &&
                        cl == (size_t)sm_stride && memcmp(cs, gs, sm_stride) == 0) {
                        chk_same++;
                    }
                }
            }
            while (n_carry > 0) {
                n_carry = stream_launch(d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                                        d_L_bs, sm_stride, 0, n_carry, &cur, h_sm_all, h_csm, false);
            }
            printf("[STREAM CHECK] batch=%d carried=%d checked=%d signed=%d verify_ok=%d byte_identical=%d\n",
                   active_batch, c0, n_chk, chk_signed, chk_ok, chk_same);

            if (include_copies) {
                cudaHostRegister(h_csm, (size_t)s_cap * sm_stride, cudaHostRegisterDefault);
            }
            cudaMemset(s_done, 0, sizeof(int));
            int carried_total = 0, flushes = 0;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
                if (include_copies) {
                    cudaMemcpyAsync(d_m_all, h_m_all, (size_t)active_batch * msg_bytes,
                                    cudaMemcpyHostToDevice, 0);
                }
                n_carry = stream_launch(d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                                        d_L_bs, sm_stride, active_batch, n_carry, &cur, h_sm_all, h_csm,
                                        include_copies);
                carried_total += n_carry;
            }
            while (n_carry > 0) {
                n_carry = stream_launch(d_params, d_sm_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                                        d_L_bs, sm_stride, 0, n_carry, &cur, h_sm_all, h_csm, include_copies);
                flushes++;
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            int done = 0;
            cudaMemcpy(&done, s_done, sizeof(int), cudaMemcpyDeviceToHost);
            if (include_copies) {
                cudaHostUnregister(h_csm);
            }
            delete[] h_csm;

            const double avg_ms = ms / NUM_ITERATIONS;
            const double ops = (active_batch * 1000.0) / avg_ms;
            printf("[STREAM] batch=%d batches=%d carried=%d flush_launches=%d signed=%d/%d\n", active_batch,
                   NUM_ITERATIONS, carried_total, flushes, done, active_batch * NUM_ITERATIONS);
            printf("   %-5d |   %-12.6f |   %-12.6f |   %-12.6f |   %-12.6f\n", active_batch, avg_ms, ops,
                   avg_ms, ops);
            printf("[SIGN THROUGHPUT] batch=%d avg_ms=%.4f ops_per_sec=%.1f\n", active_batch, avg_ms, ops);
            continue;
        }

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaMemset(d_sm_all, 0, active_batch * sm_stride);
            cudaMemset(d_smlen_all, 0, active_batch * sizeof(size_t));
            cudaMemset(d_ret_all, 0xff, active_batch * sizeof(int));

            cudaEventRecord(start);
            if (include_copies) {
                cudaMemcpyAsync(d_m_all, h_m_all, (size_t)active_batch * msg_bytes, cudaMemcpyHostToDevice,
                                0);
            }
            launch_sign(d_params, d_sm_all, d_smlen_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                        d_L_bs, d_ret_all, sm_stride, active_batch);
            if (include_copies) {
                cudaMemcpyAsync(h_sm_all, d_sm_all, (size_t)active_batch * sm_stride, cudaMemcpyDeviceToHost,
                                0);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cuerr = cudaGetLastError();
            if (cuerr != cudaSuccess) {
                printf("[ERROR] benchmark GPU signing failed: batch=%d iter=%d error=%s\n", active_batch,
                       iter, cudaGetErrorString(cuerr));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                if (outfile.is_open()) {
                    outfile.close();
                }
                goto cleanup;
            }

            float iteration_time_f = 0.0f;
            cudaEventElapsedTime(&iteration_time_f, start, stop);
            iteration_time = iteration_time_f;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            last_time_ms = iteration_time;
            last_ops_per_sec = (active_batch * 1000.0) / last_time_ms;
            total_time_ms += last_time_ms;
            if (outfile.is_open()) {
                outfile << active_batch << "," << iter << "," << last_time_ms << "," << last_ops_per_sec
                        << "\n";
            }
        }

        const double avg_time_ms = total_time_ms / NUM_ITERATIONS;
        const double avg_ops_per_sec = (active_batch * 1000.0) / avg_time_ms;

        printf("   %-5d |   %-12.6f |   %-12.6f |   %-12.6f |   %-12.6f\n", active_batch, avg_time_ms,
               avg_ops_per_sec, last_time_ms, last_ops_per_sec);
        printf("[SIGN THROUGHPUT] batch=%d avg_ms=%.4f ops_per_sec=%.1f\n", active_batch, avg_time_ms,
               avg_ops_per_sec);
    }
    if (outfile.is_open()) {
        outfile.close();
    }
    if (include_copies) {
        cudaHostUnregister(h_m_all);
        cudaHostUnregister(h_sm_all);
    }
    cudaMemcpy(h_sm_all, d_sm_all, BATCH * sm_stride, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smlen_all, d_smlen_all, BATCH * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ret_all, d_ret_all, BATCH * sizeof(int), cudaMemcpyDeviceToHost);
    printf("[SUCCESS] Benchmark Finished. Metrics saved to 'benchmark_results.txt'.\n\n");

    printf("============================================================\n");
    printf("    %s CRYPTOGRAPHIC OPERATION VERIFICATION (NIST API)   \n", PARAM_NAME(name));
    printf("============================================================\n");
    printf("[-] Pipeline Launch:        CUDA one-thread-per-sign completed.\n\n");

    // Print four batch elements from the GPU output buffer
    for (int b = 0; b < 1; b++) {
        printf("BATCH ELEMENT #%d\n", b);
        printf("[+] ret=%d smlen=%zu\n", h_ret_all[b], h_smlen_all[b]);
        printf("[+] msg:\n    ");
        for (int i = 0; i < msg_bytes; i++) {
            printf("%02x%s", msg[i], (i == msg_bytes - 1) ? "" : ", ");
        }

        // tmp is the known message digest/debug buffer from inputs.cu
        printf("\n[+] tmp:\n    ");
        for (int i = 0; i < 145; i++) {
            printf("%02x%s", tmp[i], (i == 144) ? "" : ", ");
            if ((i + 1) % 32 == 0 && i != 144) {
                printf("\n    ");
            }
        }

        // Print only the signature portion
        printf("\n\n sm\n    ");
        for (int i = 0; i < param_sig_bytes; i++) {
            printf("%02x%s", h_sm_all[b * sm_stride + i], (i == param_sig_bytes - 1) ? "" : ", ");
            if ((i + 1) % 32 == 0 && i != param_sig_bytes - 1) {
                printf("\n    ");
            }
        }
        printf("\n sm\n");
    }
    printf("============================================================\n");
    printf("                  EXECUTION SUMMARY                         \n");
    printf("============================================================\n");
    printf("  • Total Configured Batch Size : %d\n", BATCH);
    printf("  • Signature Bytes             : %d\n", param_sig_bytes);
    printf("  • Last Batch Runtime          : %.4f ms\n", iteration_time);
    printf("============================================================\n");

cleanup:
    delete[] h_m_all;
    delete[] h_sm_all;
    delete[] h_smlen_all;
    delete[] h_ret_all;
    cudaFree(d_params);
    cudaFree(d_expanded_sk);
    cudaFree(d_P1_bs);
    cudaFree(d_L_bs);
    cudaFree(d_csk);
    cudaFree(d_m_all);
    cudaFree(d_sm_all);
    cudaFree(d_smlen_all);
    cudaFree(d_ret_all);
}

// Host driver for CUDA signature verification
static void mayo_verify_impl(const mayo_params_t *p, const uint8_t *msg)
{
    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int msg_bytes = 32;
    const int sm_stride = param_sig_bytes + msg_bytes;
    const int VERIFY_BATCH = 4096;
    const int verify_grid = (VERIFY_BATCH + BLOCK - 1) / BLOCK;

    int gpu_sign_failures = 0;
    int gpu_verify_ok = 0;
    int mismatch_with_cpu = 0;
    int cpu_first_mismatch = -1;
    int gpu_tamper_ret = MAYO_ERR;
    int cpu_tamper_ret = MAYO_ERR;
    int byte_mismatches = 0;           // GPU sig bytes != CPU sig bytes
    int gpu_verify_cpu_sig = MAYO_ERR; // GPU verify of a CPU-made signature
    const int MSG2_BATCH = 128;
    int msg2_signed = 0; // GPU msg2 signatures produced
    int msg2_byte_mismatch = 0;
    int msg2_gpu_verify_ok = 0;
    int msg2_cpu_verify_ok = 0;
    int msg2_tamper_gpu = MAYO_ERR;
    int msg2_tamper_cpu = MAYO_ERR;
    bool overall_ok = false;
    const bool distinct_msgs = distinct_msgs_enabled();
    int distinct_sig_count = 0;

    printf("\n============================================================\n");
    printf("   %s CUDA SIGNATURE VERIFICATION CHECK (GPU vs CPU)    \n", PARAM_NAME(name));
    printf("============================================================\n");

    int ret = crypto_sign_keypair((unsigned char *)pk, (unsigned char *)sk);
    if (ret != MAYO_OK) {
        printf("[ERROR] verify: crypto_sign_keypair failed ret=%d\n", ret);
        return;
    }

    unsigned char cpu_sm[PARAM_NAME(sig_bytes) + 32] = {0};
    size_t cpu_smlen = sizeof(cpu_sm);
    ret = crypto_sign(cpu_sm, &cpu_smlen, msg, msg_bytes, sk);
    if (ret != MAYO_OK || cpu_smlen != (size_t)sm_stride) {
        printf("[ERROR] verify: CPU crypto_sign failed ret=%d smlen=%zu\n", ret, cpu_smlen);
        return;
    }
    unsigned char cpu_opened_msg[32] = {0};
    size_t cpu_opened_mlen = sizeof(cpu_opened_msg);
    int cpu_open_ret = crypto_sign_open(cpu_opened_msg, &cpu_opened_mlen, cpu_sm, cpu_smlen, pk);
    int cpu_detached_ret = crypto_sign_verify(cpu_sm, param_sig_bytes, msg, msg_bytes, pk);
    const bool cpu_self_ok = (cpu_open_ret == MAYO_OK) && (cpu_detached_ret == MAYO_OK) &&
                             (cpu_opened_mlen == (size_t)msg_bytes) &&
                             (memcmp(cpu_opened_msg, msg, msg_bytes) == 0);
    printf("[CPU SELF-CHECK] cpu_sign->cpu_open=%s cpu_sign_verify=%s\n", cpu_self_ok ? "OK" : "FAIL",
           (cpu_detached_ret == MAYO_OK) ? "OK" : "FAIL");
    if (!cpu_self_ok) {
        printf("[ERROR] verify: CPU library is not self-consistent; "
               "cannot use it as a reference. Aborting.\n");
        return;
    }

    uint64_t host_expanded_pk[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX];
    memset(host_expanded_pk, 0, sizeof(host_expanded_pk));
    ret = mayo_expand_pk(p, pk, host_expanded_pk);
    if (ret != MAYO_OK) {
        printf("[ERROR] verify: mayo_expand_pk failed ret=%d\n", ret);
        return;
    }

    // Expand sk on the host (needed by the GPU sign kernel).
    host_dynamic_sk_t host_dynamic_sk;
    sk_t host_expanded_sk;
    memset(&host_dynamic_sk, 0, sizeof(host_dynamic_sk));
    memset(&host_expanded_sk, 0, sizeof(host_expanded_sk));
    ret = mayo_expand_sk(p, sk, &host_dynamic_sk);
    if (ret != MAYO_OK) {
        printf("[ERROR] verify: mayo_expand_sk failed ret=%d\n", ret);
        return;
    }
    memcpy(host_expanded_sk.p, host_dynamic_sk.p, (PARAM_P1_limbs(p) + PARAM_P2_limbs(p)) * sizeof(uint64_t));
    memcpy(host_expanded_sk.O, host_dynamic_sk.O, PARAM_v(p) * PARAM_o(p));

    // 5
    mayo_params_t *d_params = NULL;
    sk_t *d_expanded_sk = NULL;
    uint64_t *d_expanded_pk = NULL;
    uint64_t *d_P1_bs = NULL;
    uint64_t *d_L_bs = NULL;
    uint64_t *d_pk_P1_bs = NULL;
    uint64_t *d_pk_P2_bs = NULL;
    uint64_t *d_pk_P3_bs = NULL;
    unsigned char *d_csk = NULL;
    unsigned char *d_m_all = NULL;
    unsigned char *d_sm_all = NULL;
    size_t *d_smlen_all = NULL;
    int *d_sign_ret_all = NULL;
    int *d_verify_ret_all = NULL;

    unsigned char *h_m_all = new unsigned char[BATCH * msg_bytes];
    unsigned char *h_sm_all = new unsigned char[BATCH * sm_stride];
    int *h_sign_ret_all = new int[BATCH];
    int *h_verify_ret_all = new int[BATCH];

    for (int b = 0; b < BATCH; b++) {
        fill_distinct_message(h_m_all + b * msg_bytes, msg, msg_bytes, distinct_msgs ? b : 0);
        memset(h_sm_all + b * sm_stride, 0, sm_stride);
        h_sign_ret_all[b] = MAYO_ERR;
        h_verify_ret_all[b] = MAYO_ERR;
    }

    cudaError_t cuerr = cudaSuccess;
    cuerr = cudaMalloc(&d_params, sizeof(mayo_params_t));
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_expanded_sk, sizeof(sk_t));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_expanded_pk, sizeof(host_expanded_pk));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_csk, PARAM_csk_bytes(p));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_m_all, BATCH * msg_bytes);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_sm_all, BATCH * sm_stride);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_smlen_all, BATCH * sizeof(size_t));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_sign_ret_all, BATCH * sizeof(int));
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMalloc(&d_verify_ret_all, BATCH * sizeof(int));
    }
    if (cuerr != cudaSuccess) {
        printf("[ERROR] verify: cudaMalloc failed: %s\n", cudaGetErrorString(cuerr));
        goto verify_cleanup;
    }

    cudaMemcpy(d_params, p, sizeof(mayo_params_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expanded_sk, &host_expanded_sk, sizeof(sk_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expanded_pk, host_expanded_pk, sizeof(host_expanded_pk), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csk, sk, PARAM_csk_bytes(p), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_all, h_m_all, BATCH * msg_bytes, cudaMemcpyHostToDevice);

    upload_bitsliced_key(p, &host_expanded_sk, &d_P1_bs, &d_L_bs);

    upload_bitsliced_pk(p, host_expanded_pk, &d_pk_P1_bs, &d_pk_P2_bs, &d_pk_P3_bs);

    cudaMemset(d_sm_all, 0, BATCH * sm_stride);
    cudaMemset(d_smlen_all, 0, BATCH * sizeof(size_t));
    cudaMemset(d_sign_ret_all, 0xff, BATCH * sizeof(int));
    launch_sign(d_params, d_sm_all, d_smlen_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs, d_L_bs,
                d_sign_ret_all, sm_stride, BATCH);
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        printf("[ERROR] verify: GPU sign kernel failed: %s\n", cudaGetErrorString(cuerr));
        goto verify_cleanup;
    }

    cudaMemset(d_verify_ret_all, 0xff, BATCH * sizeof(int));
    mayo_verify_kernel<<<verify_grid, BLOCK>>>(d_params, d_sm_all, sm_stride, d_expanded_pk, d_pk_P1_bs,
                                               d_pk_P2_bs, d_pk_P3_bs, d_verify_ret_all, sm_stride,
                                               VERIFY_BATCH);
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        printf("[ERROR] verify: GPU verify kernel failed: %s\n", cudaGetErrorString(cuerr));
        goto verify_cleanup;
    }

    cudaMemcpy(h_sm_all, d_sm_all, VERIFY_BATCH * sm_stride, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sign_ret_all, d_sign_ret_all, VERIFY_BATCH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_verify_ret_all, d_verify_ret_all, VERIFY_BATCH * sizeof(int), cudaMemcpyDeviceToHost);

    for (int b = 0; b < VERIFY_BATCH; b++) {
        if (h_sign_ret_all[b] != MAYO_OK) {
            gpu_sign_failures++;
            continue;
        }
        const int gpu_ret = h_verify_ret_all[b];
        const unsigned char *elem_msg = h_m_all + b * msg_bytes;

        // CPU reference for this exact signature and its message.
        int cpu_ret = crypto_sign_verify(h_sm_all + b * sm_stride, param_sig_bytes, elem_msg, msg_bytes, pk);
        if (gpu_ret == MAYO_OK) {
            gpu_verify_ok++;
        }
        if (gpu_ret != cpu_ret) {
            mismatch_with_cpu++;
            if (cpu_first_mismatch < 0) {
                cpu_first_mismatch = b;
            }
        }

        // Byte-identity
        bool byte_match = false;
        if (distinct_msgs) {
            unsigned char cpu_sm_b[PARAM_NAME(sig_bytes) + 32];
            size_t cpu_sm_b_len = sizeof(cpu_sm_b);
            int cpu_sret = crypto_sign_quiet(cpu_sm_b, &cpu_sm_b_len, elem_msg, msg_bytes, sk);
            byte_match = (cpu_sret == MAYO_OK) && (cpu_sm_b_len == (size_t)sm_stride) &&
                         (memcmp(h_sm_all + b * sm_stride, cpu_sm_b, sm_stride) == 0);
        } else {
            byte_match = (memcmp(h_sm_all + b * sm_stride, cpu_sm, sm_stride) == 0);
        }
        if (!byte_match) {
            byte_mismatches++;
            if (cpu_first_mismatch < 0) {
                cpu_first_mismatch = b;
            }
        }
    }

    {
        if (distinct_msgs) {
            const int HTAB = 1 << 13;
            uint64_t htab[HTAB];
            int hidx[HTAB];
            bool hused[HTAB];
            for (int i = 0; i < HTAB; i++) {
                hused[i] = false;
            }
            for (int b = 0; b < VERIFY_BATCH; b++) {
                if (h_sign_ret_all[b] != MAYO_OK) {
                    continue;
                }
                const unsigned char *sig = h_sm_all + b * sm_stride;
                uint64_t h = 14695981039346656037ULL;
                for (int i = 0; i < sm_stride; i++) {
                    h ^= sig[i];
                    h *= 1099511628211ULL;
                }
                int slot = (int)(h & (HTAB - 1));
                bool dup = false;
                int probes = 0;
                while (hused[slot] && probes < HTAB) {
                    if (htab[slot] == h && memcmp(h_sm_all + hidx[slot] * sm_stride, sig, sm_stride) == 0) {
                        dup = true;
                        break;
                    }
                    slot = (slot + 1) & (HTAB - 1);
                    probes++;
                }
                if (!dup && probes < HTAB) {
                    hused[slot] = true;
                    htab[slot] = h;
                    hidx[slot] = b;
                    distinct_sig_count++;
                }
            }
        }
    }

    // 8b
    {
        unsigned char *d_cpu_sm = NULL;
        int *d_cpu_verify_ret = NULL;
        if (cudaMalloc(&d_cpu_sm, sm_stride) == cudaSuccess &&
            cudaMalloc(&d_cpu_verify_ret, sizeof(int)) == cudaSuccess) {
            cudaMemcpy(d_cpu_sm, cpu_sm, sm_stride, cudaMemcpyHostToDevice);
            cudaMemset(d_cpu_verify_ret, 0xff, sizeof(int));
            mayo_verify_kernel<<<1, 1>>>(d_params, d_cpu_sm, sm_stride, d_expanded_pk, d_pk_P1_bs, d_pk_P2_bs,
                                         d_pk_P3_bs, d_cpu_verify_ret, sm_stride, 1);
            cudaDeviceSynchronize();
            cudaMemcpy(&gpu_verify_cpu_sig, d_cpu_verify_ret, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_cpu_sm);
            cudaFree(d_cpu_verify_ret);
        }
    }

    unsigned char tampered_sm[sm_stride];
    memcpy(tampered_sm, h_sm_all, sm_stride);
    tampered_sm[42] ^= 0x01; // flip one signature byte
    {
        unsigned char *d_tampered = NULL;
        int *d_tamper_ret = NULL;
        unsigned char *d_tampered_msg = NULL;
        size_t *d_tampered_mlen = NULL;
        int h_tamper_ret = MAYO_OK;
        if (cudaMalloc(&d_tampered, sm_stride) == cudaSuccess &&
            cudaMalloc(&d_tamper_ret, sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_tampered_msg, 64) == cudaSuccess &&
            cudaMalloc(&d_tampered_mlen, sizeof(size_t)) == cudaSuccess) {
            cudaMemcpy(d_tampered, tampered_sm, sm_stride, cudaMemcpyHostToDevice);
            cudaMemset(d_tamper_ret, 0xff, sizeof(int));
            cudaMemset(d_tampered_mlen, 0, sizeof(size_t));
            mayo_open_kernel<<<1, 1>>>(d_params, d_tampered_msg, d_tampered_mlen, d_tampered, sm_stride,
                                       d_expanded_pk, d_pk_P1_bs, d_pk_P2_bs, d_pk_P3_bs, d_tamper_ret);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_tamper_ret, d_tamper_ret, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_tampered);
            cudaFree(d_tamper_ret);
            cudaFree(d_tampered_msg);
            cudaFree(d_tampered_mlen);
        }
        gpu_tamper_ret = h_tamper_ret;
    }
    cpu_tamper_ret = crypto_sign_verify(tampered_sm, param_sig_bytes, msg, msg_bytes, pk);

    // 9b
    {
        unsigned char msg2[32];
        unsigned char cpu_sm2[PARAM_NAME(sig_bytes) + 32] = {0};
        size_t cpu_sm2len = sizeof(cpu_sm2);
        unsigned char sm2_tampered[PARAM_NAME(sig_bytes) + 32];
        for (int i = 0; i < msg_bytes; i++) {
            msg2[i] = 0xff;
        }

        int ret2 = crypto_sign_quiet(cpu_sm2, &cpu_sm2len, msg2, msg_bytes, sk);
        if (ret2 != MAYO_OK || cpu_sm2len != (size_t)sm_stride) {
            printf("[ERROR] msg2: CPU crypto_sign failed ret=%d smlen=%zu\n", ret2, cpu_sm2len);
        } else {
            const int msg2_grid = (MSG2_BATCH + BLOCK - 1) / BLOCK;
            unsigned char *h_m2 = new unsigned char[MSG2_BATCH * msg_bytes];
            unsigned char *h_sm2 = new unsigned char[MSG2_BATCH * sm_stride];
            for (int b = 0; b < MSG2_BATCH; b++) {
                fill_distinct_message(h_m2 + b * msg_bytes, msg2, msg_bytes, distinct_msgs ? b : 0);
                memset(h_sm2 + b * sm_stride, 0, sm_stride);
            }
            cudaMemcpy(d_m_all, h_m2, MSG2_BATCH * msg_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_sm_all, 0, MSG2_BATCH * sm_stride);
            cudaMemset(d_smlen_all, 0, MSG2_BATCH * sizeof(size_t));
            cudaMemset(d_sign_ret_all, 0xff, MSG2_BATCH * sizeof(int));
            cudaMemset(d_verify_ret_all, 0xff, MSG2_BATCH * sizeof(int));
            launch_sign(d_params, d_sm_all, d_smlen_all, d_m_all, msg_bytes, d_csk, d_expanded_sk, d_P1_bs,
                        d_L_bs, d_sign_ret_all, sm_stride, MSG2_BATCH);
            cudaDeviceSynchronize();
            mayo_verify_kernel<<<msg2_grid, BLOCK>>>(d_params, d_sm_all, sm_stride, d_expanded_pk, d_pk_P1_bs,
                                                     d_pk_P2_bs, d_pk_P3_bs, d_verify_ret_all, sm_stride,
                                                     MSG2_BATCH);
            cudaDeviceSynchronize();
            cudaMemcpy(h_sm2, d_sm_all, MSG2_BATCH * sm_stride, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sign_ret_all, d_sign_ret_all, MSG2_BATCH * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_verify_ret_all, d_verify_ret_all, MSG2_BATCH * sizeof(int), cudaMemcpyDeviceToHost);

            for (int b = 0; b < MSG2_BATCH; b++) {
                if (h_sign_ret_all[b] != MAYO_OK) {
                    continue;
                }
                msg2_signed++;
                const unsigned char *m2 = h_m2 + b * msg_bytes;

                unsigned char cpu_sm2_b[PARAM_NAME(sig_bytes) + 32];
                size_t cpu_sm2_b_len = sizeof(cpu_sm2_b);
                int cpu_sret2 = crypto_sign_quiet(cpu_sm2_b, &cpu_sm2_b_len, m2, msg_bytes, sk);
                if (!(cpu_sret2 == MAYO_OK && cpu_sm2_b_len == (size_t)sm_stride &&
                      memcmp(h_sm2 + b * sm_stride, cpu_sm2_b, sm_stride) == 0)) {
                    msg2_byte_mismatch++;
                }
                if (h_verify_ret_all[b] == MAYO_OK) {
                    msg2_gpu_verify_ok++;
                }
                if (crypto_sign_verify(h_sm2 + b * sm_stride, param_sig_bytes, m2, msg_bytes, pk) ==
                    MAYO_OK) {
                    msg2_cpu_verify_ok++;
                }
            }

            // Tamper round for msg2: both sides must reject.
            memcpy(sm2_tampered, cpu_sm2, sm_stride);
            sm2_tampered[42] ^= 0x01;
            msg2_tamper_cpu = crypto_sign_verify(sm2_tampered, param_sig_bytes, msg2, msg_bytes, pk);
            {
                unsigned char *d_t2 = NULL;
                int *d_t2_ret = NULL;
                unsigned char *d_t2_msg = NULL;
                size_t *d_t2_mlen = NULL;
                if (cudaMalloc(&d_t2, sm_stride) == cudaSuccess &&
                    cudaMalloc(&d_t2_ret, sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&d_t2_msg, 64) == cudaSuccess &&
                    cudaMalloc(&d_t2_mlen, sizeof(size_t)) == cudaSuccess) {
                    cudaMemcpy(d_t2, sm2_tampered, sm_stride, cudaMemcpyHostToDevice);
                    cudaMemset(d_t2_ret, 0xff, sizeof(int));
                    cudaMemset(d_t2_mlen, 0, sizeof(size_t));
                    mayo_open_kernel<<<1, 1>>>(d_params, d_t2_msg, d_t2_mlen, d_t2, sm_stride, d_expanded_pk,
                                               d_pk_P1_bs, d_pk_P2_bs, d_pk_P3_bs, d_t2_ret);
                    cudaDeviceSynchronize();
                    cudaMemcpy(&msg2_tamper_gpu, d_t2_ret, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaFree(d_t2);
                    cudaFree(d_t2_ret);
                    cudaFree(d_t2_msg);
                    cudaFree(d_t2_mlen);
                }
            }
            delete[] h_m2;
            delete[] h_sm2;
        }
    }

    printf("[GPU SIGN]      batch=%d signed_ok=%d sign_failures=%d\n", VERIFY_BATCH,
           VERIFY_BATCH - gpu_sign_failures, gpu_sign_failures);
    printf("[GPU VERIFY]    valid_signatures_accepted=%d/%d\n", gpu_verify_ok,
           VERIFY_BATCH - gpu_sign_failures);
    printf("[GPU vs CPU]    ret_mismatches=%d/%d\n", mismatch_with_cpu, VERIFY_BATCH - gpu_sign_failures);
    if (cpu_first_mismatch >= 0) {
        printf("[GPU vs CPU]    first_mismatch_at_batch=%d\n", cpu_first_mismatch);
    }
    printf("[BYTE-IDENTITY] gpu_sig==cpu_sig mismatches=%d/%d\n", byte_mismatches,
           VERIFY_BATCH - gpu_sign_failures);
    if (distinct_msgs) {
        printf("[DISTINCT SIGS] %d/%d signatures are distinct\n", distinct_sig_count,
               VERIFY_BATCH - gpu_sign_failures);
    }
    printf("[GPU<-CPU SIG]  gpu_verifies_cpu_signature=%s\n", (gpu_verify_cpu_sig == MAYO_OK) ? "yes" : "NO");
    printf("[MSG2 0xFF*32]  signed=%d byte_mismatches=%d gpu_verify_ok=%d "
           "cpu_verify_ok=%d tamper_gpu_reject=%s tamper_cpu_reject=%s\n",
           msg2_signed, msg2_byte_mismatch, msg2_gpu_verify_ok, msg2_cpu_verify_ok,
           (msg2_tamper_gpu == MAYO_ERR) ? "yes" : "NO", (msg2_tamper_cpu == MAYO_ERR) ? "yes" : "NO");
    printf("[TAMPER TEST]   corrupted_sig  gpu_reject=%s cpu_reject=%s\n",
           (gpu_tamper_ret == MAYO_ERR) ? "yes" : "NO", (cpu_tamper_ret == MAYO_ERR) ? "yes" : "NO");

    overall_ok = (gpu_sign_failures == 0) && (gpu_verify_ok == VERIFY_BATCH) && (mismatch_with_cpu == 0) &&
                 (byte_mismatches == 0) && (gpu_verify_cpu_sig == MAYO_OK) && (msg2_signed > 0) &&
                 (msg2_byte_mismatch == 0) && (msg2_gpu_verify_ok == msg2_signed) &&
                 (msg2_cpu_verify_ok == msg2_signed) && (msg2_tamper_gpu == MAYO_ERR) &&
                 (msg2_tamper_cpu == MAYO_ERR) && (gpu_tamper_ret == MAYO_ERR) &&
                 (cpu_tamper_ret == MAYO_ERR);
    printf("------------------------------------------------------------\n");
    printf("[VERDICT]       CUDA verification %s CPU reference\n", overall_ok ? "MATCHES" : "DOES NOT MATCH");
    printf("============================================================\n");

    // 10
    {
        const int NUM_VERIFY_ITERATIONS = bench_iterations("MAYO_BENCH_VERIFY_ITERS", 5);
        int batches[MAX_BATCHES];
        const int num_batches = bench_batch_list(batches, MAX_BATCHES);

        const bool include_copies =
            (getenv("MAYO_BENCH_INCLUDE_COPIES") != NULL && atoi(getenv("MAYO_BENCH_INCLUDE_COPIES")) != 0);
        printf("[VERIFY TIMING] include_copies=%d\n", include_copies ? 1 : 0);
        if (include_copies) {
            cudaMemcpy(h_sm_all, d_sm_all, (size_t)BATCH * sm_stride, cudaMemcpyDeviceToHost);
            cudaHostRegister(h_sm_all, (size_t)BATCH * sm_stride, cudaHostRegisterDefault);
            cudaHostRegister(h_verify_ret_all, (size_t)BATCH * sizeof(int), cudaHostRegisterDefault);
        }

        std::ofstream vfile("benchmark_results_verify.txt");
        if (vfile.is_open()) {
            vfile << "batch_size,iteration,ms,ops_per_sec\n";
        }

        printf("\n=== CUDA VERIFICATION THROUGHPUT BENCHMARK ===\n");
        printf("   batch |   avg ms       |   avg ops/sec  |   last ms      |   last ops/sec\n");

        for (int bi = 0; bi < num_batches; bi++) {
            const int active_batch = batches[bi];
            const int active_grid = (active_batch + BLOCK - 1) / BLOCK;
            double total_time_ms = 0.0;
            double last_time_ms = 0.0;
            double last_ops_per_sec = 0.0;

            for (int iter = 0; iter < NUM_VERIFY_ITERATIONS; iter++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start);
                if (include_copies) {
                    cudaMemcpyAsync(d_sm_all, h_sm_all, (size_t)active_batch * sm_stride,
                                    cudaMemcpyHostToDevice, 0);
                }
                mayo_verify_kernel<<<active_grid, BLOCK>>>(d_params, d_sm_all, sm_stride, d_expanded_pk,
                                                           d_pk_P1_bs, d_pk_P2_bs, d_pk_P3_bs,
                                                           d_verify_ret_all, sm_stride, active_batch);
                if (include_copies) {
                    cudaMemcpyAsync(h_verify_ret_all, d_verify_ret_all, (size_t)active_batch * sizeof(int),
                                    cudaMemcpyDeviceToHost, 0);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cuerr = cudaGetLastError();
                if (cuerr != cudaSuccess) {
                    printf("[ERROR] verification benchmark failed: batch=%d error=%s\n", active_batch,
                           cudaGetErrorString(cuerr));
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                    if (vfile.is_open()) {
                        vfile.close();
                    }
                    goto verify_cleanup;
                }

                float iteration_time_f = 0.0f;
                cudaEventElapsedTime(&iteration_time_f, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                last_time_ms = iteration_time_f;
                last_ops_per_sec = (active_batch * 1000.0) / last_time_ms;
                total_time_ms += last_time_ms;
                if (vfile.is_open()) {
                    vfile << active_batch << "," << iter << "," << last_time_ms << "," << last_ops_per_sec
                          << "\n";
                }
            }

            const double avg_time_ms = total_time_ms / NUM_VERIFY_ITERATIONS;
            const double avg_ops_per_sec = (active_batch * 1000.0) / avg_time_ms;

            printf("   %-5d |   %-12.6f |   %-12.6f |   %-12.6f |   %-12.6f\n", active_batch, avg_time_ms,
                   avg_ops_per_sec, last_time_ms, last_ops_per_sec);
            printf("[VERIFY THROUGHPUT] batch=%d avg_ms=%.4f ops_per_sec=%.1f\n", active_batch, avg_time_ms,
                   avg_ops_per_sec);
            if (include_copies) {
                int rejected = 0;
                for (int b = 0; b < active_batch; b++) {
                    if (h_verify_ret_all[b] != MAYO_OK) {
                        rejected++;
                    }
                }
                printf("[VERIFY COPIES CHECK] batch=%d rejected=%d\n", active_batch, rejected);
            }
        }
        if (include_copies) {
            cudaHostUnregister(h_sm_all);
            cudaHostUnregister(h_verify_ret_all);
        }
        if (vfile.is_open()) {
            vfile.close();
        }
        printf("============================================================\n");
    }

verify_cleanup:
    delete[] h_m_all;
    delete[] h_sm_all;
    delete[] h_sign_ret_all;
    delete[] h_verify_ret_all;
    cudaFree(d_params);
    cudaFree(d_expanded_sk);
    cudaFree(d_expanded_pk);
    cudaFree(d_P1_bs);
    cudaFree(d_L_bs);
    cudaFree(d_pk_P1_bs);
    cudaFree(d_pk_P2_bs);
    cudaFree(d_pk_P3_bs);
    cudaFree(d_csk);
    cudaFree(d_m_all);
    cudaFree(d_sm_all);
    cudaFree(d_smlen_all);
    cudaFree(d_sign_ret_all);
    cudaFree(d_verify_ret_all);
}

// C-callable wrapper used by main.c
extern "C" void run_mayo_sign()
{
    mayo_params_t params = MAYO_VARIANT;
    mayo_sign_impl(&params, msg, pk, sk);
}

extern "C" void run_mayo_verify()
{
    mayo_params_t params = MAYO_VARIANT;
    mayo_verify_impl(&params, msg);
}
