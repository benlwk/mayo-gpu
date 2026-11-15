#ifndef R3_DEVICE_CUH
#define R3_DEVICE_CUH

#define MAYO_LAMBDA_SEED_BYTES 32

__device__ __constant__ const unsigned char whip_pos_5[] = MAYO_WHIP_POS_5;
__device__ __constant__ const unsigned char whip_pos_10[] = MAYO_WHIP_POS_10;
__device__ __constant__ const unsigned char whip_pos_11[] = MAYO_WHIP_POS_11;
__device__ __constant__ const unsigned char whip_pos_12[] = MAYO_WHIP_POS_12;
__device__ const unsigned char *whip_pos = PARAM_NAME(whip_pos_arr);

#include "r3_aes_ct.cuh"

typedef struct {
    uint64_t comp[22]; // compressed AES-128 key schedule
    unsigned char iv[16];
    unsigned char buf[64]; // four keystream blocks
    uint32_t ctr;          // index of the next block to generate
    int pos;
} r3_ks_t;

__device__ void r3_ks_init(r3_ks_t *st, const unsigned char *seed)
{
    r3_aesct_keysched(st->comp, seed);
    for (int i = 0; i < 16; i++) {
        st->iv[i] = seed[16 + i];
    }
    st->ctr = 0;
    st->pos = 64;
}

__device__ void r3_ks_next(r3_ks_t *st, unsigned char *out, int len)
{
    for (int i = 0; i < len; i++) {
        if (st->pos == 64) {
            r3_aesct_lambda4(st->comp, st->iv, st->ctr, st->buf);
            st->ctr += 4;
            st->pos = 0;
        }
        out[i] = st->buf[st->pos++];
    }
}

__device__ void r3_lambda_fold_sign_device(const mayo_params_t *p, const unsigned char *Vdec,
                                           const unsigned char *O, const unsigned char *seed, uint64_t *Mtmp,
                                           uint64_t *VPV)
{
    const int n = PARAM_n(p), v = PARAM_v(p), o = PARAM_o(p), k = PARAM_k(p), m = PARAM_m(p);
    const int mvl = PARAM_m_vec_limbs(p);
    r3_ks_t st;
    r3_ks_init(&st, seed);
    for (int r = 0; r < n; r++) {
        uint64_t lam[M_VEC_LIMBS_MAX] = {0};
        r3_ks_next(&st, (unsigned char *)lam, m / 2);
        if (r < v) {
            for (int i = 0; i < k; i++) {
                m_vec_mul_add_device(mvl, lam, Vdec[i * v + r], VPV + (size_t)(i * k) * mvl);
            }
            for (int c = 0; c < o; c++) {
                m_vec_mul_add_device(mvl, lam, O[r * o + c], Mtmp + (size_t)c * mvl);
            }
        } else {
            uint64_t *dst = Mtmp + (size_t)(r - v) * mvl;
            for (int l = 0; l < mvl; l++) {
                dst[l] ^= lam[l];
            }
        }
    }
}

// verification: SPS[i][0] += Lambda(s_i)
__device__ void r3_lambda_fold_verify_device(const mayo_params_t *p, const unsigned char *s,
                                             const unsigned char *seed, uint64_t *SPS)
{
    const int n = PARAM_n(p), k = PARAM_k(p), m = PARAM_m(p);
    const int mvl = PARAM_m_vec_limbs(p);
    r3_ks_t st;
    r3_ks_init(&st, seed);
    for (int r = 0; r < n; r++) {
        uint64_t lam[M_VEC_LIMBS_MAX] = {0};
        r3_ks_next(&st, (unsigned char *)lam, m / 2);
        for (int i = 0; i < k; i++) {
            m_vec_mul_add_device(mvl, lam, s[i * n + r], SPS + (size_t)(i * k) * mvl);
        }
    }
}

#endif
