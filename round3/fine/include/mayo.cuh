#ifndef MAYO_H
#define MAYO_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include "fips202.cuh"
// #include "aes_c.h"
#include "arithmetic.cuh"
#include <cuda_runtime.h>

// Status codes
#define MAYO_OK  0
#define MAYO_ERR 1

#define THREADS (256)
#ifndef BATCH
#define BATCH 4096
#endif
#define REPEATS (10)

#ifndef MAYO_VARIANT
#define MAYO_VARIANT 2
#endif
#define F_TAIL_LEN 4

#if MAYO_VARIANT == 1
#define MAYO_P_name          "MAYO_1"
#define MAYO_P_n             88
#define MAYO_P_m             80
#define MAYO_P_m_vec_limbs   5
#define MAYO_P_o             8
#define MAYO_P_k             10
#define MAYO_P_m_bytes       40
#define MAYO_P_O_bytes       320
#define MAYO_P_v_bytes       40
#define MAYO_P_r_bytes       40
#define MAYO_P_P1_bytes      129600
#define MAYO_P_P2_bytes      25600
#define MAYO_P_P3_bytes      1440
#define MAYO_P_csk_bytes     24
#define MAYO_P_cpk_bytes     1456
#define MAYO_P_sig_bytes     464
#define MAYO_P_salt_bytes    24
#define MAYO_P_digest_bytes  32
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 24
#define MAYO_P_f_tail_init   {2, 0, 4, 2}
#define MAYO_P_whip_init                                                                                     \
    {5, 1, 6, 4, 8, 5, 8, 8, 9, 2, 8, 0, 1, 1, 2, 0, 7, 2, 9, 7, 0, 0, 6, 1, 6, 0, 5, 5,                     \
     7, 1, 3, 3, 3, 2, 7, 5, 9, 4, 4, 1, 7, 6, 8, 2, 4, 4, 8, 3, 5, 0, 1, 0, 5, 3, 9, 0,                     \
     9, 9, 8, 7, 5, 2, 9, 8, 7, 0, 5, 4, 9, 3, 7, 4, 6, 6, 8, 1, 7, 7, 4, 3, 4, 0, 8, 4,                     \
     4, 2, 9, 6, 3, 0, 2, 1, 6, 5, 9, 5, 6, 3, 8, 6, 3, 1, 2, 2, 9, 1, 6, 2, 7, 3}
#elif MAYO_VARIANT == 2
#define MAYO_P_name          "MAYO_2"
#define MAYO_P_n             86
#define MAYO_P_m             64
#define MAYO_P_m_vec_limbs   4
#define MAYO_P_o             13
#define MAYO_P_k             5
#define MAYO_P_m_bytes       32
#define MAYO_P_O_bytes       475
#define MAYO_P_v_bytes       37
#define MAYO_P_r_bytes       33
#define MAYO_P_P1_bytes      86432
#define MAYO_P_P2_bytes      30368
#define MAYO_P_P3_bytes      2912
#define MAYO_P_csk_bytes     24
#define MAYO_P_cpk_bytes     2928
#define MAYO_P_sig_bytes     239
#define MAYO_P_salt_bytes    24
#define MAYO_P_digest_bytes  32
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 24
#define MAYO_P_f_tail_init   {8, 0, 2, 8} // f(z) = z^64 + x^3 z^3 + x z^2 + x^3
#define MAYO_P_whip_init                                                                                     \
    {1, 1, 3, 2, 4, 1, 2, 0, 3, 0, 4, 2, 4, 0, 4, 4, 2, 2, 4, 3, 3, 1, 0, 0, 3, 3, 1, 0, 2, 1}
#elif MAYO_VARIANT == 3
#define MAYO_P_name          "MAYO_3"
#define MAYO_P_n             118
#define MAYO_P_m             108
#define MAYO_P_m_vec_limbs   7
#define MAYO_P_o             10
#define MAYO_P_k             11
#define MAYO_P_m_bytes       54
#define MAYO_P_O_bytes       540
#define MAYO_P_v_bytes       54
#define MAYO_P_r_bytes       55
#define MAYO_P_P1_bytes      317844
#define MAYO_P_P2_bytes      58320
#define MAYO_P_P3_bytes      2970
#define MAYO_P_csk_bytes     32
#define MAYO_P_cpk_bytes     2986
#define MAYO_P_sig_bytes     681
#define MAYO_P_salt_bytes    32
#define MAYO_P_digest_bytes  48
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 32
#define MAYO_P_f_tail_init   {8, 0, 1, 7} // f(z) = z^108 + (x^2+x+1) z^3 + z^2 + x^3
#define MAYO_P_whip_init                                                                                     \
    {5, 4,  9, 6, 10, 8, 8, 5,  4,  0, 4, 3, 2,  2,  5, 0, 10, 5, 9, 8,  7,  3, 3,  1, 9, 1, 1,              \
     0, 7,  0, 3, 2,  6, 2, 10, 6,  3, 3, 5, 1,  10, 1, 5, 3,  5, 5, 10, 4,  9, 3,  8, 8, 7, 7,              \
     6, 5,  6, 3, 7,  4, 8, 6,  10, 3, 3, 0, 10, 10, 7, 2, 0,  0, 8, 1,  6,  6, 4,  4, 2, 0, 6,              \
     0, 10, 7, 2, 1,  8, 0, 8,  2,  7, 6, 9, 2,  5,  2, 9, 5,  7, 1, 10, 2,  8, 3,  8, 7, 9, 9,              \
     8, 4,  1, 1, 6,  4, 4, 2,  6,  1, 9, 7, 9,  4,  7, 5, 9,  0, 4, 1,  10, 0, 10, 9}
#elif MAYO_VARIANT == 5
#define MAYO_P_name          "MAYO_5"
#define MAYO_P_n             154
#define MAYO_P_m             142
#define MAYO_P_m_vec_limbs   9
#define MAYO_P_o             12
#define MAYO_P_k             12
#define MAYO_P_m_bytes       71
#define MAYO_P_O_bytes       852
#define MAYO_P_v_bytes       71
#define MAYO_P_r_bytes       72
#define MAYO_P_P1_bytes      720863
#define MAYO_P_P2_bytes      120984
#define MAYO_P_P3_bytes      5538
#define MAYO_P_csk_bytes     40
#define MAYO_P_cpk_bytes     5554
#define MAYO_P_sig_bytes     964
#define MAYO_P_salt_bytes    40
#define MAYO_P_digest_bytes  64
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 40
#define MAYO_P_f_tail_init   {4, 0, 8, 1} // f(z) = z^142 + z^3 + x^3 z^2 + x^2
#define MAYO_P_whip_init                                                                                     \
    {5,  0,  9,  5,  7,  4, 11, 9, 8,  4, 6,  3, 6,  4, 1,  1, 10, 3, 9,  1, 7,  2, 10, 2,  7,  0,           \
     11, 10, 6,  6,  11, 1, 4,  3, 9,  3, 7,  3, 11, 2, 9,  0, 10, 5, 6,  5, 8,  5, 4,  0,  6,  0,           \
     3,  2,  5,  4,  9,  9, 7,  6, 1,  0, 6,  1, 11, 0, 8,  8, 9,  2, 8,  1, 10, 8, 2,  1,  11, 4,           \
     10, 7,  8,  2,  4,  4, 8,  3, 4,  1, 8,  6, 5,  2, 10, 9, 7,  1, 11, 5, 7,  5, 10, 10, 9,  4,           \
     2,  0,  3,  0,  6,  2, 3,  1, 11, 7, 10, 0, 7,  7, 9,  7, 5,  3, 3,  3, 0,  0, 5,  5,  2,  2,           \
     9,  6,  11, 11, 8,  7, 10, 4, 10, 1, 5,  1, 8,  0, 11, 6, 9,  8, 11, 8, 11, 3, 4,  2,  10, 6}
#else
#error "MAYO_VARIANT must be 1, 2, 3 or 5"
#endif
#define MAYO_P_v      (MAYO_P_n - MAYO_P_o)
#define MAYO_P_A_cols (MAYO_P_k * MAYO_P_o + 1)
#define MAYO_P_q      16

__device__ __constant__ unsigned char d_f_tail[4] = MAYO_P_f_tail_init;
#define d_f_tail_64 d_f_tail
// Round 3 whipping schedule: label l -> (row, col)
__device__ __constant__ unsigned char d_whip_pos[MAYO_P_k * (MAYO_P_k + 1)] = MAYO_P_whip_init;

// Hybrid bit-sliced m-vector layout
#define MAYO_P_bs_groups      (MAYO_P_m / 64)
#define MAYO_P_bs_bulk        (64 * MAYO_P_bs_groups)
#define MAYO_P_bs_tail        (MAYO_P_m_vec_limbs - 4 * MAYO_P_bs_groups)
#define MAYO_P_solver_threads (((MAYO_P_m + 31) / 32) * 32)

#define M_VEC_LIMBS_MAX MAYO_P_m_vec_limbs

#define P1_LIMBS_MAX (MAYO_P_v * (MAYO_P_v + 1) / 2 * MAYO_P_m_vec_limbs)
#define P2_LIMBS_MAX (MAYO_P_v * MAYO_P_o * MAYO_P_m_vec_limbs)
#define P3_LIMBS_MAX (MAYO_P_o * (MAYO_P_o + 1) / 2 * MAYO_P_m_vec_limbs)

#define EPK_LIMBS (P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX)

typedef struct sk_t {
    uint64_t p[P1_LIMBS_MAX + P2_LIMBS_MAX];
    uint8_t O[MAYO_P_v * MAYO_P_o];
} sk_t;

// Functions
int mayo2_sign_batch(unsigned char *msgs, unsigned char *sigs, const unsigned char *m, size_t mlen,
                     const unsigned char *csk);

int mayo2_sign(unsigned char *sm, size_t *smlen, const unsigned char *m, size_t mlen,
               const unsigned char *csk);

#endif
