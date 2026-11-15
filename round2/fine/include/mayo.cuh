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
#define MAYO_P_n             86
#define MAYO_P_m             78
#define MAYO_P_m_vec_limbs   5
#define MAYO_P_o             8
#define MAYO_P_k             10
#define MAYO_P_m_bytes       39
#define MAYO_P_O_bytes       312
#define MAYO_P_v_bytes       39
#define MAYO_P_r_bytes       40
#define MAYO_P_P1_bytes      120159
#define MAYO_P_P2_bytes      24336
#define MAYO_P_P3_bytes      1404
#define MAYO_P_csk_bytes     24
#define MAYO_P_cpk_bytes     1420
#define MAYO_P_sig_bytes     454
#define MAYO_P_salt_bytes    24
#define MAYO_P_digest_bytes  32
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 24
#define MAYO_P_f_tail_init   {8, 1, 1, 0} // f(z) = z^78 + z^2 + z + x^3
#elif MAYO_VARIANT == 2
#define MAYO_P_name          "MAYO_2"
#define MAYO_P_n             81
#define MAYO_P_m             64
#define MAYO_P_m_vec_limbs   4
#define MAYO_P_o             17
#define MAYO_P_k             4
#define MAYO_P_m_bytes       32
#define MAYO_P_O_bytes       544
#define MAYO_P_v_bytes       32
#define MAYO_P_r_bytes       34
#define MAYO_P_P1_bytes      66560
#define MAYO_P_P2_bytes      34816
#define MAYO_P_P3_bytes      4896
#define MAYO_P_csk_bytes     24
#define MAYO_P_cpk_bytes     4912
#define MAYO_P_sig_bytes     186
#define MAYO_P_salt_bytes    24
#define MAYO_P_digest_bytes  32
#define MAYO_P_pk_seed_bytes 16
#define MAYO_P_sk_seed_bytes 24
#define MAYO_P_f_tail_init   {8, 0, 2, 8} // f(z) = z^64 + x^3 z^3 + x z^2 + x^3
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
#else
#error "MAYO_VARIANT must be 1, 2, 3 or 5"
#endif
#define MAYO_P_v      (MAYO_P_n - MAYO_P_o)
#define MAYO_P_A_cols (MAYO_P_k * MAYO_P_o + 1)
#define MAYO_P_q      16

__device__ __constant__ unsigned char d_f_tail[4] = MAYO_P_f_tail_init;
#define d_f_tail_64 d_f_tail

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
