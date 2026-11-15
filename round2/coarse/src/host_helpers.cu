#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "../include/host_helpers.h"

// Deterministic randombytes hook
extern "C" int randombytes(unsigned char *x, size_t xlen)
{
    for (size_t i = 0; i < xlen; i++) {
        x[i] = 0x01;
    }
    return 0;
}

static inline void m_vec_add(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        acc[i] ^= in[i];
    }
}

static inline uint64_t gf16v_mul_u64(uint64_t a, unsigned char b)
{
    uint64_t res = 0;

    for (int i = 0; i < 8; i++) {
        unsigned char lane = (a >> (i * 8)) & 0xff;
        unsigned char nib = lane & 0x0f;
        unsigned char p = 0;
        unsigned char aa = nib;
        unsigned char bb = b & 0xf;
        for (int j = 0; j < 4; j++) {
            if (bb & 1) {
                p ^= aa;
            }
            unsigned char hi = aa & 8;
            aa <<= 1;
            if (hi) {
                aa ^= 0x3;
            }
            bb >>= 1;
        }
        res |= (uint64_t)(p & 0xf) << (i * 8);
    }
    return res;
}

static inline void m_vec_mul_add(int m_vec_limbs, const uint64_t *in, unsigned char a, uint64_t *acc)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        acc[i] ^= gf16v_mul_u64(in[i], a);
    }
}

static inline void m_vec_copy(int m_vec_limbs, const uint64_t *in, uint64_t *out)
{
    for (int i = 0; i < m_vec_limbs; i++) {
        out[i] = in[i];
    }
}

static inline void m_vec_mul_add_x(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    uint64_t mask_msb = 0x8888888888888888ULL;
    for (int i = 0; i < m_vec_limbs; i++) {
        uint64_t t = in[i] & mask_msb;
        acc[i] ^= ((in[i] ^ t) << 1) ^ ((t >> 3) * 3);
    }
}

static inline void m_vec_mul_add_x_inv(int m_vec_limbs, const uint64_t *in, uint64_t *acc)
{
    uint64_t mask_lsb = 0x1111111111111111ULL;
    for (int i = 0; i < m_vec_limbs; i++) {
        uint64_t t = in[i] & mask_lsb;
        acc[i] ^= ((in[i] ^ t) >> 1) ^ (t * 9);
    }
}

static inline void m_vec_multiply_bins(int m_vec_limbs, uint64_t *bins, uint64_t *out)
{
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 5 * m_vec_limbs, bins + 10 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 11 * m_vec_limbs, bins + 12 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 10 * m_vec_limbs, bins + 7 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 12 * m_vec_limbs, bins + 6 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 7 * m_vec_limbs, bins + 14 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 6 * m_vec_limbs, bins + 3 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 14 * m_vec_limbs, bins + 15 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 3 * m_vec_limbs, bins + 8 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 15 * m_vec_limbs, bins + 13 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 8 * m_vec_limbs, bins + 4 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 13 * m_vec_limbs, bins + 9 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 4 * m_vec_limbs, bins + 2 * m_vec_limbs);
    m_vec_mul_add_x_inv(m_vec_limbs, bins + 9 * m_vec_limbs, bins + 1 * m_vec_limbs);
    m_vec_mul_add_x(m_vec_limbs, bins + 2 * m_vec_limbs, bins + 1 * m_vec_limbs);
    m_vec_copy(m_vec_limbs, bins + m_vec_limbs, out);
}

// Unpack/Pack utilities (from mayo.c)
static void unpack_m_vecs(const unsigned char *in, uint64_t *out, int vecs, int m)
{
    const int m_vec_limbs = (m + 15) / 16;
    unsigned char *_out = (unsigned char *)out;
    uint64_t tmp[(64 + 15) / 16];
    for (int i = vecs - 1; i >= 0; i--) {
        memcpy(tmp, in + i * m / 2, m / 2);
        memcpy(_out + i * m_vec_limbs * sizeof(uint64_t), tmp, m_vec_limbs * sizeof(uint64_t));
    }
}

static void pack_m_vecs(const uint64_t *in, unsigned char *out, int vecs, int m)
{
    const int m_vec_limbs = (m + 15) / 16;
    unsigned char *_in = (unsigned char *)in;
    for (int i = 0; i < vecs; i++) {
        memmove(out + (i * m / 2), _in + i * m_vec_limbs * sizeof(uint64_t), m / 2);
    }
}

// P1P1t_times_O port (from generic_arithmetic.h)
static void P1P1t_times_O_host(const int param_v, const int param_o, const int m_vec_limbs,
                               const uint64_t *P1, const unsigned char *O, uint64_t *acc)
{
    int bs_mat_entries_used = 0;
    for (int r = 0; r < param_v; r++) {
        for (int c = r; c < param_v; c++) {
            if (c == r) {
                bs_mat_entries_used += 1;
                continue;
            }
            for (int k = 0; k < param_o; k += 1) {
                m_vec_mul_add(m_vec_limbs, P1 + m_vec_limbs * bs_mat_entries_used, O[c * param_o + k],
                              acc + m_vec_limbs * (r * param_o + k));
                m_vec_mul_add(m_vec_limbs, P1 + m_vec_limbs * bs_mat_entries_used, O[r * param_o + k],
                              acc + m_vec_limbs * (c * param_o + k));
            }
            bs_mat_entries_used += 1;
        }
    }
}

// expand_P1_P2
static void expand_P1_P2_host(const unsigned char *cpk, uint64_t *P, int total_bytes, int m)
{
    memcpy((unsigned char *)P, cpk, total_bytes);
    // unpack into m-vecs; assume vecs = total_bytes / (m/2)
    int vecs = total_bytes / (m / 2);
    unpack_m_vecs((unsigned char *)P, P, vecs, m);
}

// Public helper called from mayo_sign
void compute_host_P1_P2_and_L(const unsigned char *cpk, uint64_t *P1, uint64_t *L)
{
    // parameter constants used in this repo
    const int param_v = 66;
    const int param_o = 8;
    const int param_m = 64;
    const int m_vec_limbs = (param_m + 15) / 16;

    int total_P_bytes = 4096;
    // expand P1||P2 into temporary buffer
    unsigned char *tmp = (unsigned char *)malloc(total_P_bytes);
    if (!tmp) {
        return;
    }
    // For now, assume cpk holds expanded P1||P2 at its start; copy
    memcpy(tmp, cpk, total_P_bytes);

    unpack_m_vecs(tmp, P1, total_P_bytes / (param_m / 2), param_m);

    unsigned char *O = (unsigned char *)malloc(param_v * param_o);
    for (int i = 0; i < param_v * param_o; i++) {
        O[i] = 0x01;
    }

    // compute acc initialized to zero
    int L_entries = param_v * param_o * m_vec_limbs;
    for (int i = 0; i < L_entries; i++) {
        L[i] = 0;
    }

    P1P1t_times_O_host(param_v, param_o, m_vec_limbs, P1, O, L);
}
