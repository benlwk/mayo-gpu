// Tensor-Core P * s^t over GF(16): bit planes and AND.POPC binary MMA
#ifndef TC_PS_CUH
#define TC_PS_CUH

#include <stdint.h>

#define TC_RT   ((MAYO_P_n + 7) / 8)     // row tiles of 8
#define TC_KT   ((MAYO_P_n + 127) / 128) // K tiles of 128 bits
#define TC_COLS (MAYO_P_k * BATCH)       // columns of S^T over the batch
#define TC_CT   ((TC_COLS + 7) / 8)      // column tiles of 8
#ifndef TC_NC
#define TC_NC 2 // column tiles per warp (A-fragment reuse)
#endif
#ifndef TC_KARATSUBA
#define TC_KARATSUBA 0
#endif
#if TC_KARATSUBA
#define TC_NF 9 // operand combinations per fragment
#else
#define TC_NF 4
#endif
#define TC_PA_WORDS ((size_t)MAYO_P_m * TC_RT * TC_KT * TC_NF * 32)

__device__ __forceinline__ uint32_t tc_key_nibble(const uint64_t *__restrict__ P, int r, int j, int p)
{
    const int v = MAYO_P_v, o = MAYO_P_o, ml = MAYO_P_m_vec_limbs;
    size_t idx;
    if (r < v) {
        if (j < v) {
            idx = (size_t)(r * v - (r * (r - 1)) / 2 + (j - r)) * ml;
        } else {
            idx = P1_LIMBS_MAX + (size_t)(r * o + (j - v)) * ml;
        }
    } else {
        const int rr = r - v, jj = j - v;
        idx = P1_LIMBS_MAX + P2_LIMBS_MAX + (size_t)(rr * o - (rr * (rr - 1)) / 2 + (jj - rr)) * ml;
    }
    return (uint32_t)(P[idx + (p >> 4)] >> (4 * (p & 15))) & 0xF;
}

__device__ __forceinline__ void tc_combos(const uint32_t w[4], uint32_t f[TC_NF])
{
#if TC_KARATSUBA
    f[0] = w[0];
    f[1] = w[1];
    f[2] = w[0] ^ w[1];
    f[3] = w[2];
    f[4] = w[3];
    f[5] = w[2] ^ w[3];
    f[6] = w[0] ^ w[2];
    f[7] = w[1] ^ w[3];
    f[8] = f[6] ^ f[7];
#else
    f[0] = w[0];
    f[1] = w[1];
    f[2] = w[2];
    f[3] = w[3];
#endif
}

// A fragments
__global__ void tc_build_key_frags(const uint64_t *__restrict__ P, uint32_t *__restrict__ PA)
{
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // one fragment lane
    if (g >= TC_PA_WORDS / TC_NF) {
        return;
    }
    const int lane = (int)(g & 31);
    size_t t = g >> 5;
    const int kt = (int)(t % TC_KT);
    t /= TC_KT;
    const int rt = (int)(t % TC_RT);
    t /= TC_RT;
    const int p = (int)t;
    const int r = rt * 8 + (lane >> 2);
    const int jb = kt * 128 + (lane & 3) * 32;
    uint32_t w[4] = {0, 0, 0, 0};
    if (r < MAYO_P_n) {
        for (int q = 0; q < 32; q++) {
            const int j = jb + q;
            if (j >= r && j < MAYO_P_n) {
                const uint32_t e = tc_key_nibble(P, r, j, p);
                for (int i = 0; i < 4; i++) {
                    w[i] |= ((e >> i) & 1u) << q;
                }
            }
        }
    }
    uint32_t f[TC_NF];
    tc_combos(w, f);
    uint32_t *dst = PA + ((size_t)(p * TC_RT + rt) * TC_KT + kt) * (TC_NF * 32) + lane;
    for (int i = 0; i < TC_NF; i++) {
        dst[i * 32] = f[i];
    }
}

__device__ __forceinline__ void tc_bmma(int &d0, int &d1, uint32_t a, uint32_t b)
{
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a), "r"(b));
}

__device__ __forceinline__ uint32_t tc_nibble(const int acc[TC_NF > 7 ? TC_NF : 7][2], int e)
{
#if TC_KARATSUBA
    uint32_t P[9];
#pragma unroll
    for (int i = 0; i < 9; i++) {
        P[i] = (uint32_t)acc[i][e] & 1u;
    }

    const uint32_t L0 = P[0], L1 = P[2] ^ P[0] ^ P[1], L2 = P[1];
    const uint32_t H0 = P[3], H1 = P[5] ^ P[3] ^ P[4], H2 = P[4];
    const uint32_t S0 = P[6], S1 = P[8] ^ P[6] ^ P[7], S2 = P[7];

    const uint32_t c0 = L0, c1 = L1, c2 = L2 ^ S0 ^ L0 ^ H0, c3 = S1 ^ L1 ^ H1;
    const uint32_t c4 = S2 ^ L2 ^ H2 ^ H0, c5 = H1, c6 = H2;
#else
    const uint32_t c0 = acc[0][e] & 1, c1 = acc[1][e] & 1, c2 = acc[2][e] & 1, c3 = acc[3][e] & 1;
    const uint32_t c4 = acc[4][e] & 1, c5 = acc[5][e] & 1, c6 = acc[6][e] & 1;
#endif
    return (c0 ^ c4) | ((c1 ^ c4 ^ c5) << 1) | ((c2 ^ c5 ^ c6) << 2) | ((c3 ^ c6) << 3);
}

#ifndef TC_WPB
#define TC_WPB 4 // warps per block; all share one row tile
#endif
#define TC_GROUPS   ((TC_CT + TC_NC - 1) / TC_NC) // column groups of TC_NC tiles
#define TC_BLOCKS   (TC_RT * ((TC_GROUPS + TC_WPB - 1) / TC_WPB))
#define TC_SH_WORDS (16 * TC_KT * TC_NF * 32) // A fragments of 16 polynomials

// One block
__global__ void __launch_bounds__(TC_WPB * 32)
    verify_calc_PS_tc(const uint32_t *__restrict__ PA, const unsigned char *__restrict__ d_s,
                      uint64_t *__restrict__ d_PS)
{
    const int n = MAYO_P_n, k = MAYO_P_k, ml = MAYO_P_m_vec_limbs;
    __shared__ uint32_t shA[TC_SH_WORDS];
    const int lane = threadIdx.x & 31;
    const int rt = blockIdx.x % TC_RT;
    const int grp = (blockIdx.x / TC_RT) * TC_WPB + (threadIdx.x >> 5);
    const int ct0 = grp * TC_NC;

    uint32_t B[TC_NC][TC_KT][TC_NF];
#pragma unroll
    for (int t = 0; t < TC_NC; t++) {
        const int col = (ct0 + t) * 8 + (lane >> 2);
        const unsigned char *s =
            (col < TC_COLS) ? d_s + (size_t)(col / k) * (k * n) + (size_t)(col % k) * n : NULL;
#pragma unroll
        for (int kt = 0; kt < TC_KT; kt++) {
            uint32_t w[4] = {0, 0, 0, 0};
            if (s) {
                const int jb = kt * 128 + (lane & 3) * 32;
                for (int q = 0; q < 32; q++) {
                    const int j = jb + q;
                    const uint32_t e = (j < n) ? s[j] : 0u;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        w[i] |= ((e >> i) & 1u) << q;
                    }
                }
            }
            tc_combos(w, B[t][kt]);
        }
    }

    // D fragment
    const int row = rt * 8 + (lane >> 2);
    uint64_t *dst[TC_NC][2];
#pragma unroll
    for (int t = 0; t < TC_NC; t++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int col = (ct0 + t) * 8 + (lane & 3) * 2 + e;
            dst[t][e] = (row < n && col < TC_COLS)
                            ? d_PS + (size_t)(col / k) * (n * k * ml) + (size_t)(row * k + (col % k)) * ml
                            : NULL;
        }

#pragma unroll 1
    for (int l = 0; l < MAYO_P_m_vec_limbs; l++) {
        const int np = (16 * (l + 1) < MAYO_P_m) ? 16 : MAYO_P_m - 16 * l; // polynomials in this limb
        __syncthreads();
        for (int w = threadIdx.x; w < np * TC_KT * TC_NF * 32; w += TC_WPB * 32) {
            const int pp = w / (TC_KT * TC_NF * 32), rest = w % (TC_KT * TC_NF * 32);
            shA[w] = __ldg(PA + ((size_t)((16 * l + pp) * TC_RT + rt) * TC_KT) * (TC_NF * 32) + rest);
        }
        __syncthreads();
        uint32_t lo[TC_NC][2], hi[TC_NC][2]; // nibbles 0..7 and 8..15 of the limb
#pragma unroll
        for (int t = 0; t < TC_NC; t++) {
            lo[t][0] = lo[t][1] = hi[t][0] = hi[t][1] = 0;
        }
#pragma unroll
        for (int q = 0; q < 16; q++) {
            if (q >= np) {
                break;
            }
            uint32_t A[TC_KT][TC_NF];
            const uint32_t *pa = shA + q * (TC_KT * TC_NF * 32) + lane;
#pragma unroll
            for (int kt = 0; kt < TC_KT; kt++)
#pragma unroll
                for (int i = 0; i < TC_NF; i++) {
                    A[kt][i] = pa[(kt * TC_NF + i) * 32];
                }
#pragma unroll
            for (int t = 0; t < TC_NC; t++) {
                int acc[TC_NF > 7 ? TC_NF : 7][2];
#pragma unroll
                for (int d = 0; d < (TC_NF > 7 ? TC_NF : 7); d++) {
                    acc[d][0] = acc[d][1] = 0;
                }
#pragma unroll
                for (int kt = 0; kt < TC_KT; kt++) {
#if TC_KARATSUBA
#pragma unroll
                    for (int i = 0; i < 9; i++) {
                        tc_bmma(acc[i][0], acc[i][1], A[kt][i], B[t][kt][i]);
                    }
#else
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            tc_bmma(acc[i + j][0], acc[i + j][1], A[kt][i], B[t][kt][j]);
                        }
#endif
                }
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (q < 8) {
                        lo[t][e] |= tc_nibble(acc, e) << (4 * q);
                    } else {
                        hi[t][e] |= tc_nibble(acc, e) << (4 * (q - 8));
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TC_NC; t++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                if (dst[t][e]) {
                    dst[t][e][l] = ((uint64_t)hi[t][e] << 32) | lo[t][e];
                }
            }
    }
}

#endif
