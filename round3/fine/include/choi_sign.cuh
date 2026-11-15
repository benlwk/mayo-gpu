// Tensor-Core products of the first signing attempt, after Choi et al. (ePrint 2026/2577)
#ifndef CHOI_SIGN_CUH
#define CHOI_SIGN_CUH
#include "tc_ps.cuh" // tc_bmma

#define CS_KT       ((MAYO_P_v + 127) / 128) // K tiles over v
#define CS_RT_P1    ((MAYO_P_v + 7) / 8)     // row tiles of P1
#define CS_RT_LT    ((MAYO_P_o + 7) / 8)     // row tiles of L^T
#define CS_COLS     (MAYO_P_k * BATCH)
#define CS_CT       ((CS_COLS + 7) / 8)
#define CS_NC       2 // column tiles per warp
#define CS_VB_WORDS ((size_t)CS_CT * CS_KT * 4 * 32)

// coefficient p of an m-vector in the hybrid layout
__device__ __forceinline__ uint32_t cs_coef(const uint64_t *__restrict__ vec, int p)
{
    if (p < MAYO_P_bs_bulk) {
        const int g = p >> 6, e = p & 63;
        return (uint32_t)(((vec[4 * g] >> e) & 1) | (((vec[4 * g + 1] >> e) & 1) << 1) |
                          (((vec[4 * g + 2] >> e) & 1) << 2) | (((vec[4 * g + 3] >> e) & 1) << 3));
    }
    const int t = p - MAYO_P_bs_bulk;
    return (uint32_t)(vec[4 * MAYO_P_bs_groups + (t >> 4)] >> (4 * (t & 15))) & 0xF;
}

// A fragments of the key-side matrix
template <int MODE>
__global__ void choi_sign_key_frags(const uint64_t *__restrict__ planes, uint32_t *__restrict__ KA)
{
    constexpr int RT = MODE == 0 ? CS_RT_P1 : CS_RT_LT, ROWS = MODE == 0 ? MAYO_P_v : MAYO_P_o;
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (size_t)MAYO_P_m * RT * CS_KT * 32) {
        return;
    }
    const int lane = (int)(g & 31);
    size_t t = g >> 5;
    const int kt = (int)(t % CS_KT);
    t /= CS_KT;
    const int rt = (int)(t % RT);
    t /= RT;
    const int p = (int)t;
    const int r = rt * 8 + (lane >> 2);
    const int cb = kt * 128 + (lane & 3) * 32;
    uint32_t w[4] = {0, 0, 0, 0};
    if (r < ROWS) {
        for (int q = 0; q < 32; q++) {
            const int c = cb + q;
            if (c >= MAYO_P_v) {
                break;
            }
            uint32_t e = 0;
            if (MODE == 0) {
                if (c >= r) {
                    e = cs_coef(planes +
                                    (size_t)(r * MAYO_P_v - (r * (r - 1)) / 2 + (c - r)) * MAYO_P_m_vec_limbs,
                                p);
                }
            } else {
                e = cs_coef(planes + (size_t)(c * MAYO_P_o + r) * MAYO_P_m_vec_limbs, p);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                w[i] |= ((e >> i) & 1u) << q;
            }
        }
    }
    uint32_t *dst = KA + ((size_t)(p * RT + rt) * CS_KT + kt) * 128 + lane;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[i * 32] = w[i];
    }
}

__global__ void choi_pack_V(const unsigned char *__restrict__ Vdec, uint32_t *__restrict__ VB)
{
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (size_t)CS_CT * CS_KT * 32) {
        return;
    }
    const int lane = (int)(g & 31);
    const int kt = (int)((g >> 5) % CS_KT);
    const int ct = (int)((g >> 5) / CS_KT);
    const int col = ct * 8 + (lane >> 2);
    uint32_t w[4] = {0, 0, 0, 0};
    if (col < CS_COLS) {
        const unsigned char *v =
            Vdec + (size_t)(col / MAYO_P_k) * (MAYO_P_k * MAYO_P_v) + (size_t)(col % MAYO_P_k) * MAYO_P_v;
        const int cb = kt * 128 + (lane & 3) * 32;
        for (int q = 0; q < 32; q++) {
            const int c = cb + q;
            const uint32_t e = (c < MAYO_P_v) ? v[c] : 0u;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                w[i] |= ((e >> i) & 1u) << q;
            }
        }
    }
    uint32_t *dst = VB + ((size_t)ct * CS_KT + kt) * 128 + lane;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[i * 32] = w[i];
    }
}

// GF(16) value of one D-fragment entry
__device__ __forceinline__ uint32_t cs_nib(const int D[7][2], int e)
{
    const uint32_t c4 = D[4][e], c5 = D[5][e], c6 = D[6][e];
    return ((D[0][e] ^ c4) & 1u) | (((D[1][e] ^ c4 ^ c5) & 1u) << 1) | (((D[2][e] ^ c5 ^ c6) & 1u) << 2) |
           (((D[3][e] ^ c6) & 1u) << 3);
}

// One warp
template <int MODE>
__global__ void __launch_bounds__(128)
    choi_sign_mm(const uint32_t *__restrict__ KA, const uint32_t *__restrict__ VB, uint64_t *__restrict__ out)
{
    constexpr int RT = MODE == 0 ? CS_RT_P1 : CS_RT_LT, ROWS = MODE == 0 ? MAYO_P_v : MAYO_P_o;
    constexpr int k = MAYO_P_k, ml = MAYO_P_m_vec_limbs;
    const int lane = threadIdx.x & 31;
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups = (CS_CT + CS_NC - 1) / CS_NC;
    if (gw >= RT * groups) {
        return;
    }
    const int rt = gw % RT, ct0 = (gw / RT) * CS_NC;
    const int kt0 = (MODE == 0 && CS_KT > 1 && rt * 8 >= 128) ? 1 : 0; // upper triangular P1

    uint32_t B[CS_NC][CS_KT][4];
#pragma unroll
    for (int t = 0; t < CS_NC; t++)
#pragma unroll
        for (int kt = 0; kt < CS_KT; kt++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                B[t][kt][d] = (ct0 + t < CS_CT)
                                  ? __ldg(VB + ((size_t)(ct0 + t) * CS_KT + kt) * 128 + d * 32 + lane)
                                  : 0u;
            }

    const int row = rt * 8 + (lane >> 2);
    uint64_t *dst[CS_NC][2];
#pragma unroll
    for (int t = 0; t < CS_NC; t++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int col = (ct0 + t) * 8 + (lane & 3) * 2 + e;
            const int b = col / k, i = col % k;
            dst[t][e] = (row < ROWS && col < CS_COLS)
                            ? out + (size_t)b * (MODE == 0 ? MAYO_P_v * k * ml : k * MAYO_P_o * ml) +
                                  (size_t)(MODE == 0 ? row * k + i : i * MAYO_P_o + row) * ml
                            : NULL;
        }

    auto run = [&](int p0, int np, uint32_t acc[CS_NC][2][4], bool planes) {
#pragma unroll 2
        for (int q = 0; q < np; q++) {
            const int p = p0 + q;
            const uint32_t *pa = KA + ((size_t)(p * RT + rt) * CS_KT) * 128 + lane;
            uint32_t A[CS_KT][4];
#pragma unroll
            for (int kt = 0; kt < CS_KT; kt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
                    A[kt][i] = (kt >= kt0) ? __ldg(pa + (kt * 4 + i) * 32) : 0u;
#pragma unroll
            for (int t = 0; t < CS_NC; t++) {
                int D[7][2];
#pragma unroll
                for (int d = 0; d < 7; d++)
                    D[d][0] = D[d][1] = 0;
#pragma unroll
                for (int kt = 0; kt < CS_KT; kt++) {
                    if (kt < kt0)
                        continue;
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            tc_bmma(D[i + j][0], D[i + j][1], A[kt][i], B[t][kt][j]);
                }
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const uint32_t v = cs_nib(D, e);
                    if (planes) { // bit q of plane d (q < 32 within a half)
#pragma unroll
                        for (int d = 0; d < 4; d++)
                            acc[t][e][d] |= ((v >> d) & 1u) << q;
                    } else { // nibble q of a 64-bit word: two halves
                        if (q < 8)
                            acc[t][e][0] |= v << (4 * q);
                        else
                            acc[t][e][1] |= v << (4 * (q - 8));
                    }
                }
            }
        }
    };

#pragma unroll 1
    for (int g = 0; g < MAYO_P_bs_groups; g++) {
        uint32_t lo[CS_NC][2][4] = {}, hi[CS_NC][2][4] = {};
        run(64 * g, 32, lo, true);
        run(64 * g + 32, 32, hi, true);
#pragma unroll
        for (int t = 0; t < CS_NC; t++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                if (dst[t][e])
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        dst[t][e][4 * g + d] = ((uint64_t)hi[t][e][d] << 32) | lo[t][e][d];
                    }
            }
    }
    // nibble tail words: 16 polynomials each
#pragma unroll 1
    for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
        const int p0 = MAYO_P_bs_bulk + 16 * tw;
        const int np = (p0 + 16 <= MAYO_P_m) ? 16 : MAYO_P_m - p0;
        uint32_t nb[CS_NC][2][4] = {};
        run(p0, np, nb, false);
#pragma unroll
        for (int t = 0; t < CS_NC; t++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                if (dst[t][e]) {
                    dst[t][e][4 * MAYO_P_bs_groups + tw] = ((uint64_t)nb[t][e][1] << 32) | nb[t][e][0];
                }
            }
    }
}

// Sign
#include "choi_verify.cuh"
#define choi_sign_VPV choi_fused_SPS<MAYO_P_v, CS_RT_P1, CS_KT, CS_CT, CS_COLS>

#endif
