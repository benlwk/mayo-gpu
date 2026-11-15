// Right-hand side y, one warp per signature
#ifndef RHS_W_CUH
#define RHS_W_CUH
#include "choi_solve.cuh" // csv_gf16_mul

#define RHS_SPW (32 / MAYO_P_m_vec_limbs) // signatures per warp
#ifndef RHS_PF
#define RHS_PF 8 // pair terms loaded ahead
#endif

__global__ void __launch_bounds__(128)
    compute_rhs_w(const uint64_t *__restrict__ vPv, const unsigned char *__restrict__ t,
                  unsigned char *__restrict__ y, int nsig)
{
    constexpr int ml = MAYO_P_m_vec_limbs, k = MAYO_P_k, m = MAYO_P_m;
    constexpr int top_pos = ((m - 1) % 16) * 4;
    constexpr unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int ls = lane / ml, limb = lane % ml, base = lane - limb;
    const int batch = warp * RHS_SPW + ls;
    const bool act = ls < RHS_SPW && batch < nsig;
    const uint64_t *v = vPv + (size_t)(act ? batch : 0) * k * k * ml + limb;

    // the pair terms do not depend on the Horner chain
    constexpr int NP = k * (k + 1) / 2;
    uint64_t temp = 0;
#pragma unroll 1
    for (int l0 = NP - 1; l0 >= 0; l0 -= RHS_PF) { // public loop
        uint64_t ab[RHS_PF];
#pragma unroll
        for (int u = 0; u < RHS_PF; u++) {
            const int l = l0 - u;
            uint64_t t = 0;
            if (l >= 0) {
                const int i = d_whip_pos[2 * l], j = d_whip_pos[2 * l + 1];
                const uint64_t a = act ? __ldg(v + (i * k + j) * ml) : 0;
                const uint64_t b = (act && i != j) ? __ldg(v + (j * k + i) * ml) : 0;
                t = a ^ b;
            }
            ab[u] = t;
        }
#pragma unroll
        for (int u = 0; u < RHS_PF; u++) {
            if (l0 - u < 0) {
                break;
            }
            const uint64_t topl = __shfl_sync(FULL, temp, (base + ml - 1) & 31);
            const uint64_t below = __shfl_up_sync(FULL, temp, 1);
            const uint32_t top = (uint32_t)(topl >> top_pos) & 0xFu;
            temp = (temp << 4) ^ (limb > 0 ? below >> 60 : 0);
            if (limb == 0) { // public condition: z^m = f_tail(z)
#pragma unroll
                for (int jj = 0; jj < F_TAIL_LEN; jj++) {
                    temp ^= (uint64_t)csv_gf16_mul(top, d_f_tail_64[jj]) << (4 * jj);
                }
            }
            temp ^= ab[u];
        }
    }
    if (!act) {
        return;
    }
    const unsigned char *t_b = t + (size_t)batch * m;
    unsigned char *y_b = y + (size_t)batch * m;
#pragma unroll
    for (int q = 0; q < 16; q++) {
        const int idx = 16 * limb + q;
        if (idx < m) {
            y_b[idx] = t_b[idx] ^ (unsigned char)((temp >> (4 * q)) & 0xF);
        }
    }
}

#include "arithmetic.cuh" // gf16_vec_mul
#define RHX_NP  (MAYO_P_k * (MAYO_P_k + 1) / 2)
#define RHX_LE  ((MAYO_P_m + RHX_NP + 15) / 16) // limbs of the unreduced sum
#define RHX_SPW (32 / RHX_LE)                   // signatures per warp
static_assert(RHX_NP + 3 < MAYO_P_m, "one f(z) fold must suffice");

__global__ void __launch_bounds__(128)
    compute_rhs_x(const uint64_t *__restrict__ vPv, const unsigned char *__restrict__ t,
                  unsigned char *__restrict__ y, int nsig)
{
    constexpr int ml = MAYO_P_m_vec_limbs, k = MAYO_P_k, m = MAYO_P_m, NP = RHX_NP, LE = RHX_LE;
    constexpr unsigned FULL = 0xffffffffu;
    constexpr uint64_t lastmask = (m % 16) ? ((1ull << (4 * (m % 16))) - 1) : ~0ull;
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int ls = lane / LE, L = lane % LE, base = lane - L;
    const int batch = warp * RHX_SPW + ls;
    const bool act = ls < RHX_SPW && batch < nsig;
    const uint64_t *v = vPv + (size_t)(act ? batch : 0) * k * k * ml;

    uint64_t e = 0;
#pragma unroll 5
    for (int l = 0; l < NP; l++) { // public loop
        const int i = d_whip_pos[2 * l], j = d_whip_pos[2 * l + 1];
        const int q = l >> 4, sh = 4 * (l & 15), hi = L - q, lo = hi - 1;
        uint64_t vh = 0, vl = 0;
        if (act && hi >= 0 && hi < ml) {
            vh = __ldg(v + (i * k + j) * ml + hi) ^ (i != j ? __ldg(v + (j * k + i) * ml + hi) : 0);
            if (hi == ml - 1) {
                vh &= lastmask;
            }
        }
        if (act && sh != 0 && lo >= 0 && lo < ml) {
            vl = __ldg(v + (i * k + j) * ml + lo) ^ (i != j ? __ldg(v + (j * k + i) * ml + lo) : 0);
            if (lo == ml - 1) {
                vl &= lastmask;
            }
        }
        e ^= (vh << sh) | (sh != 0 ? vl >> (64 - sh) : 0);
    }

    constexpr int hq = m / 16, hs = 4 * (m % 16);
    const int s0 = hq + L, s1 = s0 + 1;
    const uint64_t x0 = __shfl_sync(FULL, e, (base + (s0 < LE ? s0 : 0)) & 31);
    const uint64_t x1 = __shfl_sync(FULL, e, (base + (s1 < LE ? s1 : 0)) & 31);
    const uint64_t h0 = s0 < LE ? x0 : 0, h1 = s1 < LE ? x1 : 0;
    const uint64_t H = hs ? (h0 >> hs) | (h1 << (64 - hs)) : h0;
    const uint64_t Hp = __shfl_up_sync(FULL, H, 1);
    const uint64_t Hb = L > 0 ? Hp : 0; // limb L - 1 of H

    if (!act || L >= ml) {
        return;
    }
    uint64_t r = e; // coefficients < m (masked below)
#pragma unroll
    for (int s = 0; s < F_TAIL_LEN; s++) { // + f_tail[s] z^s H
        const uint64_t Hs = s ? (H << (4 * s)) | (Hb >> (64 - 4 * s)) : H;
        r ^= gf16_vec_mul(Hs, d_f_tail_64[s]);
    }
    if (L == ml - 1) {
        r &= lastmask;
    }
    const unsigned char *t_b = t + (size_t)batch * m;
    unsigned char *y_b = y + (size_t)batch * m;
#pragma unroll
    for (int qq = 0; qq < 16; qq++) {
        const int idx = 16 * L + qq;
        if (idx < m) {
            y_b[idx] = t_b[idx] ^ (unsigned char)((r >> (4 * qq)) & 0xF);
        }
    }
}

#endif
