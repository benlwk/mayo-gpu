// Constant-time GF(16) solvers on bit planes: block solver and warp solver
#ifndef CHOI_SOLVE_CUH
#define CHOI_SOLVE_CUH
#include "choi_verify.cuh" // choi_mac

#define CSV_G       ((MAYO_P_m + 31) / 32)
#define CSV_THREADS (((MAYO_P_A_cols + 31) / 32) * 32)
#define CSV_WARPS   (CSV_THREADS / 32)

__device__ __forceinline__ uint32_t csv_range(int lo, int hi, int g)
{
    const int s = lo - 32 * g, e = hi - 32 * g;
    const uint32_t from = (s <= 0) ? 0xffffffffu : (s >= 32 ? 0u : (0xffffffffu << s));
    const uint32_t upto = (e >= 31) ? 0xffffffffu : (e < 0 ? 0u : (0xffffffffu >> (31 - e)));
    return from & upto;
}

#define CSV_INV_TABLE 0x834A5C2F67BDE910ULL

__device__ __forceinline__ uint32_t csv_gf16_mul(uint32_t a, uint32_t b)
{
    uint32_t r = 0;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        r ^= a & (0u - ((b >> t) & 1u));
        a = ((a << 1) & 0xFu) ^ (((a >> 3) & 1u) * 3u); // a * x mod x^4 + x + 1
    }
    return r;
}

__device__ __forceinline__ uint32_t csv_get(const uint32_t col[CSV_G][4], int row)
{
    uint32_t v = 0;
#pragma unroll
    for (int g = 0; g < CSV_G; g++) {
        const uint32_t sel = 0u - (uint32_t)(g == (row >> 5));
        const int b = row & 31;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            v |= (((col[g][d] >> b) & 1u) & sel) << d;
        }
    }
    return v;
}

#define CSV_PAIRS (MAYO_P_k * (MAYO_P_k + 1) / 2)
#define CSV_GE    ((MAYO_P_m + CSV_PAIRS + 31) / 32) // groups of a column before the f(z) fold
#define CSV_GU    ((CSV_PAIRS + 31) / 32)            // groups of the overflow coefficients

// BUILD = false
template <bool BUILD>
__global__ void __launch_bounds__(CSV_THREADS)
    choi_solve_t(const unsigned char *__restrict__ A, const uint64_t *__restrict__ VL,
                 const unsigned char *__restrict__ yv, const unsigned char *__restrict__ rv,
                 unsigned char *__restrict__ x, int *__restrict__ sol_batch_out, int *__restrict__ solved)
{
    constexpr int m = MAYO_P_m, ncols = MAYO_P_A_cols, ko = MAYO_P_k * MAYO_P_o;
    constexpr int k = MAYO_P_k, o = MAYO_P_o, ml = MAYO_P_m_vec_limbs;
    const int batch = blockIdx.x, j = threadIdx.x, lane = j & 31, warp = j >> 5;
    const bool own = j < ncols;

    __shared__ uint32_t sh_col[CSV_G][4];                               // back-substitution: the RHS column
    __shared__ uint32_t sh_sel[CSV_G], sh_bit[CSV_G], sh_R, sh_nonzero; // pivot data
    __shared__ uint4 sh_X4[4][CSV_G];

    uint32_t col[CSV_G][4];
#pragma unroll
    for (int g = 0; g < CSV_G; g++) {
        col[g][0] = col[g][1] = col[g][2] = col[g][3] = 0;
    }
    uint32_t rj = 0;
    if (!BUILD) {
        const unsigned char *A_b = A + (size_t)batch * m * ncols;
        if (own) {
            for (int r = 0; r < m; r++) {
                const uint32_t e = A_b[r * ncols + j];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    col[r >> 5][d] |= ((e >> d) & 1u) << (r & 31);
                }
            }
        }
    } else {
        __shared__ uint32_t sh_M[k * o][CSV_G][4];
        __shared__ unsigned char sh_lab[k * k];
        __shared__ uint32_t sh_red[CSV_WARPS][CSV_G][4];
        if (j < CSV_PAIRS) {
            const int a = d_whip_pos[2 * j], b = d_whip_pos[2 * j + 1];
            sh_lab[a * k + b] = (unsigned char)j;
            sh_lab[b * k + a] = (unsigned char)j;
        }

        const uint64_t *VL_b = VL + (size_t)batch * k * o * ml;
        for (int vec = j; vec < k * o; vec += CSV_THREADS) {
            const uint64_t *src = VL_b + (size_t)vec * ml;
            uint32_t w[CSV_G][4] = {};
#pragma unroll
            for (int g64 = 0; g64 < MAYO_P_bs_groups; g64++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    w[2 * g64][d] = (uint32_t)src[4 * g64 + d];
                    w[2 * g64 + 1][d] = (uint32_t)(src[4 * g64 + d] >> 32);
                }
#pragma unroll
            for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
                const uint64_t t = src[4 * MAYO_P_bs_groups + tw];
#pragma unroll
                for (int q = 0; q < 16; q++) {
                    const int deg = MAYO_P_bs_bulk + 16 * tw + q;
                    if (deg >= m) {
                        break;
                    }
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        w[deg >> 5][d] |= (uint32_t)((t >> (4 * q + d)) & 1) << (deg & 31);
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < CSV_G; g++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    sh_M[vec][g][d] = w[g][d];
                }
        }
        __syncthreads();

        uint32_t contrib[CSV_G][4] = {};
        if (j < ko) {
            const int i = j / o, c = j % o;
            uint32_t ext[CSV_GE][4] = {};
            for (int s = 0; s < k; s++) { // public loop and public shifts
                const int l = sh_lab[i * k + s], ws = l >> 5, bs = l & 31;
                const uint32_t(*src)[4] = sh_M[s * o + c];
#pragma unroll
                for (int w = 0; w < CSV_GE; w++) {
                    const int hi = w - ws, lo = hi - 1;
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const uint32_t vh = (hi >= 0 && hi < CSV_G) ? src[hi][d] : 0u;
                        const uint32_t vl = (lo >= 0 && lo < CSV_G) ? src[lo][d] : 0u;
                        ext[w][d] ^= __funnelshift_l(vl, vh, bs);
                    }
                }
            }

            uint32_t U[CSV_GU][4];
#pragma unroll
            for (int u = 0; u < CSV_GU; u++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const int w0 = (m >> 5) + u;
                    const uint32_t a0 = (w0 < CSV_GE) ? ext[w0 < CSV_GE ? w0 : 0][d] : 0u;
                    const uint32_t a1 = (w0 + 1 < CSV_GE) ? ext[w0 + 1 < CSV_GE ? w0 + 1 : 0][d] : 0u;
                    U[u][d] = __funnelshift_r(a0, a1, m & 31);
                }
#pragma unroll
            for (int g = 0; g < CSV_G; g++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const uint32_t keep = (32 * (g + 1) <= m) ? 0xffffffffu : csv_range(0, m - 1, g);
                    col[g][d] = ext[g][d] & keep;
                }
#pragma unroll
            for (int s = 0; s < 4; s++) {
                uint32_t V[CSV_GU][4] = {};
#pragma unroll
                for (int u = 0; u < CSV_GU; u++) {
                    choi_mac(V[u], d_f_tail[s], U[u]);
                }
#pragma unroll
                for (int g = 0; g < CSV_G; g++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const uint32_t vh = (g < CSV_GU) ? V[g < CSV_GU ? g : 0][d] : 0u;
                        const uint32_t vl =
                            (g >= 1 && g - 1 < CSV_GU) ? V[(g >= 1 && g - 1 < CSV_GU) ? g - 1 : 0][d] : 0u;
                        col[g][d] ^= __funnelshift_l(vl, vh, s);
                    }
            }
            rj = rv[(size_t)batch * ko + j];
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                choi_mac(contrib[g], rj, col[g]);
            }
        }
        // RHS = y + sum_j r_j col_j
#pragma unroll
        for (int g = 0; g < CSV_G; g++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                const uint32_t v = __reduce_xor_sync(0xffffffffu, contrib[g][d]);
                if (lane == 0) {
                    sh_red[warp][g][d] = v;
                }
            }
        __syncthreads();
        if (j == ncols - 1) {
            const unsigned char *y_b = yv + (size_t)batch * m;
            for (int r = 0; r < m; r++) {
                const uint32_t e = y_b[r];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    col[r >> 5][d] |= ((e >> d) & 1u) << (r & 31);
                }
            }
#pragma unroll
            for (int w = 0; w < CSV_WARPS; w++)
#pragma unroll
                for (int g = 0; g < CSV_G; g++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        col[g][d] ^= sh_red[w][g][d];
                    }
        }
    }

    int pivot_row = 0;                   // same value in every thread, secret
    for (int pc = 0; pc < ncols; pc++) { // public loop
        int lower = pc + m - ncols;
        if (lower < 0) {
            lower = 0;
        }
        int upper = pc;
        if (upper > m - 1) {
            upper = m - 1;
        }
        int search = upper + 32;
        if (search > m - 1) {
            search = m - 1;
        }

        const int g0 = lower >> 5;

        if (j == pc) {
            const int from = pivot_row > lower ? pivot_row : lower;
            int p = search;
#pragma unroll
            for (int g = CSV_G - 1; g >= 0; g--) {
                if (g < g0) {
                    continue;
                }
                const uint32_t nz = col[g][0] | col[g][1] | col[g][2] | col[g][3];
                const int f = __ffs(nz & csv_range(from, search, g));
                p = (f != 0) ? 32 * g + f - 1 : p;
            }
            uint32_t pivot = 0, sel[CSV_G];
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                sel[g] = csv_range(from, p, g);
                sh_sel[g] = sel[g];
            }
#pragma unroll
            for (int d = 0; d < 4; d++) {
                uint32_t xw = 0;
#pragma unroll
                for (int g = 0; g < CSV_G; g++) {
                    xw ^= (g < g0) ? 0u : col[g][d] & sel[g];
                }
                pivot |= (__popc(xw) & 1u) << d;
            }
            const uint32_t nonzero = 0u - (uint32_t)(pivot != 0);
            const uint32_t do_copy = nonzero & (0u - (uint32_t)(pivot_row >= lower && pivot_row <= upper));
            const int bfrom = pivot_row + 1 > lower ? pivot_row + 1 : lower;
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                if (g < g0) {
                    continue;
                }
                sh_bit[g] = csv_range(pivot_row, pivot_row, g) & do_copy;
                const uint32_t below = csv_range(bfrom, m - 1, g);
                uint32_t y0 = col[g][0] & below, y1 = col[g][1] & below, y2 = col[g][2] & below,
                         y3 = col[g][3] & below;
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    sh_X4[t][g] = make_uint4(y0, y1, y2, y3);
                    const uint32_t t3 = y3;
                    y3 = y2;
                    y2 = y1;
                    y1 = y0 ^ t3;
                    y0 = t3;
                }
            }

            uint32_t c = (uint32_t)(CSV_INV_TABLE >> (4 * pivot)) & 0xFu, R = 0;
#pragma unroll
            for (int d = 0; d < 4; d++) { // c = inv * x^d
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    R |= ((c >> t) & 1u) << (8 * t + d);
                }
                c = ((c << 1) & 0xFu) ^ (((c >> 3) & 1u) * 3u);
            }
            sh_R = R;
            sh_nonzero = nonzero & 1u;
        }
        __syncthreads();

        uint32_t pr = 0;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            uint32_t xw = 0;
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                xw ^= (g < g0) ? 0u : col[g][d] & sh_sel[g];
            }
            pr |= (__popc(xw) & 1u) << d;
        }
        const uint32_t R = sh_R;
        uint32_t msk[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            msk[t] = 0u - (__popc(pr & (R >> (8 * t)) & 0xFu) & 1u);
        }

#pragma unroll
        for (int g = 0; g < CSV_G; g++) {
            if (g < g0) {
                continue;
            }

            const uint32_t bit = sh_bit[g];
#pragma unroll
            for (int d = 0; d < 4; d++) {
                col[g][d] = (col[g][d] & ~bit) | (bit & msk[d]);
            }

#pragma unroll
            for (int t = 0; t < 4; t++) {
                const uint4 X = sh_X4[t][g];
                col[g][0] ^= X.x & msk[t];
                col[g][1] ^= X.y & msk[t];
                col[g][2] ^= X.z & msk[t];
                col[g][3] ^= X.w & msk[t];
            }
        }
        pivot_row += (int)sh_nonzero;
        __syncthreads();
    }

    const int full = __syncthreads_or(own && j < ko && csv_get(col, m - 1) != 0);
    if (!full) {
        return;
    }

    // Back-substitution, reference semantics
    __shared__ unsigned char sh_rhs0[m];
    __shared__ uint32_t sh_part[2][CSV_WARPS], sh_bal[2][CSV_WARPS];
    if (j == ncols - 1) {
#pragma unroll
        for (int g = 0; g < CSV_G; g++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                sh_col[g][d] = col[g][d];
            }
    }
    __syncthreads();
    for (int r = j; r < m; r += CSV_THREADS) {
        uint32_t v = 0;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            v |= ((sh_col[r >> 5][d] >> (r & 31)) & 1u) << d;
        }
        sh_rhs0[r] = (unsigned char)v;
    }
    __syncthreads();

    uint32_t xacc = 0;

#pragma unroll
    for (int g = CSV_G - 1; g >= 0; g--) {
        for (int rr = 31; rr >= 0; rr--) {
            const int row = 32 * g + rr;
            if (row >= m) {
                continue;
            }
            int cub = row + 32 / (m - row);
            if (cub > ko) {
                cub = ko;
            }
            const bool inwin = own && j >= row && j <= cub;
            const uint32_t a = ((col[g][0] >> rr) & 1u) | (((col[g][1] >> rr) & 1u) << 1) |
                               (((col[g][2] >> rr) & 1u) << 2) | (((col[g][3] >> rr) & 1u) << 3);
            const uint32_t part = __reduce_xor_sync(0xffffffffu, csv_gf16_mul(xacc, a));
            const uint32_t bal = __ballot_sync(0xffffffffu, inwin && a != 0);
            const int buf = row & 1;
            if (lane == 0) {
                sh_part[buf][warp] = part;
                sh_bal[buf][warp] = bal;
            }
            __syncthreads();
            uint32_t u = sh_rhs0[row];
            int win = -1;
#pragma unroll
            for (int w = CSV_WARPS - 1; w >= 0; w--) {
                u ^= sh_part[buf][w];
                const int f = __ffs(sh_bal[buf][w]);
                win = (f != 0) ? 32 * w + f - 1 : win;
            }
            xacc ^= (j == win) ? u : 0u; // select, not a branch on a secret
        }
    }

    if (own) {
        unsigned char *xp = x + (size_t)batch * (MAYO_P_k * MAYO_P_n) + j;
        if (BUILD && j < ko) {
            *xp = (unsigned char)(rj ^ xacc);
        } else {
            *xp ^= (unsigned char)xacc;
        }
    }
    if (j == 0) {
        atomicMin(sol_batch_out, batch);
        if (solved) {
            solved[batch] = 1;
        }
    }
}

#define CSW_NC  ((MAYO_P_A_cols + 31) / 32)
#define CSW_WPB 2 // warps (signatures) per block

__global__ void __launch_bounds__(32 * CSW_WPB)
    choi_solve_w(const uint64_t *__restrict__ VL, const unsigned char *__restrict__ yv,
                 const unsigned char *__restrict__ rv, unsigned char *__restrict__ x,
                 int *__restrict__ sol_batch_out, int *__restrict__ solved, int nb)
{
    constexpr int m = MAYO_P_m, ncols = MAYO_P_A_cols, ko = MAYO_P_k * MAYO_P_o;
    constexpr int k = MAYO_P_k, o = MAYO_P_o, ml = MAYO_P_m_vec_limbs, NC = CSW_NC;
    constexpr unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    const int batch = blockIdx.x * CSW_WPB + wi;
    if (batch >= nb) {
        return; // the whole warp
    }

    __shared__ uint32_t sh_M[CSW_WPB][k * o][CSV_G][4];
    __shared__ unsigned char sh_lab[CSW_WPB][k * k];
    for (int p = lane; p < CSV_PAIRS; p += 32) {
        const int a = d_whip_pos[2 * p], b = d_whip_pos[2 * p + 1];
        sh_lab[wi][a * k + b] = (unsigned char)p;
        sh_lab[wi][b * k + a] = (unsigned char)p;
    }

    const uint64_t *VL_b = VL + (size_t)batch * ko * ml;
    for (int vec = lane; vec < ko; vec += 32) {
        const uint64_t *src = VL_b + (size_t)vec * ml;
        uint32_t w[CSV_G][4] = {};
#pragma unroll
        for (int g64 = 0; g64 < MAYO_P_bs_groups; g64++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                w[2 * g64][d] = (uint32_t)src[4 * g64 + d];
                w[2 * g64 + 1][d] = (uint32_t)(src[4 * g64 + d] >> 32);
            }
#pragma unroll
        for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
            const uint64_t t = src[4 * MAYO_P_bs_groups + tw];
#pragma unroll
            for (int q = 0; q < 16; q++) {
                const int deg = MAYO_P_bs_bulk + 16 * tw + q;
                if (deg >= m) {
                    break;
                }
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    w[deg >> 5][d] |= (uint32_t)((t >> (4 * q + d)) & 1) << (deg & 31);
                }
            }
        }
#pragma unroll
        for (int g = 0; g < CSV_G; g++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                sh_M[wi][vec][g][d] = w[g][d];
            }
    }
    __syncwarp();

    uint32_t col[NC][CSV_G][4], rj[NC], contrib[CSV_G][4] = {};
#pragma unroll
    for (int s = 0; s < NC; s++) {
        const int j = 32 * s + lane;
#pragma unroll
        for (int g = 0; g < CSV_G; g++) {
            col[s][g][0] = col[s][g][1] = col[s][g][2] = col[s][g][3] = 0;
        }
        rj[s] = 0;
        if (j < ko) {
            const int i = j / o, c = j % o;
            uint32_t ext[CSV_GE][4] = {};
            for (int t = 0; t < k; t++) { // public loop and public shifts
                const int l = sh_lab[wi][i * k + t], ws = l >> 5, bs = l & 31;
                const uint32_t(*src)[4] = sh_M[wi][t * o + c];
#pragma unroll
                for (int w = 0; w < CSV_GE; w++) {
                    const int hi = w - ws, lo = hi - 1;
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const uint32_t vh = (hi >= 0 && hi < CSV_G) ? src[hi][d] : 0u;
                        const uint32_t vl = (lo >= 0 && lo < CSV_G) ? src[lo][d] : 0u;
                        ext[w][d] ^= __funnelshift_l(vl, vh, bs);
                    }
                }
            }
            uint32_t U[CSV_GU][4];
#pragma unroll
            for (int u = 0; u < CSV_GU; u++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const int w0 = (m >> 5) + u;
                    const uint32_t a0 = (w0 < CSV_GE) ? ext[w0 < CSV_GE ? w0 : 0][d] : 0u;
                    const uint32_t a1 = (w0 + 1 < CSV_GE) ? ext[w0 + 1 < CSV_GE ? w0 + 1 : 0][d] : 0u;
                    U[u][d] = __funnelshift_r(a0, a1, m & 31);
                }
#pragma unroll
            for (int g = 0; g < CSV_G; g++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const uint32_t keep = (32 * (g + 1) <= m) ? 0xffffffffu : csv_range(0, m - 1, g);
                    col[s][g][d] = ext[g][d] & keep;
                }
#pragma unroll
            for (int sf = 0; sf < 4; sf++) {
                uint32_t V[CSV_GU][4] = {};
#pragma unroll
                for (int u = 0; u < CSV_GU; u++) {
                    choi_mac(V[u], d_f_tail[sf], U[u]);
                }
#pragma unroll
                for (int g = 0; g < CSV_G; g++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const uint32_t vh = (g < CSV_GU) ? V[g < CSV_GU ? g : 0][d] : 0u;
                        const uint32_t vl =
                            (g >= 1 && g - 1 < CSV_GU) ? V[(g >= 1 && g - 1 < CSV_GU) ? g - 1 : 0][d] : 0u;
                        col[s][g][d] ^= __funnelshift_l(vl, vh, sf);
                    }
            }
            rj[s] = rv[(size_t)batch * ko + j];
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                choi_mac(contrib[g], rj[s], col[s][g]);
            }
        }
    }

    constexpr int SR = (ncols - 1) >> 5, LR = (ncols - 1) & 31;
    const unsigned char *y_b = yv + (size_t)batch * m;
#pragma unroll
    for (int g = 0; g < CSV_G; g++) {
        const uint32_t e = (32 * g + lane < m) ? y_b[32 * g + lane] : 0u;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            const uint32_t v = __ballot_sync(FULL, (e >> d) & 1u) ^ __reduce_xor_sync(FULL, contrib[g][d]);
            col[SR][g][d] = (lane == LR) ? v : col[SR][g][d];
        }
    }

    int pivot_row = 0; // same value in every lane, secret
#pragma unroll
    for (int s0 = 0; s0 < NC; s0++) {
        const int lend = (32 * (s0 + 1) <= ncols) ? 32 : ncols - 32 * s0;
#pragma unroll 1
        for (int l = 0; l < lend; l++) { // public loop: pivot column pc
            const int pc = 32 * s0 + l;
            int lower = pc + m - ncols;
            if (lower < 0) {
                lower = 0;
            }
            int upper = pc;
            if (upper > m - 1) {
                upper = m - 1;
            }
            int search = upper + 32;
            if (search > m - 1) {
                search = m - 1;
            }
            const int g0 = lower >> 5;

            uint32_t pcol[CSV_G][4];
#pragma unroll
            for (int g = 0; g < CSV_G; g++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    pcol[g][d] = (g < g0) ? 0u : __shfl_sync(FULL, col[s0][g][d], l);
                }

            const int from = pivot_row > lower ? pivot_row : lower;
            int p = search;
#pragma unroll
            for (int g = CSV_G - 1; g >= 0; g--) {
                const uint32_t nz = pcol[g][0] | pcol[g][1] | pcol[g][2] | pcol[g][3];
                const int f = __ffs(nz & csv_range(from, search, g));
                p = (f != 0) ? 32 * g + f - 1 : p;
            }
            uint32_t sel[CSV_G], pivot = 0;
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                sel[g] = csv_range(from, p, g);
            }
#pragma unroll
            for (int d = 0; d < 4; d++) {
                uint32_t xw = 0;
#pragma unroll
                for (int g = 0; g < CSV_G; g++) {
                    xw ^= pcol[g][d] & sel[g];
                }
                pivot |= (__popc(xw) & 1u) << d;
            }
            const uint32_t nonzero = 0u - (uint32_t)(pivot != 0);
            const uint32_t do_copy = nonzero & (0u - (uint32_t)(pivot_row >= lower && pivot_row <= upper));
            const int bfrom = pivot_row + 1 > lower ? pivot_row + 1 : lower;
            uint32_t bit[CSV_G];
#pragma unroll
            for (int g = 0; g < CSV_G; g++) {
                bit[g] = csv_range(pivot_row, pivot_row, g) & do_copy;
                const uint32_t below = csv_range(bfrom, m - 1, g);
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    pcol[g][d] &= below; // pivot column below the pivot row
                }
            }
            uint32_t cinv = (uint32_t)(CSV_INV_TABLE >> (4 * pivot)) & 0xFu, R = 0;
#pragma unroll
            for (int d = 0; d < 4; d++) {
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    R |= ((cinv >> t) & 1u) << (8 * t + d);
                }
                cinv = ((cinv << 1) & 0xFu) ^ (((cinv >> 3) & 1u) * 3u);
            }

#pragma unroll
            for (int s = 0; s < NC; s++) {
                uint32_t pr = 0;
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    uint32_t xw = 0;
#pragma unroll
                    for (int g = 0; g < CSV_G; g++) {
                        xw ^= (g < g0) ? 0u : col[s][g][d] & sel[g];
                    }
                    pr |= (__popc(xw) & 1u) << d;
                }
                uint32_t msk[4];
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    msk[t] = 0u - (__popc(pr & (R >> (8 * t)) & 0xFu) & 1u);
                }
#pragma unroll
                for (int g = 0; g < CSV_G; g++) {
                    if (g < g0) {
                        continue;
                    }
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        col[s][g][d] = (col[s][g][d] & ~bit[g]) | (bit[g] & msk[d]);
                    }
                    uint32_t y0 = pcol[g][0], y1 = pcol[g][1], y2 = pcol[g][2], y3 = pcol[g][3];
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        col[s][g][0] ^= y0 & msk[t];
                        col[s][g][1] ^= y1 & msk[t];
                        col[s][g][2] ^= y2 & msk[t];
                        col[s][g][3] ^= y3 & msk[t];
                        const uint32_t t3 = y3;
                        y3 = y2;
                        y2 = y1;
                        y1 = y0 ^ t3;
                        y0 = t3;
                    }
                }
            }
            pivot_row += (int)(nonzero & 1u);
        }
    }

    bool lastnz = false;
#pragma unroll
    for (int s = 0; s < NC; s++) {
        lastnz |= (32 * s + lane < ko) && csv_get(col[s], m - 1) != 0;
    }
    if (!__any_sync(FULL, lastnz)) {
        return;
    }

    uint32_t rhs[CSV_G][4], xacc[NC];
#pragma unroll
    for (int g = 0; g < CSV_G; g++)
#pragma unroll
        for (int d = 0; d < 4; d++) {
            rhs[g][d] = __shfl_sync(FULL, col[SR][g][d], LR);
        }
#pragma unroll
    for (int s = 0; s < NC; s++) {
        xacc[s] = 0;
    }
#pragma unroll
    for (int g = CSV_G - 1; g >= 0; g--) {
        for (int rr = 31; rr >= 0; rr--) { // public loop over the rows
            const int row = 32 * g + rr;
            if (row >= m) {
                continue;
            }
            int cub = row + 32 / (m - row);
            if (cub > ko) {
                cub = ko;
            }
            uint32_t part = 0;
            int win = -1;
            uint32_t bal[NC];
#pragma unroll
            for (int s = 0; s < NC; s++) {
                const int j = 32 * s + lane;
                const uint32_t a = ((col[s][g][0] >> rr) & 1u) | (((col[s][g][1] >> rr) & 1u) << 1) |
                                   (((col[s][g][2] >> rr) & 1u) << 2) | (((col[s][g][3] >> rr) & 1u) << 3);
                part ^= csv_gf16_mul(xacc[s], a);
                bal[s] = __ballot_sync(FULL, j < ncols && j >= row && j <= cub && a != 0);
            }
#pragma unroll
            for (int s = NC - 1; s >= 0; s--) { // first nonzero window column
                const int f = __ffs(bal[s]);
                win = (f != 0) ? 32 * s + f - 1 : win;
            }
            const uint32_t u = (((rhs[g][0] >> rr) & 1u) | (((rhs[g][1] >> rr) & 1u) << 1) |
                                (((rhs[g][2] >> rr) & 1u) << 2) | (((rhs[g][3] >> rr) & 1u) << 3)) ^
                               __reduce_xor_sync(FULL, part);
#pragma unroll
            for (int s = 0; s < NC; s++) {
                xacc[s] ^= (32 * s + lane == win) ? u : 0u; // select
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NC; s++) {
        const int j = 32 * s + lane;
        if (j >= ncols) {
            continue;
        }
        unsigned char *xp = x + (size_t)batch * (MAYO_P_k * MAYO_P_n) + j;
        if (j < ko) {
            *xp = (unsigned char)(rj[s] ^ xacc[s]);
        } else {
            *xp ^= (unsigned char)xacc[s];
        }
    }
    if (lane == 0) {
        atomicMin(sol_batch_out, batch);
        if (solved) {
            solved[batch] = 1;
        }
    }
}

#endif
