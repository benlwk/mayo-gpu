// Fused Tensor-Core verification, after Choi et al. (ePrint 2026/2577)
#ifndef CHOI_VERIFY_CUH
#define CHOI_VERIFY_CUH
#include "tc_ps.cuh"

#if TC_KARATSUBA
#error "choi_verify.cuh needs the 4-plane key fragments (TC_KARATSUBA 0)"
#endif

#define CH_G        ((MAYO_P_m + 31) / 32) // groups of 32 coefficients
#define CH_NI       ((MAYO_P_k + 3) / 4)
#define CH_SB_WORDS ((size_t)TC_CT * TC_KT * 4 * 32)
#define CH_NP       (MAYO_P_k * (MAYO_P_k + 1) / 2) // index pairs i <= j
#define CH_GE       (CH_G + (CH_NP - 1) / 32 + 1)   // plane words of the unreduced y
#define CH_YS       (8 / MAYO_P_k + 2)              // signatures a tile of 8 columns can touch
__constant__ unsigned char choi_lab[MAYO_P_k * MAYO_P_k];

// B fragments
__global__ void choi_pack_S(const unsigned char *__restrict__ d_s, uint32_t *__restrict__ SB)
{
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (size_t)TC_CT * TC_KT * 32) {
        return;
    }
    const int lane = (int)(g & 31);
    const int kt = (int)((g >> 5) % TC_KT);
    const int ct = (int)((g >> 5) / TC_KT);
    const int col = ct * 8 + (lane >> 2);
    uint32_t w[4] = {0, 0, 0, 0};
    if (col < TC_COLS) {
        const unsigned char *s =
            d_s + (size_t)(col / MAYO_P_k) * (MAYO_P_k * MAYO_P_n) + (size_t)(col % MAYO_P_k) * MAYO_P_n;
        const int jb = kt * 128 + (lane & 3) * 32;
        for (int q = 0; q < 32; q++) {
            const int j = jb + q;
            const uint32_t e = (j < MAYO_P_n) ? s[j] : 0u;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                w[i] |= ((e >> i) & 1u) << q;
            }
        }
    }
    uint32_t *dst = SB + ((size_t)ct * TC_KT + kt) * 128 + lane;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[i * 32] = w[i];
    }
}

__device__ __forceinline__ void choi_mac(uint32_t acc[4], uint32_t a, const uint32_t x[4])
{
    uint32_t y0 = x[0], y1 = x[1], y2 = x[2], y3 = x[3];
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const uint32_t msk = 0u - ((a >> t) & 1u);
        acc[0] ^= y0 & msk;
        acc[1] ^= y1 & msk;
        acc[2] ^= y2 & msk;
        acc[3] ^= y3 & msk;
        // y = y * x: (y0, y1, y2, y3) -> (y3, y0 + y3, y1, y2)
        const uint32_t t3 = y3;
        y3 = y2;
        y2 = y1;
        y1 = y0 ^ t3;
        y0 = t3;
    }
}

// the four masks 0 - bit t of a (a < 16)
__device__ __forceinline__ void choi_masks(uint32_t a, uint32_t m[4])
{
    const uint32_t x = a * 0x10204080u;
    asm("prmt.b32 %0, %1, 0, 0x8888;" : "=r"(m[0]) : "r"(x));
    asm("prmt.b32 %0, %1, 0, 0x9999;" : "=r"(m[1]) : "r"(x));
    asm("prmt.b32 %0, %1, 0, 0xAAAA;" : "=r"(m[2]) : "r"(x));
    asm("prmt.b32 %0, %1, 0, 0xBBBB;" : "=r"(m[3]) : "r"(x));
}

// the same for two scalars
__device__ __forceinline__ void choi_masks2(uint32_t a0, uint32_t a1, uint32_t m[4])
{
    const uint32_t x0 = a0 * 0x10204080u, x1 = a1 * 0x10204080u;
    asm("prmt.b32 %0, %1, %2, 0xCC88;" : "=r"(m[0]) : "r"(x0), "r"(x1));
    asm("prmt.b32 %0, %1, %2, 0xDD99;" : "=r"(m[1]) : "r"(x0), "r"(x1));
    asm("prmt.b32 %0, %1, %2, 0xEEAA;" : "=r"(m[2]) : "r"(x0), "r"(x1));
    asm("prmt.b32 %0, %1, %2, 0xFFBB;" : "=r"(m[3]) : "r"(x0), "r"(x1));
}

// CHOI_SPS_IMADMAC
#ifndef CHOI_SPS_IMADMAC
#define CHOI_SPS_IMADMAC 0
#endif
__constant__ uint32_t choi_mulc[5] = {0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 2u};
__device__ __forceinline__ void choi_bits(uint32_t a, uint32_t bt[4])
{
#pragma unroll
    for (int t = 0; t < 4; t++) {
        bt[t] = __umulhi(a * choi_mulc[t], choi_mulc[4]);
    }
}

// A fragments in lane-major order
__global__ void frag_lane_major(const uint32_t *__restrict__ in, uint4 *__restrict__ out, size_t nfrag)
{
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nfrag * 32) {
        return;
    }
    const size_t f = g >> 5, lane = g & 31;
    const uint32_t *s = in + f * 128 + lane;
    out[g] = make_uint4(s[0], s[32], s[64], s[96]);
}

// CHOI_SPS_XJ = 1
#ifndef CHOI_SPS_XJ
#define CHOI_SPS_XJ 0
#endif
#ifndef CHOI_SPS_PAIR
#define CHOI_SPS_PAIR 1
#endif
#ifndef CHOI_SPS_NPL
#define CHOI_SPS_NPL 7
#endif

__global__ void frag_lane_major_xj(const uint32_t *__restrict__ in, uint4 *__restrict__ out, size_t nfrag)
{
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nfrag * 32) {
        return;
    }
    const size_t f = g >> 5, lane = g & 31;
    const uint32_t *s = in + f * 128 + lane;
    uint32_t y0 = s[0], y1 = s[32], y2 = s[64], y3 = s[96];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        out[(f * 4 + j) * 32 + lane] = make_uint4(y0, y1, y2, y3);
        const uint32_t t3 = y3;
        y3 = y2;
        y2 = y1;
        y1 = y0 ^ t3;
        y0 = t3; // y *= x
    }
}

template <int N_, int RT_, int KT_, int CT_, int COLS_>
__global__ void __launch_bounds__(128)
    choi_fused_SPS(const uint4 *__restrict__ PA, const uint32_t *__restrict__ SB,
                   const unsigned char *__restrict__ d_s, uint64_t *__restrict__ d_SPS,
                   const uint4 *__restrict__ lamP, int lam_blocks, uint32_t *__restrict__ yext)
{
    const int n = N_, k = MAYO_P_k, ml = MAYO_P_m_vec_limbs;
    const int lane = threadIdx.x & 31;
    const int ct = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (ct >= CT_) {
        return;
    }

    // yext != NULL (verify)
    __shared__ uint32_t sh_y[4][CH_YS][CH_GE][4];
    uint32_t(*shy)[CH_GE][4] = sh_y[(threadIdx.x >> 5) & 3];
    const int bfirst = (ct * 8) / MAYO_P_k;
    if (yext != NULL) {
        for (int x = lane; x < CH_YS * CH_GE * 4; x += 32) {
            (&shy[0][0][0])[x] = 0;
        }
        __syncwarp();
    }

    uint32_t B[KT_][4];
#pragma unroll
    for (int kt = 0; kt < KT_; kt++)
#pragma unroll
        for (int d = 0; d < 4; d++) {
            B[kt][d] = __ldg(SB + ((size_t)ct * KT_ + kt) * 128 + d * 32 + lane);
        }

    const int oc = lane >> 2, qq = lane & 3;
    const int col = ct * 8 + oc;
    const bool cvalid = col < COLS_;
    const int b = cvalid ? col / k : 0, c = cvalid ? col % k : 0;
    const unsigned char *sb = d_s + (size_t)b * (k * n);

#pragma unroll
    for (int g = 0; g < CH_G; g++) {
        const int qend = (32 * (g + 1) <= MAYO_P_m) ? 32 : MAYO_P_m - 32 * g;

        const bool paired = CHOI_SPS_PAIR && qend <= 16 && CH_NI > 1;
        uint32_t H[CH_NI][4];
#pragma unroll
        for (int ii = 0; ii < CH_NI; ii++) {
            H[ii][0] = H[ii][1] = H[ii][2] = H[ii][3] = 0;
        }

#pragma unroll 1
        for (int rt = 0; rt < RT_; rt++) {
            const int kt0 = (KT_ > 1 && rt * 8 >= 128) ? 1 : 0; // upper triangular: skip zero K tile
#if CHOI_SPS_XJ
            uint32_t pl[2][4] = {};
#pragma unroll 2
            for (int q = 0; q < qend; q++) {
                const int p = 32 * g + q;
                const uint4 *pa = PA + ((size_t)(p * RT_ + rt) * KT_) * 128 + lane;
                int D[4][2];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    D[d][0] = D[d][1] = 0;
                }
#pragma unroll
                for (int kt = 0; kt < KT_; kt++) {
                    if (kt < kt0) {
                        continue;
                    }
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint4 a4 = __ldg(pa + (kt * 4 + j) * 32);
                        tc_bmma(D[0][0], D[0][1], a4.x, B[kt][j]);
                        tc_bmma(D[1][0], D[1][1], a4.y, B[kt][j]);
                        tc_bmma(D[2][0], D[2][1], a4.z, B[kt][j]);
                        tc_bmma(D[3][0], D[3][1], a4.w, B[kt][j]);
                    }
                }
#pragma unroll
                for (int e = 0; e < 2; e++)
#pragma unroll
                    for (int d = 0; d < 4; d++) {
                        pl[e][d] = __funnelshift_r(pl[e][d], (uint32_t)D[d][e], 1);
                    }
            }
            uint32_t pr[2][4]; // bit q is at 32 - qend + q
#pragma unroll
            for (int e = 0; e < 2; e++)
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    pr[e][d] = pl[e][d] >> (32 - qend);
                }
#else
            // CHOI_SPS_NPL < 7
            constexpr int NPL = CHOI_SPS_NPL;
            uint32_t pl[2][NPL] = {};
#pragma unroll 2
            for (int q = 0; q < qend; q++) {
                const int p = 32 * g + q;
                const uint4 *pa = PA + ((size_t)(p * RT_ + rt) * KT_) * 32 + lane;
                int D[NPL][2];
#pragma unroll
                for (int d = 0; d < NPL; d++) {
                    D[d][0] = D[d][1] = 0;
                }
#pragma unroll
                for (int kt = 0; kt < KT_; kt++) {
                    if (kt < kt0) {
                        continue;
                    }
                    const uint4 a4 = __ldg(pa + kt * 32);
                    const uint32_t A[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int deg = i + j;
                            if (deg < NPL) {
                                tc_bmma(D[deg][0], D[deg][1], A[i], B[kt][j]);
                            } else {
                                tc_bmma(D[deg - 4][0], D[deg - 4][1], A[i], B[kt][j]);
                                tc_bmma(D[deg - 3][0], D[deg - 3][1], A[i], B[kt][j]);
                            }
                        }
                }
#pragma unroll
                for (int e = 0; e < 2; e++)
#pragma unroll
                    for (int d = 0; d < NPL; d++) {
                        pl[e][d] = __funnelshift_r(pl[e][d], (uint32_t)D[d][e], 1);
                    }
            }

            uint32_t pr[2][4];
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t c4 = NPL > 4 ? pl[e][NPL > 4 ? 4 : 0] : 0u,
                               c5 = NPL > 5 ? pl[e][NPL > 5 ? 5 : 0] : 0u;
                const uint32_t c6 = NPL > 6 ? pl[e][NPL > 6 ? 6 : 0] : 0u;
                pr[e][0] = (pl[e][0] ^ c4) >> (32 - qend);
                pr[e][1] = (pl[e][1] ^ c4 ^ c5) >> (32 - qend);
                pr[e][2] = (pl[e][2] ^ c5 ^ c6) >> (32 - qend);
                pr[e][3] = (pl[e][3] ^ c6) >> (32 - qend);
            }
#endif
            // Verify fold (lamP != NULL)
            if (lamP != NULL) {
                const int rowE = rt * 8 + (lane >> 2);
                const uint32_t qm = (qend == 32) ? 0xffffffffu : ((1u << qend) - 1u);
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int colE = ct * 8 + 2 * (lane & 3) + e;
                    if (colE < COLS_ && colE % k == 0 && rowE < n) {
                        const int s0 = rowE * MAYO_P_m + 32 * g, blk = s0 >> 5, sh = s0 & 31;
                        const uint4 *Lb = lamP + (size_t)(colE / k) * lam_blocks;
                        const uint4 lo = __ldg(Lb + blk);
                        const uint4 hi =
                            (blk + 1 < lam_blocks) ? __ldg(Lb + blk + 1) : make_uint4(0, 0, 0, 0);
                        pr[e][0] ^= __funnelshift_r(lo.x, hi.x, sh) & qm;
                        pr[e][1] ^= __funnelshift_r(lo.y, hi.y, sh) & qm;
                        pr[e][2] ^= __funnelshift_r(lo.z, hi.z, sh) & qm;
                        pr[e][3] ^= __funnelshift_r(lo.w, hi.w, sh) & qm;
                    }
                }
            }
            // lane L holds rows rt*8 + L/4, columns (L%4)*2 + e
#pragma unroll
            for (int r8 = 0; r8 < 8; r8++) {
                const int src = r8 * 4 + (oc >> 1);
                uint32_t w[4];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const uint32_t w0 = __shfl_sync(0xffffffffu, pr[0][d], src);
                    const uint32_t w1 = __shfl_sync(0xffffffffu, pr[1][d], src);
                    w[d] = (oc & 1) ? w1 : w0;
                }
                const int r = rt * 8 + r8;
                if (r >= n || !cvalid) {
                    continue;
                }

                uint32_t W[4][4];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    W[0][d] = paired ? w[d] * 0x10001u : w[d];
                }
#pragma unroll
                for (int t = 1; t < 4; t++) {
                    W[t][0] = W[t - 1][3];
                    W[t][1] = W[t - 1][0] ^ W[t - 1][3];
                    W[t][2] = W[t - 1][1];
                    W[t][3] = W[t - 1][2];
                }
                if (paired) {
#pragma unroll
                    for (int pi = 0; pi < (CH_NI + 1) / 2; pi++) {
                        const int i0 = qq + 8 * pi, i1 = i0 + 4;
                        if (i0 >= k) {
                            continue;
                        }
                        uint32_t m[4];
                        choi_masks2(sb[i0 * n + r], (2 * pi + 1 < CH_NI && i1 < k) ? sb[i1 * n + r] : 0u, m);
#pragma unroll
                        for (int d = 0; d < 4; d++) {
                            H[pi][d] ^=
                                (W[0][d] & m[0]) ^ (W[1][d] & m[1]) ^ (W[2][d] & m[2]) ^ (W[3][d] & m[3]);
                        }
                    }
                } else {
#pragma unroll
                    for (int ii = 0; ii < CH_NI; ii++) {
                        const int i = qq + 4 * ii;
                        if (i >= k) {
                            continue;
                        }
#if CHOI_SPS_IMADMAC
                        uint32_t bt[4];
                        choi_bits(sb[i * n + r], bt);
#pragma unroll
                        for (int d = 0; d < 4; d++) {
                            H[ii][d] ^=
                                (W[0][d] * bt[0]) ^ (W[1][d] * bt[1]) ^ (W[2][d] * bt[2]) ^ (W[3][d] * bt[3]);
                        }
#else
                        uint32_t m[4];
                        choi_masks(sb[i * n + r], m);
#pragma unroll
                        for (int d = 0; d < 4; d++) {
                            H[ii][d] ^=
                                (W[0][d] & m[0]) ^ (W[1][d] & m[1]) ^ (W[2][d] & m[2]) ^ (W[3][d] & m[3]);
                        }
#endif
                    }
                }
            }
        }

        // limbs 2g and 2g + 1 of this group
        if (!cvalid) {
            continue;
        }
#pragma unroll
        for (int ii = 0; ii < CH_NI; ii++) {
            const int i = qq + 4 * ii;
            if (i >= k) {
                continue;
            }
            uint64_t *dst = d_SPS + (size_t)b * (k * k * ml) + (size_t)(i * k + c) * ml;
            uint32_t Hi[4];
#pragma unroll
            for (int d = 0; d < 4; d++) {
                Hi[d] = paired ? H[ii >> 1][d] >> (16 * (ii & 1)) : H[ii][d];
            }
            if (yext != NULL) {
                const uint32_t qm = (qend == 32) ? 0xffffffffu : ((1u << qend) - 1u);
                const int lab = choi_lab[i * k + c], w0 = g + (lab >> 5), bs = lab & 31;
                uint32_t(*dy)[4] = shy[b - bfirst];
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const uint32_t h = Hi[d] & qm;
                    atomicXor(&dy[w0][d], h << bs);
                    if (bs != 0) {
                        atomicXor(&dy[w0 + 1][d], h >> (32 - bs));
                    }
                }
                continue;
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int l = 2 * g + h;
                if (l >= ml) {
                    break;
                }
                uint64_t v = 0;
#pragma unroll
                for (int t = 0; t < 16; t++) {
                    const int q = 16 * h + t;
                    if (32 * g + q >= MAYO_P_m) {
                        break;
                    }
                    const uint64_t nib = ((Hi[0] >> q) & 1u) | (((Hi[1] >> q) & 1u) << 1) |
                                         (((Hi[2] >> q) & 1u) << 2) | (((Hi[3] >> q) & 1u) << 3);
                    v |= nib << (4 * t);
                }
                dst[l] = v;
            }
        }
    }

    if (yext != NULL) {
        __syncwarp();
        constexpr int nsig = COLS_ / MAYO_P_k;
        for (int x = lane; x < CH_YS * CH_GE * 4; x += 32) {
            const int sl = x / (CH_GE * 4), bs = bfirst + sl;
            if (bs < nsig) {
                atomicXor(yext + (size_t)bs * (CH_GE * 4) + (x % (CH_GE * 4)), (&shy[0][0][0])[x]);
            }
        }
    }
}

__global__ void choi_rhs_from_ext(const uint32_t *__restrict__ yext, const unsigned char *__restrict__ t,
                                  unsigned char *__restrict__ y, int nsig)
{
    constexpr int m = MAYO_P_m, GU = (CH_NP + 31) / 32 + 1;
    const int sig = blockIdx.x * blockDim.x + threadIdx.x;
    if (sig >= nsig) {
        return;
    }
    uint32_t E[CH_GE][4];
#pragma unroll
    for (int w = 0; w < CH_GE; w++) {
        const uint4 v = __ldg((const uint4 *)(yext + (size_t)sig * (CH_GE * 4)) + w);
        E[w][0] = v.x;
        E[w][1] = v.y;
        E[w][2] = v.z;
        E[w][3] = v.w;
    }
    uint32_t U[GU][4];
#pragma unroll
    for (int u = 0; u < GU; u++)
#pragma unroll
        for (int d = 0; d < 4; d++) {
            const int w0 = m / 32 + u;
            const uint32_t a0 = (w0 < CH_GE) ? E[w0 < CH_GE ? w0 : 0][d] : 0u;
            const uint32_t a1 = (w0 + 1 < CH_GE) ? E[w0 + 1 < CH_GE ? w0 + 1 : 0][d] : 0u;
            U[u][d] = __funnelshift_r(a0, a1, m & 31);
        }
    uint32_t R[CH_G][4];
#pragma unroll
    for (int g = 0; g < CH_G; g++)
#pragma unroll
        for (int d = 0; d < 4; d++) {
            R[g][d] = E[g][d];
        }
#pragma unroll
    for (int s = 0; s < 4; s++) { // + f_tail[s] z^s H
        uint32_t V[GU][4] = {};
#pragma unroll
        for (int u = 0; u < GU; u++) {
            const uint32_t x[4] = {U[u][0], U[u][1], U[u][2], U[u][3]};
            choi_mac(V[u], d_f_tail[s], x);
        }
#pragma unroll
        for (int g = 0; g < CH_G; g++)
#pragma unroll
            for (int d = 0; d < 4; d++) {
                const uint32_t vh = (g < GU) ? V[g < GU ? g : 0][d] : 0u;
                const uint32_t vl = (g >= 1 && g - 1 < GU) ? V[(g >= 1 && g - 1 < GU) ? g - 1 : 0][d] : 0u;
                R[g][d] ^= __funnelshift_l(vl, vh, s);
            }
    }
    const unsigned char *t_b = t + (size_t)sig * m;
    unsigned char *y_b = y + (size_t)sig * m;
#pragma unroll
    for (int p = 0; p < m; p++) {
        const int g = p >> 5, q = p & 31;
        const uint32_t nib = ((R[g][0] >> q) & 1u) | (((R[g][1] >> q) & 1u) << 1) |
                             (((R[g][2] >> q) & 1u) << 2) | (((R[g][3] >> q) & 1u) << 3);
        y_b[p] = t_b[p] ^ (unsigned char)nib;
    }
}

static void choi_lab_init(void)
{
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const unsigned char pos[] = MAYO_P_whip_init;
    unsigned char lab[MAYO_P_k * MAYO_P_k];
    for (int l = 0; l < CH_NP; l++) {
        lab[pos[2 * l] * MAYO_P_k + pos[2 * l + 1]] = (unsigned char)l;
        lab[pos[2 * l + 1] * MAYO_P_k + pos[2 * l]] = (unsigned char)l;
    }
    cudaMemcpyToSymbol(choi_lab, lab, sizeof(lab));
}

// Verify: H = S (P S^T) over n, key-side P (Sec. 5.3)
#define choi_verify_SPS choi_fused_SPS<MAYO_P_n, TC_RT, TC_KT, TC_CT, TC_COLS>

#endif
