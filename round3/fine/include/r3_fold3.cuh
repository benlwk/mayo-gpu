// Bit-sliced lambda folds
#ifndef R3_FOLD3_CUH
#define R3_FOLD3_CUH

#define F3_G         ((MAYO_P_m + 31) / 32)
#define F3_TPS_SIGN  ((MAYO_P_k + MAYO_P_o) * F3_G)
#define F3_TPS_VER   (MAYO_P_k * F3_G)
#define F3_SMEM_SIGS ((40 * 1024) / (MAYO_P_n * F3_G * 16)) // signatures whose planes fit in 40 KB
#define F3_MIN(a, b) ((a) < (b) ? (a) : (b))
#define F3_SIGS_SIGN F3_MIN(F3_TPS_SIGN >= 256 ? 1 : 256 / F3_TPS_SIGN, F3_SMEM_SIGS)
#define F3_SIGS_VER  F3_MIN(F3_TPS_VER >= 256 ? 1 : 256 / F3_TPS_VER, F3_SMEM_SIGS)

__device__ __forceinline__ void f3_stage(const unsigned char *__restrict__ lam, uint4 *sh, int sig0, int nsig,
                                         int nsigs_blk, int nrows)
{
    const int nl = (nsig - sig0 < nsigs_blk) ? nsig - sig0 : nsigs_blk;
    for (int it = threadIdx.x; it < nl * nrows * F3_G; it += blockDim.x) {
        const int g = it % F3_G, row = (it / F3_G) % nrows, ls = it / (F3_G * nrows);
        const unsigned char *src =
            lam + (size_t)(sig0 + ls) * R3_LAMBDA_STRIDE + (size_t)row * (MAYO_P_m / 2);
        uint32_t p[4] = {0, 0, 0, 0};
#pragma unroll
        for (int q = 0; q < 16; q++) {
            const int byte = 16 * g + q;
            const uint32_t b = (byte < MAYO_P_m / 2) ? src[byte] : 0u;
#pragma unroll
            for (int d = 0; d < 4; d++) {
                p[d] |= (((b >> d) & 1u) << (2 * q)) | (((b >> (4 + d)) & 1u) << (2 * q + 1));
            }
        }
        sh[(ls * nrows + row) * F3_G + g] = make_uint4(p[0], p[1], p[2], p[3]);
    }
}

__device__ __forceinline__ void f3_mac(uint32_t acc[4], uint32_t s, uint4 x)
{
    uint32_t y0 = x.x, y1 = x.y, y2 = x.z, y3 = x.w;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const uint32_t m = 0u - ((s >> t) & 1u);
        acc[0] ^= y0 & m;
        acc[1] ^= y1 & m;
        acc[2] ^= y2 & m;
        acc[3] ^= y3 & m;
        const uint32_t t3 = y3;
        y3 = y2;
        y2 = y1;
        y1 = y0 ^ t3;
        y0 = t3; // y *= x
    }
}

__device__ __forceinline__ uint64_t f3_limb(const uint32_t acc[4], int h)
{
    uint64_t v = 0;
#pragma unroll
    for (int d = 0; d < 4; d++) {
        v |= gf16_bs_spread4((acc[d] >> (16 * h)) & 0xFFFFu) << d;
    }
    return v;
}

// yext != NULL
__global__ void __launch_bounds__(256)
    r3_lambda_fold_sign3(const unsigned char *lam, const unsigned char *Vdec, const unsigned char *O,
                         uint64_t *VL_planes, uint64_t *VPV, int nsig, uint32_t *yext, int vl_part)
{
    constexpr int v = MAYO_P_v, o = MAYO_P_o, k = MAYO_P_k, ml = MAYO_P_m_vec_limbs, n = MAYO_P_n;
    __shared__ uint4 sh[F3_SIGS_SIGN * n * F3_G];
    const int sig0 = blockIdx.x * F3_SIGS_SIGN;
    f3_stage(lam, sh, sig0, nsig, F3_SIGS_SIGN, n);
    __syncthreads();
    const int ls = threadIdx.x / F3_TPS_SIGN, rem = threadIdx.x % F3_TPS_SIGN;
    const int t = rem / F3_G, g = rem % F3_G, sig = sig0 + ls;
    if (ls >= F3_SIGS_SIGN || sig >= nsig) {
        return;
    }
    const uint4 *L = sh + (size_t)ls * n * F3_G + g;
    uint32_t acc[4] = {0, 0, 0, 0};
    if (t < k) {
        const unsigned char *vi = Vdec + (size_t)sig * k * v + (size_t)t * v;
#pragma unroll 4
        for (int r = 0; r < v; r++) {
            f3_mac(acc, vi[r], L[r * F3_G]);
        }
        if (yext != NULL) {
            const int lab = choi_lab[t * k], w0 = g + (lab >> 5), bs = lab & 31;
            uint32_t *dy = yext + (size_t)sig * (CH_GE * 4);
#pragma unroll
            for (int d = 0; d < 4; d++) {
                atomicXor(dy + w0 * 4 + d, acc[d] << bs);
                if (bs != 0) {
                    atomicXor(dy + (w0 + 1) * 4 + d, acc[d] >> (32 - bs));
                }
            }
            return;
        }
        // nibble-packed VPV[t][0]: limbs 2g and 2g + 1
        uint64_t *dst = VPV + ((size_t)sig * k * k + (size_t)t * k) * ml;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (2 * g + h < ml) {
                dst[2 * g + h] ^= f3_limb(acc, h);
            }
        }
    } else {
        if (!vl_part) {
            return;
        }
        const int c = t - k;
#pragma unroll 4
        for (int r = 0; r < v; r++) {
            f3_mac(acc, O[r * o + c], L[r * F3_G]);
        }
        const uint4 lo = L[(v + c) * F3_G];
        acc[0] ^= lo.x;
        acc[1] ^= lo.y;
        acc[2] ^= lo.z;
        acc[3] ^= lo.w;
        uint64_t *dst = VL_planes + ((size_t)sig * k * o + (size_t)c) * ml;
        if (g < 2 * MAYO_P_bs_groups) {
            uint32_t *d32 = (uint32_t *)dst;
#pragma unroll
            for (int d = 0; d < 4; d++) {
                d32[2 * (4 * (g >> 1) + d) + (g & 1)] ^= acc[d];
            }
        } else {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int tw = 2 * (g - 2 * MAYO_P_bs_groups) + h;
                if (tw < MAYO_P_bs_tail) {
                    dst[4 * MAYO_P_bs_groups + tw] ^= f3_limb(acc, h);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256)
    r3_lambda_fold_verify3(const unsigned char *lam, const unsigned char *s, uint64_t *SPS, int nsig)
{
    constexpr int k = MAYO_P_k, n = MAYO_P_n, ml = MAYO_P_m_vec_limbs;
    __shared__ uint4 sh[F3_SIGS_VER * n * F3_G];
    const int sig0 = blockIdx.x * F3_SIGS_VER;
    f3_stage(lam, sh, sig0, nsig, F3_SIGS_VER, n);
    __syncthreads();
    const int ls = threadIdx.x / F3_TPS_VER, rem = threadIdx.x % F3_TPS_VER;
    const int i = rem / F3_G, g = rem % F3_G, sig = sig0 + ls;
    if (ls >= F3_SIGS_VER || sig >= nsig) {
        return;
    }
    const uint4 *L = sh + (size_t)ls * n * F3_G + g;
    const unsigned char *si = s + (size_t)sig * k * n + (size_t)i * n;
    uint32_t acc[4] = {0, 0, 0, 0};
#pragma unroll 4
    for (int r = 0; r < n; r++) {
        f3_mac(acc, si[r], L[r * F3_G]);
    }
    uint64_t *dst = SPS + ((size_t)sig * k * k + (size_t)i * k) * ml;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (2 * g + h < ml) {
            dst[2 * g + h] ^= f3_limb(acc, h);
        }
    }
}

#endif
