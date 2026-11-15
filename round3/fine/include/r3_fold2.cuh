// Staged lambda folds
#ifndef R3_FOLD2_CUH
#define R3_FOLD2_CUH

#define FOLD2_TPS_SIGN  ((MAYO_P_k + MAYO_P_o) * MAYO_P_m_vec_limbs) // threads per signature
#define FOLD2_TPS_VER   (MAYO_P_k * MAYO_P_m_vec_limbs)
#define FOLD2_SIGS_SIGN (FOLD2_TPS_SIGN >= 256 ? 1 : 256 / FOLD2_TPS_SIGN)
#define FOLD2_SIGS_VER  (FOLD2_TPS_VER >= 256 ? 1 : 256 / FOLD2_TPS_VER)
#define FOLD2_SIGS      (FOLD2_SIGS_SIGN > FOLD2_SIGS_VER ? FOLD2_SIGS_SIGN : FOLD2_SIGS_VER) // staging capacity

__device__ __forceinline__ void fold2_stage(const unsigned char *lam, uint32_t *sh, int sig0, int nsig,
                                            int nmax)
{
    const int words = R3_LAMBDA_STRIDE / 4; // the stride is a multiple of 16
    const int nl = (nsig - sig0 < nmax) ? nsig - sig0 : nmax;
    const uint4 *src = (const uint4 *)(lam + (size_t)sig0 * R3_LAMBDA_STRIDE);
    uint4 *dst = (uint4 *)sh;
    for (int i = threadIdx.x; i < nl * words / 4; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void fold2_row(const uint32_t *lam_s, int r, uint64_t row[M_VEC_LIMBS_MAX])
{
#pragma unroll
    for (int l = 0; l < MAYO_P_m_vec_limbs; l++) {
        const int byte = r * (MAYO_P_m / 2) + 8 * l, w = byte >> 2, sh = (byte & 3) * 8;
        const uint32_t a = lam_s[w], b = lam_s[w + 1], c = lam_s[w + 2];
        const uint32_t lo = __funnelshift_r(a, b, sh), hi = __funnelshift_r(b, c, sh);
        uint64_t v = ((uint64_t)hi << 32) | lo;
        const int left = MAYO_P_m - 16 * l; // nibbles of this limb inside the row
        if (left < 16) {
            v &= (1ULL << (4 * left)) - 1;
        }
        row[l] = v;
    }
}

// limb l of lambda row r (one 64-bit limb of fold2_row)
__device__ __forceinline__ uint64_t fold2_limb(const uint32_t *lam_s, int r, int l)
{
    const int byte = r * (MAYO_P_m / 2) + 8 * l, w = byte >> 2, sh = (byte & 3) * 8;
    const uint32_t a = lam_s[w], b = lam_s[w + 1], c = lam_s[w + 2];
    uint64_t v = ((uint64_t)__funnelshift_r(b, c, sh) << 32) | __funnelshift_r(a, b, sh);
    const int left = MAYO_P_m - 16 * l; // nibbles of this limb inside the row
    if (left < 16) {
        v &= (1ULL << (4 * left)) - 1;
    }
    return v;
}

__global__ void __launch_bounds__(256)
    r3_lambda_fold_sign2(const unsigned char *lam, const unsigned char *Vdec, const unsigned char *O,
                         uint64_t *VL_planes, uint64_t *VPV, int nsig)
{
    __shared__ uint32_t sh[FOLD2_SIGS_SIGN * (R3_LAMBDA_STRIDE / 4) + 4];
    const int v = MAYO_P_v, o = MAYO_P_o, k = MAYO_P_k, ml = MAYO_P_m_vec_limbs;
    const int sig0 = blockIdx.x * FOLD2_SIGS_SIGN;
    fold2_stage(lam, sh, sig0, nsig, FOLD2_SIGS_SIGN);
    if (threadIdx.x < 4) {
        sh[FOLD2_SIGS_SIGN * (R3_LAMBDA_STRIDE / 4) + threadIdx.x] = 0; // read past the end
    }
    __syncthreads();
    const int ls = threadIdx.x / FOLD2_TPS_SIGN, rem = threadIdx.x % FOLD2_TPS_SIGN;
    const int t = rem / ml, l = rem % ml, sig = sig0 + ls;
    if (ls >= FOLD2_SIGS_SIGN || sig >= nsig) {
        return;
    }
    const uint32_t *lam_s = sh + ls * (R3_LAMBDA_STRIDE / 4);
    uint64_t acc = 0;
    if (t < k) {
        const unsigned char *vi = Vdec + (size_t)sig * k * v + (size_t)t * v;
#pragma unroll 4
        for (int r = 0; r < v; r++) {
            acc ^= gf16_vec_mul(fold2_limb(lam_s, r, l), vi[r]);
        }
        VPV[((size_t)sig * k * k + (size_t)t * k) * ml + l] ^= acc;
    } else {
        const int c = t - k;
#pragma unroll 4
        for (int r = 0; r < v; r++) {
            acc ^= gf16_vec_mul(fold2_limb(lam_s, r, l), O[r * o + c]);
        }
        acc ^= fold2_limb(lam_s, v + c, l);
        uint64_t *dst = VL_planes + ((size_t)sig * k * o + (size_t)c) * ml;
        if (l < 4 * MAYO_P_bs_groups) {
            // bit-plane group l / 4
            const int g = l >> 2, sh16 = 16 * (l & 3);
#pragma unroll
            for (int d = 0; d < 4; d++) {
                atomicXor(
                    (unsigned long long *)&dst[4 * g + d],
                    (unsigned long long)(gf16_bs_compress4((acc >> d) & 0x1111111111111111ULL) << sh16));
            }
        } else {
            dst[l] ^= acc; // nibble tail word
        }
    }
}

__global__ void __launch_bounds__(256)
    r3_lambda_fold_verify2(const unsigned char *lam, const unsigned char *s, uint64_t *SPS, int nsig)
{
    __shared__ uint32_t sh[FOLD2_SIGS_VER * (R3_LAMBDA_STRIDE / 4) + 4];
    const int k = MAYO_P_k, n = MAYO_P_n, ml = MAYO_P_m_vec_limbs;
    const int sig0 = blockIdx.x * FOLD2_SIGS_VER;
    fold2_stage(lam, sh, sig0, nsig, FOLD2_SIGS_VER);
    if (threadIdx.x < 4) {
        sh[FOLD2_SIGS_VER * (R3_LAMBDA_STRIDE / 4) + threadIdx.x] = 0;
    }
    __syncthreads();
    const int ls = threadIdx.x / FOLD2_TPS_VER, rem = threadIdx.x % FOLD2_TPS_VER;
    const int i = rem / ml, l = rem % ml, sig = sig0 + ls;
    if (ls >= FOLD2_SIGS_VER || sig >= nsig) {
        return;
    }
    const uint32_t *lam_s = sh + ls * (R3_LAMBDA_STRIDE / 4);
    const unsigned char *si = s + (size_t)sig * k * n + (size_t)i * n;
    uint64_t acc = 0;
#pragma unroll 4
    for (int r = 0; r < n; r++) {
        acc ^= gf16_vec_mul(fold2_limb(lam_s, r, l), si[r]);
    }
    SPS[((size_t)sig * k * k + (size_t)i * k) * ml + l] ^= acc;
}

#endif
