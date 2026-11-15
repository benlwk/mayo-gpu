#ifndef R3_FINE_CUH
#define R3_FINE_CUH
#include "r3_aes_ct.cuh"

#define MAYO_LAMBDA_SEED_BYTES 32
#define R3_TENC_STRIDE         (MAYO_P_m_bytes + MAYO_LAMBDA_SEED_BYTES)
#define R3_LAMBDA_BYTES        ((MAYO_P_n * MAYO_P_m) / 2)
#define R3_LAMBDA_BLOCKS       ((R3_LAMBDA_BYTES + 15) / 16)
#define R3_LAMBDA_STRIDE       (R3_LAMBDA_BLOCKS * 16)
#define R3_LAMBDA_CHUNK        8
#define R3_LAMBDA_CHUNKS       ((R3_LAMBDA_BLOCKS + R3_LAMBDA_CHUNK - 1) / R3_LAMBDA_CHUNK)

__global__ void r3_lambda_gen(const unsigned char *tenc, unsigned char *lam, int nsig)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * R3_LAMBDA_CHUNKS) {
        return;
    }
    const int sig = tid / R3_LAMBDA_CHUNKS, ch = tid % R3_LAMBDA_CHUNKS;
    const unsigned char *seed = tenc + (size_t)sig * R3_TENC_STRIDE + MAYO_P_m_bytes;
    uint64_t comp[22];
    r3_aesct_keysched(comp, seed);
    unsigned char *out = lam + (size_t)sig * R3_LAMBDA_STRIDE;
    const int b_end = min((ch + 1) * R3_LAMBDA_CHUNK, R3_LAMBDA_BLOCKS);
    for (int b = ch * R3_LAMBDA_CHUNK; b < b_end; b += 4) {
        unsigned char o[64];
        r3_aesct_lambda4(comp, seed + 16, (uint32_t)b, o);
        const int nb = min(4, b_end - b);
        for (int i = 0; i < 16 * nb; i++) {
            out[b * 16 + i] = o[i];
        }
    }
}

static __device__ __forceinline__ void r3_lambda_row(const unsigned char *lam_sig, int r, uint64_t *row)
{
    unsigned char *rb = (unsigned char *)row;
    for (int l = 0; l < MAYO_P_m_vec_limbs; l++) {
        row[l] = 0;
    }
    for (int b = 0; b < MAYO_P_m / 2; b++) {
        rb[b] = lam_sig[r * (MAYO_P_m / 2) + b];
    }
}

__global__ void r3_lambda_fold_sign(const unsigned char *lam, const unsigned char *Vdec,
                                    const unsigned char *O, uint64_t *VL_planes, uint64_t *VPV, int nsig)
{
    const int per = MAYO_P_k + MAYO_P_o;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * per) {
        return;
    }
    const int sig = tid / per, t = tid % per;
    const int v = MAYO_P_v, o = MAYO_P_o, k = MAYO_P_k, mvl = MAYO_P_m_vec_limbs;
    const unsigned char *lam_s = lam + (size_t)sig * R3_LAMBDA_STRIDE;
    uint64_t acc[M_VEC_LIMBS_MAX] = {0}, row[M_VEC_LIMBS_MAX];
    if (t < k) {
        const unsigned char *vi = Vdec + (size_t)sig * k * v + (size_t)t * v;
        for (int r = 0; r < v; r++) {
            r3_lambda_row(lam_s, r, row);
            const uint8_t s = vi[r];
            for (int l = 0; l < mvl; l++) {
                acc[l] ^= gf16_vec_mul(row[l], s);
            }
        }
        uint64_t *dst = VPV + ((size_t)sig * k * k + (size_t)t * k) * mvl;
        for (int l = 0; l < mvl; l++) {
            dst[l] ^= acc[l];
        }
    } else {
        const int c = t - k;
        for (int r = 0; r < v; r++) {
            r3_lambda_row(lam_s, r, row);
            const uint8_t s = O[r * o + c];
            for (int l = 0; l < mvl; l++) {
                acc[l] ^= gf16_vec_mul(row[l], s);
            }
        }
        r3_lambda_row(lam_s, v + c, row);
        for (int l = 0; l < mvl; l++) {
            acc[l] ^= row[l];
        }
        uint64_t planes[M_VEC_LIMBS_MAX];
        hyb_pack_to_planes(acc, planes);
        uint64_t *dst = VL_planes + ((size_t)sig * k * o + (size_t)c) * mvl;
        for (int l = 0; l < mvl; l++) {
            dst[l] ^= planes[l];
        }
    }
}

__global__ void r3_lambda_fold_verify(const unsigned char *lam, const unsigned char *s, uint64_t *SPS,
                                      int nsig)
{
    const int k = MAYO_P_k, n = MAYO_P_n, mvl = MAYO_P_m_vec_limbs;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * k) {
        return;
    }
    const int sig = tid / k, i = tid % k;
    const unsigned char *lam_s = lam + (size_t)sig * R3_LAMBDA_STRIDE;
    const unsigned char *si = s + (size_t)sig * k * n + (size_t)i * n;
    uint64_t acc[M_VEC_LIMBS_MAX] = {0}, row[M_VEC_LIMBS_MAX];
    for (int r = 0; r < n; r++) {
        r3_lambda_row(lam_s, r, row);
        const uint8_t sv = si[r];
        for (int l = 0; l < mvl; l++) {
            acc[l] ^= gf16_vec_mul(row[l], sv);
        }
    }
    uint64_t *dst = SPS + ((size_t)sig * k * k + (size_t)i * k) * mvl;
    for (int l = 0; l < mvl; l++) {
        dst[l] ^= acc[l];
    }
}
#endif
