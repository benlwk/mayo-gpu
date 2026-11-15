#ifndef GF16_BITSLICE_H
#define GF16_BITSLICE_H

#include <stdint.h>

#if defined(__CUDACC__)
#define GF16_BS_INLINE __device__ __forceinline__
#else
#define GF16_BS_INLINE static inline
#endif

GF16_BS_INLINE uint8_t gf16_bs_mul4(uint8_t a, uint8_t b)
{
    uint8_t r = 0;
    a &= 0x0F;
    b &= 0x0F;

    while (b) {
        if (b & 1) {
            r ^= a;
        }

        b >>= 1;

        uint8_t carry = a & 0x08;
        a <<= 1;

        if (carry) {
            a ^= 0x03;
        }

        a &= 0x0F;
    }

    return r & 0x0F;
}

// Branch-free GF(16) multiply table for a scalar b
GF16_BS_INLINE uint32_t gf16_bs_mul4_tab(uint8_t b)
{
    uint32_t x = ((uint32_t)b) * 0x08040201u;
    uint32_t high_nibble_mask = 0xf0f0f0f0u;
    uint32_t high_half = x & high_nibble_mask;
    return x ^ (high_half >> 4) ^ (high_half >> 3);
}

// Delta (butterfly) transpose primitives.

// at positions 4i.

GF16_BS_INLINE uint64_t gf16_bs_compress4(uint64_t p)
{
    p = (p & 0x0101010101010101ULL) | ((p & 0x1010101010101010ULL) >> 2);

    p = (p & 0x0005000500050005ULL) | ((p & 0x0F000F000F000F00ULL) >> 4);

    p = (p & 0x0000005500000055ULL) | ((p & 0x00FF000000FF0000ULL) >> 8);

    p = (p & 0x000000000000FFFFULL) | ((p & 0xFFFFFFFF00000000ULL) >> 16);

    p = (p | (p >> 1)) & 0x3333333333333333ULL;
    p = (p | (p >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
    p = (p | (p >> 4)) & 0x00FF00FF00FF00FFULL;
    p = (p | (p >> 8)) & 0x0000FFFF0000FFFFULL;
    p = (p | (p >> 16)) & 0x00000000FFFFFFFFULL;

    return p;
}

GF16_BS_INLINE uint64_t gf16_bs_spread4(uint64_t p)
{
    p &= 0x00000000FFFFFFFFULL;
    p = (p | (p << 16)) & 0x0000FFFF0000FFFFULL;
    p = (p | (p << 8)) & 0x00FF00FF00FF00FFULL;
    p = (p | (p << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    p = (p | (p << 2)) & 0x3333333333333333ULL;
    p = (p | (p << 1)) & 0x5555555555555555ULL;

    p = (p & 0xFFFFFFFF0000FFFFULL) | ((p & 0x00000000FFFF0000ULL) << 16);

    p = (p & 0xFFFF00FFFFFF00FFULL) | ((p & 0x0000FF000000FF00ULL) << 8);

    p = (p & 0xFF0FFF0FFF0FFF0FULL) | ((p & 0x00F000F000F000F0ULL) << 4);

    p = (p & 0xFBFBFBFBFBFBFBFBULL) | ((p & 0x0404040404040404ULL) << 2);

    return p;
}

GF16_BS_INLINE void gf16_bitslice_pack_to_planes(const uint64_t pack[4], uint64_t planes[4])
{
    planes[0] = 0;
    planes[1] = 0;
    planes[2] = 0;
    planes[3] = 0;

    for (int l = 0; l < 4; l++) {
        uint64_t x = pack[l];
        uint64_t q0 = gf16_bs_compress4(x & 0x1111111111111111ULL);
        uint64_t q1 = gf16_bs_compress4((x >> 1) & 0x1111111111111111ULL);
        uint64_t q2 = gf16_bs_compress4((x >> 2) & 0x1111111111111111ULL);
        uint64_t q3 = gf16_bs_compress4((x >> 3) & 0x1111111111111111ULL);

        planes[0] |= q0 << (16 * l);
        planes[1] |= q1 << (16 * l);
        planes[2] |= q2 << (16 * l);
        planes[3] |= q3 << (16 * l);
    }
}

GF16_BS_INLINE void gf16_bitslice_planes_to_pack(const uint64_t planes[4], uint64_t pack[4])
{
    for (int l = 0; l < 4; l++) {
        uint64_t q0 = gf16_bs_spread4((planes[0] >> (16 * l)) & 0xFFFF);
        uint64_t q1 = gf16_bs_spread4((planes[1] >> (16 * l)) & 0xFFFF);
        uint64_t q2 = gf16_bs_spread4((planes[2] >> (16 * l)) & 0xFFFF);
        uint64_t q3 = gf16_bs_spread4((planes[3] >> (16 * l)) & 0xFFFF);

        pack[l] = q0 | (q1 << 1) | (q2 << 2) | (q3 << 3);
    }
}

// This is the analogue of gf16_vec_mul but on bit-plane words.

GF16_BS_INLINE void gf16_bitslice_mul_const(const uint64_t in[4], uint8_t a, uint64_t out[4])
{
    // 4x4 GF(2) bit-matrix of the constant: m[k] = a * (1 << k).
    uint32_t tab = gf16_bs_mul4_tab(a);
    const uint8_t m0 = (uint8_t)(tab & 0xF);         // a * 1
    const uint8_t m1 = (uint8_t)((tab >> 8) & 0xF);  // a * 2
    const uint8_t m2 = (uint8_t)((tab >> 16) & 0xF); // a * 4
    const uint8_t m3 = (uint8_t)((tab >> 24) & 0xF); // a * 8

    uint64_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;

    uint64_t s = in[0];
    o0 ^= s & (uint64_t)(-(int64_t)((m0 >> 0) & 1));
    o1 ^= s & (uint64_t)(-(int64_t)((m0 >> 1) & 1));
    o2 ^= s & (uint64_t)(-(int64_t)((m0 >> 2) & 1));
    o3 ^= s & (uint64_t)(-(int64_t)((m0 >> 3) & 1));

    s = in[1];
    o0 ^= s & (uint64_t)(-(int64_t)((m1 >> 0) & 1));
    o1 ^= s & (uint64_t)(-(int64_t)((m1 >> 1) & 1));
    o2 ^= s & (uint64_t)(-(int64_t)((m1 >> 2) & 1));
    o3 ^= s & (uint64_t)(-(int64_t)((m1 >> 3) & 1));

    s = in[2];
    o0 ^= s & (uint64_t)(-(int64_t)((m2 >> 0) & 1));
    o1 ^= s & (uint64_t)(-(int64_t)((m2 >> 1) & 1));
    o2 ^= s & (uint64_t)(-(int64_t)((m2 >> 2) & 1));
    o3 ^= s & (uint64_t)(-(int64_t)((m2 >> 3) & 1));

    s = in[3];
    o0 ^= s & (uint64_t)(-(int64_t)((m3 >> 0) & 1));
    o1 ^= s & (uint64_t)(-(int64_t)((m3 >> 1) & 1));
    o2 ^= s & (uint64_t)(-(int64_t)((m3 >> 2) & 1));
    o3 ^= s & (uint64_t)(-(int64_t)((m3 >> 3) & 1));

    out[0] = o0;
    out[1] = o1;
    out[2] = o2;
    out[3] = o3;
}

GF16_BS_INLINE void gf16_bitslice_mul_add_const(uint64_t acc[4], const uint64_t in[4], uint8_t a)
{
    uint64_t m[4];
    gf16_bitslice_mul_const(in, a, m);
    acc[0] ^= m[0];
    acc[1] ^= m[1];
    acc[2] ^= m[2];
    acc[3] ^= m[3];
}

GF16_BS_INLINE void gf16_bitslice_vec_mul_packed(const uint64_t pack_in[4], uint8_t a, uint64_t pack_out[4])
{
    uint64_t planes[4];
    uint64_t res[4];

    gf16_bitslice_pack_to_planes(pack_in, planes);
    gf16_bitslice_mul_const(planes, a, res);
    gf16_bitslice_planes_to_pack(res, pack_out);
}

#undef GF16_BS_INLINE

#endif // GF16_BITSLICE_H
