// s = v + O x with bit-sliced dot products
#ifndef BUILD_S2_CUH
#define BUILD_S2_CUH

#if MAYO_P_o > 32
#error "build_s2 assumes o <= 32"
#endif

__global__ void build_O_planes(const unsigned char *__restrict__ O, uint4 *__restrict__ Op)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= MAYO_P_v) {
        return;
    }
    uint32_t p[4] = {0, 0, 0, 0};
    for (int j = 0; j < MAYO_P_o; j++) {
        const uint32_t e = O[r * MAYO_P_o + j];
#pragma unroll
        for (int d = 0; d < 4; d++) {
            p[d] |= ((e >> d) & 1u) << j;
        }
    }
    Op[r] = make_uint4(p[0], p[1], p[2], p[3]);
}

__global__ void build_s2(const uint4 *__restrict__ Op, const unsigned char *__restrict__ Vdec,
                         const unsigned char *__restrict__ x, unsigned char *__restrict__ s,
                         const int *__restrict__ map, int nsig)
{
    const int k = MAYO_P_k, n = MAYO_P_n, o = MAYO_P_o, v = MAYO_P_v;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * k * n) {
        return;
    }
    const int batch = tid / (k * n), local = tid % (k * n), i = local / n, pos = local % n;
    const unsigned char *x_i = x + (size_t)batch * (k * n) + i * o;
    unsigned char *s_b = s + (size_t)(map ? map[batch] : batch) * (k * n);
    if (pos >= v) {
        s_b[i * n + pos] = x_i[pos - v] & 0x0F;
        return;
    }

    uint32_t X0 = 0, X1 = 0, X2 = 0, X3 = 0;
#pragma unroll
    for (int j = 0; j < MAYO_P_o; j++) {
        const uint32_t e = x_i[j];
        X0 |= (e & 1u) << j;
        X1 |= ((e >> 1) & 1u) << j;
        X2 |= ((e >> 2) & 1u) << j;
        X3 |= ((e >> 3) & 1u) << j;
    }
    const uint4 R = __ldg(Op + pos);
    const uint32_t O0 = R.x, O1 = R.y, O2 = R.z, O3 = R.w;
    const uint32_t c0 = __popc(O0 & X0) & 1u;
    const uint32_t c1 = __popc((O0 & X1) ^ (O1 & X0)) & 1u;
    const uint32_t c2 = __popc((O0 & X2) ^ (O1 & X1) ^ (O2 & X0)) & 1u;
    const uint32_t c3 = __popc((O0 & X3) ^ (O1 & X2) ^ (O2 & X1) ^ (O3 & X0)) & 1u;
    const uint32_t c4 = __popc((O1 & X3) ^ (O2 & X2) ^ (O3 & X1)) & 1u;
    const uint32_t c5 = __popc((O2 & X3) ^ (O3 & X2)) & 1u;
    const uint32_t c6 = __popc(O3 & X3) & 1u;
    const uint32_t Ox = (c0 ^ c4) | ((c1 ^ c4 ^ c5) << 1) | ((c2 ^ c5 ^ c6) << 2) | ((c3 ^ c6) << 3);
    const unsigned char vi = Vdec[(size_t)batch * (k * v) + i * v + pos] & 0x0F;
    s_b[i * n + pos] = (unsigned char)((vi ^ Ox) & 0x0F);
}

#endif
