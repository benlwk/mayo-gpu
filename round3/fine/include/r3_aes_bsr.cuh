// Barrel-shiftrows bit-sliced AES-128 for lambda, with a table-free key schedule
#ifndef R3_AES_BSR_CUH
#define R3_AES_BSR_CUH

#define BSR_CHUNK  8 // blocks per thread
#define BSR_CHUNKS ((R3_LAMBDA_BLOCKS + BSR_CHUNK - 1) / BSR_CHUNK)

static __device__ __forceinline__ void bsr_sbox(uint32_t *q)
{
    uint32_t x0, x1, x2, x3, x4, x5, x6, x7;
    uint32_t y1, y2, y3, y4, y5, y6, y7, y8, y9;
    uint32_t y10, y11, y12, y13, y14, y15, y16, y17, y18, y19;
    uint32_t y20, y21;
    uint32_t z0, z1, z2, z3, z4, z5, z6, z7, z8, z9;
    uint32_t z10, z11, z12, z13, z14, z15, z16, z17;
    uint32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    uint32_t t10, t11, t12, t13, t14, t15, t16, t17, t18, t19;
    uint32_t t20, t21, t22, t23, t24, t25, t26, t27, t28, t29;
    uint32_t t30, t31, t32, t33, t34, t35, t36, t37, t38, t39;
    uint32_t t40, t41, t42, t43, t44, t45, t46, t47, t48, t49;
    uint32_t t50, t51, t52, t53, t54, t55, t56, t57, t58, t59;
    uint32_t t60, t61, t62, t63, t64, t65, t66, t67;
    uint32_t s0, s1, s2, s3, s4, s5, s6, s7;

    x0 = q[7];
    x1 = q[6];
    x2 = q[5];
    x3 = q[4];
    x4 = q[3];
    x5 = q[2];
    x6 = q[1];
    x7 = q[0];

    // Top linear transformation
    y14 = x3 ^ x5;
    y13 = x0 ^ x6;
    y9 = x0 ^ x3;
    y8 = x0 ^ x5;
    t0 = x1 ^ x2;
    y1 = t0 ^ x7;
    y4 = y1 ^ x3;
    y12 = y13 ^ y14;
    y2 = y1 ^ x0;
    y5 = y1 ^ x6;
    y3 = y5 ^ y8;
    t1 = x4 ^ y12;
    y15 = t1 ^ x5;
    y20 = t1 ^ x1;
    y6 = y15 ^ x7;
    y10 = y15 ^ t0;
    y11 = y20 ^ y9;
    y7 = x7 ^ y11;
    y17 = y10 ^ y11;
    y19 = y10 ^ y8;
    y16 = t0 ^ y11;
    y21 = y13 ^ y16;
    y18 = x0 ^ y16;

    // Non-linear section
    t2 = y12 & y15;
    t3 = y3 & y6;
    t4 = t3 ^ t2;
    t5 = y4 & x7;
    t6 = t5 ^ t2;
    t7 = y13 & y16;
    t8 = y5 & y1;
    t9 = t8 ^ t7;
    t10 = y2 & y7;
    t11 = t10 ^ t7;
    t12 = y9 & y11;
    t13 = y14 & y17;
    t14 = t13 ^ t12;
    t15 = y8 & y10;
    t16 = t15 ^ t12;
    t17 = t4 ^ t14;
    t18 = t6 ^ t16;
    t19 = t9 ^ t14;
    t20 = t11 ^ t16;
    t21 = t17 ^ y20;
    t22 = t18 ^ y19;
    t23 = t19 ^ y21;
    t24 = t20 ^ y18;

    t25 = t21 ^ t22;
    t26 = t21 & t23;
    t27 = t24 ^ t26;
    t28 = t25 & t27;
    t29 = t28 ^ t22;
    t30 = t23 ^ t24;
    t31 = t22 ^ t26;
    t32 = t31 & t30;
    t33 = t32 ^ t24;
    t34 = t23 ^ t33;
    t35 = t27 ^ t33;
    t36 = t24 & t35;
    t37 = t36 ^ t34;
    t38 = t27 ^ t36;
    t39 = t29 & t38;
    t40 = t25 ^ t39;

    t41 = t40 ^ t37;
    t42 = t29 ^ t33;
    t43 = t29 ^ t40;
    t44 = t33 ^ t37;
    t45 = t42 ^ t41;
    z0 = t44 & y15;
    z1 = t37 & y6;
    z2 = t33 & x7;
    z3 = t43 & y16;
    z4 = t40 & y1;
    z5 = t29 & y7;
    z6 = t42 & y11;
    z7 = t45 & y17;
    z8 = t41 & y10;
    z9 = t44 & y12;
    z10 = t37 & y3;
    z11 = t33 & y4;
    z12 = t43 & y13;
    z13 = t40 & y5;
    z14 = t29 & y2;
    z15 = t42 & y9;
    z16 = t45 & y14;
    z17 = t41 & y8;

    // Bottom linear transformation
    t46 = z15 ^ z16;
    t47 = z10 ^ z11;
    t48 = z5 ^ z13;
    t49 = z9 ^ z10;
    t50 = z2 ^ z12;
    t51 = z2 ^ z5;
    t52 = z7 ^ z8;
    t53 = z0 ^ z3;
    t54 = z6 ^ z7;
    t55 = z16 ^ z17;
    t56 = z12 ^ t48;
    t57 = t50 ^ t53;
    t58 = z4 ^ t46;
    t59 = z3 ^ t54;
    t60 = t46 ^ t57;
    t61 = z14 ^ t57;
    t62 = t52 ^ t58;
    t63 = t49 ^ t58;
    t64 = z4 ^ t59;
    t65 = t61 ^ t62;
    t66 = z1 ^ t63;
    s0 = t59 ^ t63;
    s6 = t56 ^ ~t62;
    s7 = t48 ^ ~t60;
    t67 = t64 ^ t65;
    s3 = t53 ^ t66;
    s4 = t51 ^ t66;
    s5 = t47 ^ t65;
    s1 = t64 ^ ~s3;
    s2 = t55 ^ ~t67;

    q[7] = s0;
    q[6] = s1;
    q[5] = s2;
    q[4] = s3;
    q[3] = s4;
    q[2] = s5;
    q[1] = s6;
    q[0] = s7;
}

__device__ __forceinline__ void bsr_rows(const uint32_t col[4], uint32_t row[4])
{
    const uint32_t a = __byte_perm(col[0], col[1], 0x5140), b = __byte_perm(col[2], col[3], 0x5140);
    const uint32_t c = __byte_perm(col[0], col[1], 0x7362), d = __byte_perm(col[2], col[3], 0x7362);
    row[0] = __byte_perm(a, b, 0x5410);
    row[1] = __byte_perm(a, b, 0x7632);
    row[2] = __byte_perm(c, d, 0x5410);
    row[3] = __byte_perm(c, d, 0x7632);
}

__device__ __forceinline__ void bsr_replicate(const uint32_t row[4], uint32_t w[32])
{
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int b = 0; b < 8; b++) {
            w[8 * r + 7 - b] = ((row[r] >> b) & 0x01010101u) * 0xffu;
        }
}

__device__ __forceinline__ void bsr_mixcolumns(uint32_t w[32])
{
    uint32_t a[4][8], T[8];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int b = 0; b < 8; b++) {
            a[r][b] = w[8 * r + 7 - b];
        }
#pragma unroll
    for (int b = 0; b < 8; b++) {
        T[b] = a[0][b] ^ a[1][b] ^ a[2][b] ^ a[3][b];
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
        uint32_t u[8];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            u[b] = a[r][b] ^ a[(r + 1) & 3][b];
        }
        const uint32_t x[8] = {u[7], u[0] ^ u[7], u[1], u[2] ^ u[7], u[3] ^ u[7], u[4], u[5], u[6]};
#pragma unroll
        for (int b = 0; b < 8; b++) {
            w[8 * r + 7 - b] = a[r][b] ^ T[b] ^ x[b];
        }
    }
}

__device__ __forceinline__ void bsr_subshift(uint32_t w[32])
{
#pragma unroll
    for (int r = 0; r < 4; r++) {
        uint32_t q[8];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            q[b] = w[8 * r + 7 - b];
        }
        bsr_sbox(q);
#pragma unroll
        for (int b = 0; b < 8; b++) {
            w[8 * r + 7 - b] = r ? __funnelshift_r(q[b], q[b], 8 * r) : q[b]; // lane c <- lane c + r
        }
    }
}

__device__ __forceinline__ void bsr_transpose8(uint32_t W[8])
{
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const uint32_t t = ((W[i] >> 1) ^ W[i + 1]) & 0x55555555u;
        W[i + 1] ^= t;
        W[i] ^= t << 1;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (!(i & 2)) {
            const uint32_t t = ((W[i] >> 2) ^ W[i + 2]) & 0x33333333u;
            W[i + 2] ^= t;
            W[i] ^= t << 2;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t t = ((W[i] >> 4) ^ W[i + 4]) & 0x0f0f0f0fu;
        W[i + 4] ^= t;
        W[i] ^= t << 4;
    }
}

// SubWord without a table
__device__ __forceinline__ uint32_t bsr_subword(uint32_t t)
{
    uint32_t q[8];
#pragma unroll
    for (int b = 0; b < 8; b++) {
        q[b] = (t >> b) & 0x01010101u;
    }
    bsr_sbox(q);
    uint32_t s = 0;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        s |= (q[b] & 0x01010101u) << b;
    }
    return s;
}

__device__ __forceinline__ void bsr_keystream(const unsigned char *seed, int b0, uint32_t w[32])
{
    // round keys as column words
    uint32_t K[11][4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
        K[0][c] = (uint32_t)seed[4 * c] | ((uint32_t)seed[4 * c + 1] << 8) |
                  ((uint32_t)seed[4 * c + 2] << 16) | ((uint32_t)seed[4 * c + 3] << 24);
    }
    uint32_t rcon = 1;
#pragma unroll
    for (int rd = 1; rd <= 10; rd++) {
        const uint32_t t = K[rd - 1][3];
        uint32_t s = bsr_subword(__funnelshift_r(t, t, 8)); // SubWord(RotWord(t))
        s ^= rcon;
        rcon = (rcon << 1) ^ (((rcon >> 7) & 1) * 0x11b);
        K[rd][0] = K[rd - 1][0] ^ s;
        K[rd][1] = K[rd - 1][1] ^ K[rd][0];
        K[rd][2] = K[rd - 1][2] ^ K[rd][1];
        K[rd][3] = K[rd - 1][3] ^ K[rd][2];
    }

    uint32_t ivc[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
        ivc[c] = (uint32_t)seed[16 + 4 * c] | ((uint32_t)seed[17 + 4 * c] << 8) |
                 ((uint32_t)seed[18 + 4 * c] << 16) | ((uint32_t)seed[19 + 4 * c] << 24);
    }
    ivc[3] ^= __byte_perm((uint32_t)b0, 0, 0x0123);
    uint32_t rk[32], row[4];
    bsr_rows(ivc, row);
    bsr_replicate(row, w);

    w[8 * 3 + 7 - 0] ^= 0x55000000u;
    w[8 * 3 + 7 - 1] ^= 0x33000000u;
    w[8 * 3 + 7 - 2] ^= 0x0f000000u;

    bsr_rows(K[0], row);
    bsr_replicate(row, rk);
#pragma unroll
    for (int i = 0; i < 32; i++) {
        w[i] ^= rk[i];
    }
#pragma unroll 1
    for (int rd = 1; rd < 10; rd++) {
        bsr_subshift(w);
        bsr_mixcolumns(w);
        bsr_rows(K[rd], row);
        bsr_replicate(row, rk);
#pragma unroll
        for (int i = 0; i < 32; i++) {
            w[i] ^= rk[i];
        }
    }
    bsr_subshift(w);
    bsr_rows(K[10], row);
    bsr_replicate(row, rk);
#pragma unroll
    for (int i = 0; i < 32; i++) {
        w[i] ^= rk[i];
    }
}

__global__ void __launch_bounds__(256)
    r3_lambda_gen_bsr(const unsigned char *tenc, unsigned char *lam, int nsig)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * BSR_CHUNKS) {
        return;
    }
    const int sig = tid / BSR_CHUNKS, ch = tid % BSR_CHUNKS;
    const int b0 = ch * BSR_CHUNK;
    uint32_t w[32];
    bsr_keystream(tenc + (size_t)sig * R3_TENC_STRIDE + MAYO_P_m_bytes, b0, w);

    uint32_t O[4][8];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        uint32_t W[8];
#pragma unroll
        for (int b = 0; b < 8; b++) {
            W[b] = w[8 * r + 7 - b];
        }
        bsr_transpose8(W);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            O[r][j] = W[7 - j];
        }
    }
    uint4 *out = (uint4 *)(lam + (size_t)sig * R3_LAMBDA_STRIDE);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (b0 + j >= R3_LAMBDA_BLOCKS) {
            break;
        }
        const uint32_t rows[4] = {O[0][j], O[1][j], O[2][j], O[3][j]};
        uint32_t cols[4];
        bsr_rows(rows, cols);
        out[b0 + j] = make_uint4(cols[0], cols[1], cols[2], cols[3]);
    }
}

// The same keystream as GF(16) bit planes
__global__ void __launch_bounds__(256)
    r3_lambda_gen_bsr_planes(const unsigned char *tenc, uint4 *lamP, int nsig)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nsig * BSR_CHUNKS) {
        return;
    }
    const int sig = tid / BSR_CHUNKS, ch = tid % BSR_CHUNKS;
    const int b0 = ch * BSR_CHUNK;
    uint32_t w[32];
    bsr_keystream(tenc + (size_t)sig * R3_TENC_STRIDE + MAYO_P_m_bytes, b0, w);

    uint32_t P[4][8];
#pragma unroll
    for (int d = 0; d < 4; d++) {
        uint32_t W[8];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                W[2 * r + h] = w[8 * r + 7 - d - 4 * h];
            }
        bsr_transpose8(W);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            P[d][j] = W[7 - j];
        }
    }
    uint4 *out = lamP + (size_t)sig * (R3_LAMBDA_STRIDE / 16);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (b0 + j >= R3_LAMBDA_BLOCKS) {
            break;
        }
        out[b0 + j] = make_uint4(P[0][j], P[1][j], P[2][j], P[3][j]);
    }
}
#endif
