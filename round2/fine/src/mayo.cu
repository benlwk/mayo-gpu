#include <stdio.h>
#include <assert.h>

#include "mayo.cuh"
#include "aes_cu.cuh"
#include "gf16_bitslice.cuh"
#include "tc_ps.cuh"

static __device__ __forceinline__ void hyb_mul_add(uint64_t *acc, const uint64_t *in, uint8_t a)
{
#pragma unroll
    for (int g = 0; g < MAYO_P_bs_groups; g++) {
        gf16_bitslice_mul_add_const(acc + 4 * g, in + 4 * g, a);
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
        acc[4 * MAYO_P_bs_groups + tw] ^= gf16_vec_mul(in[4 * MAYO_P_bs_groups + tw], a);
    }
}

static __device__ __forceinline__ void hyb_pack_to_planes(const uint64_t *pack, uint64_t *planes)
{
#pragma unroll
    for (int g = 0; g < MAYO_P_bs_groups; g++) {
        gf16_bitslice_pack_to_planes(pack + 4 * g, planes + 4 * g);
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
        planes[4 * MAYO_P_bs_groups + tw] = pack[4 * MAYO_P_bs_groups + tw];
    }
}

static __device__ __forceinline__ void hyb_planes_to_pack(const uint64_t *planes, uint64_t *pack)
{
#pragma unroll
    for (int g = 0; g < MAYO_P_bs_groups; g++) {
        gf16_bitslice_planes_to_pack(planes + 4 * g, pack + 4 * g);
    }
#pragma unroll
    for (int tw = 0; tw < MAYO_P_bs_tail; tw++) {
        pack[4 * MAYO_P_bs_groups + tw] = planes[4 * MAYO_P_bs_groups + tw];
    }
}

__global__ void unpack_vecs_single(const unsigned char *in, uint64_t *out, int nvecs)
{
    const int in_bytes = MAYO_P_m_bytes;
    const int out_bytes = MAYO_P_m_vec_limbs * 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nvecs * out_bytes) {
        return;
    }
    int vi = tid / out_bytes, b = tid % out_bytes;
    ((unsigned char *)out)[(size_t)vi * out_bytes + b] = (b < in_bytes) ? in[(size_t)vi * in_bytes + b] : 0;
}

// Precomputed GF(16) reduction tables for compute_A_fused_bs
__device__ __constant__ unsigned char bs_tail_tab[16]; // filled by gf16_luts_init

__constant__ unsigned char gf16_mul_lut[256]; // lut[(a<<4)|b] = a*b
__constant__ unsigned char gf16_inv_lut[16];  // inv[a] = a^-1 (or 0)

static int gf16_luts_ready = 0;

static inline unsigned char host_gf16_mul(unsigned char a, unsigned char b)
{
    unsigned char r = 0;
    a &= 0x0F;
    b &= 0x0F;

    while (b) {
        if (b & 1) {
            r ^= a;
        }
        b >>= 1;
        unsigned char carry = a & 0x08;
        a <<= 1;
        if (carry) {
            a ^= 0x03;
        }
        a &= 0x0F;
    }

    return r & 0x0F;
}

static void gf16_luts_init(void)
{
    if (gf16_luts_ready) {
        return;
    }

    unsigned char mul[256];
    unsigned char inv[16];

    for (int i = 0; i < 16; i++) {
        inv[i] = 0;
        for (int j = 0; j < 16; j++) {
            mul[i * 16 + j] = host_gf16_mul((unsigned char)i, (unsigned char)j);
            if (inv[i] == 0 && mul[i * 16 + j] == 1) {
                inv[i] = (unsigned char)j;
            }
        }
    }

    cudaMemcpyToSymbol(gf16_mul_lut, mul, 256);
    cudaMemcpyToSymbol(gf16_inv_lut, inv, 16);
    {
        static const unsigned char h_tail[4] = MAYO_P_f_tail_init;
        unsigned char tab[16];
        for (int tt = 0; tt < 4; tt++) {
            for (int j = 0; j < 4; j++) {
                tab[4 * tt + j] = host_gf16_mul(h_tail[tt], (unsigned char)(1 << j));
            }
        }
        cudaMemcpyToSymbol(bs_tail_tab, tab, 16);
    }

    gf16_luts_ready = 1;
}

#define PRINT_RESULT 0

int mayo2_sign_signature(unsigned char *sig, size_t *siglen, const unsigned char *m, size_t mlen,
                         const unsigned char *csk);

#include <cstdlib>

// MAYO_TC_PS: 0 = nibble-sliced P * s^t, 1 = Tensor Cores, 2 = compare
static int mayo_tc_ps_mode(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MAYO_TC_PS");
        v = (e != NULL) ? atoi(e) : 0;
    }
    return v;
}
static uint32_t *g_tc_PA = NULL; // key-side A fragments, built once per key

static int mayo_bench_include_copies(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MAYO_BENCH_INCLUDE_COPIES");
        v = (e != NULL && atoi(e) != 0) ? 1 : 0;
    }
    return v;
}

static int mayo_distinct_msgs(void)
{
    const char *e = getenv("MAYO_DISTINCT_MSGS");
    return (e != NULL && atoi(e) > 0) ? 1 : 0;
}
int mayo_expand_sk(const unsigned char *csk, sk_t *sk);

typedef struct mayo2_key_ctx {
    unsigned char *d_seed_sk;
    unsigned char *d_O; // MAYO_P_v * MAYO_P_o bytes, decoded O
    uint64_t *d_P_limbs;
    uint64_t *d_L;
    uint64_t *d_P1_planes;
    uint64_t *d_L_planes;
} mayo2_key_ctx_t;

int mayo2_key_init(mayo2_key_ctx_t *ctx, const unsigned char *csk);
void mayo2_key_destroy(mayo2_key_ctx_t *ctx);

typedef struct mayo2_sign_workspace {
    size_t mlen;

    unsigned char *h_msg;
    unsigned char *h_s;

    unsigned char *d_tmp;
    unsigned char *d_m;
    unsigned char *d_salt;
    unsigned char *d_tenc;
    unsigned char *d_t;
    unsigned char *d_V;
    unsigned char *d_digest;
    unsigned char *d_Vdec;
    uint64_t *d_VL;
    unsigned char *d_y;
    unsigned char *d_A;
    uint64_t *d_Pv;
    unsigned char *d_A_out;
    unsigned char *d_x;
    unsigned char *d_r;

    int *d_sol_batch;
    unsigned char *d_s;

    // retry queue: unsolved slots packed at the front
    unsigned char *d_tmp2, *d_t2, *d_tenc2;
    int *d_map, *d_map2, *d_solved, *d_count;
    unsigned char *d_sig, *h_sig; // encoded signatures of the whole batch
    int distinct;                 // a different message per slot
} mayo2_sign_workspace_t;

void mayo2_workspace_init(mayo2_sign_workspace_t *ws, size_t mlen);
void mayo2_workspace_destroy(mayo2_sign_workspace_t *ws);

int mayo2_sign_with_ctx(mayo2_key_ctx_t *ctx, mayo2_sign_workspace_t *ws, unsigned char *sig, size_t *siglen,
                        const unsigned char *m, size_t mlen, float *elapsed_ms_out);

static void encode(const unsigned char *m, unsigned char *menc, int mlen)
{
    int i;
    for (i = 0; i < mlen / 2; ++i, m += 2) {
        menc[i] = (*m) | (*(m + 1) << 4);
    }

    if (mlen % 2 == 1) {
        menc[i] = (*m);
    }
}

__global__ void decode(const unsigned char *m, unsigned char *mdec, int mdeclen, int blocks_per_sig,
                       int input_stride, int output_stride);

__global__ void decode_v_kbatch(const unsigned char *m, unsigned char *mdec, int mdeclen, int blocks_per_sig,
                                int input_stride, int output_stride, int k_in_stride, int k_out_stride);

__global__ void unpack_m_vecs(const unsigned char *in, uint64_t *out, int total_vecs);

__global__ void P1P1t_times_O(const uint64_t *__restrict__ P1, const unsigned char *__restrict__ O,
                              uint64_t *__restrict__ acc);

__device__ __forceinline__ int upper_triangular_index(int r, int c, int param_v);

void mayo_secure_clear(void *mem, size_t size);

template <int BS_MAT_COLS>
__global__ void mul_add_mat_x_m_mat(const unsigned char *__restrict__ mat,
                                    const uint64_t *__restrict__ bs_mat, uint64_t *__restrict__ acc,
                                    int mat_stride, int bs_mat_stride, int acc_stride);

__global__ void P1_times_Vt(const uint64_t *P1, const unsigned char *Vdec, uint64_t *Pv, int p_limbs_per_sig,
                            int Vdec_stride, int Pv_stride);

__global__ void compute_rhs_finegrain(const uint64_t *vPv, const unsigned char *t, unsigned char *y);

__global__ void pack_P1_to_planes_kernel(const uint64_t *d_P_limbs, uint64_t *d_P1_planes, int p1_vecs);

__global__ void P1_times_Vt_bs(const uint64_t *P1_planes, const unsigned char *Vdec, uint64_t *Pv,
                               int p_limbs_per_sig, int Vdec_stride, int Pv_stride);

template <int BS_MAT_COLS, bool PACK_OUT>
__global__ void
mul_add_mat_x_m_mat_bs(const unsigned char *__restrict__ mat, const uint64_t *__restrict__ bs_mat_planes,
                       uint64_t *__restrict__ acc_planes, int mat_stride, int bs_mat_stride, int acc_stride);

__global__ void compute_rhs_finegrain_bs(const uint64_t *vPv, const unsigned char *t, unsigned char *y);

__global__ void compute_A_fused_bs(const uint64_t *VtL_planes, unsigned char *A_out);

__global__ void sample_solution_prepare(const unsigned char *r, unsigned char *x, unsigned char *A,
                                        const unsigned char *y);

__device__ void EF_device_cooperative(unsigned char *A, int tid);

__global__ void sample_solution_finish(unsigned char *A, unsigned char *x, int *sol_batch_out, int *solved);

__global__ void build_s(const unsigned char *O, const unsigned char *Vdec, const unsigned char *x,
                        unsigned char *s, int O_stride, const int *map, int nsig);

__global__ void pack_sk_kernel(sk_t *d_sk, const unsigned char *d_O, const uint64_t *d_P_limbs);

__global__ void build_tmp_for_salt_kernel(unsigned char *d_tmp, const unsigned char *d_digest,
                                          const unsigned char *d_seed_sk, int seed_sk_stride);

__global__ void insert_salt_in_tmp_kernel(unsigned char *d_tmp, const unsigned char *d_salt);

__global__ void set_ctr_in_tmp_kernel(unsigned char *d_tmp, int ctr_group);

void printElement(unsigned char *element, int n, const char *title)
{
    printf("\r\n");
    printf("%s\r\n", title);
    for (int i = 0; i < n; i++) {
        printf("%02x, ", element[i]);
    }
    printf("\r\n\n");
}

void printBatch(unsigned char *element, int n, int nBatch, const char *title)
{
    for (int i = 0; i < nBatch; i++) {
        printf("Batch %d\r\n", i);
        printElement(element + i * n, n, title);
    }
}

void printFirstAndLast(unsigned char *element, int n, int nBatch, const char *title)
{
    printf("%s\r\n", title);
    for (int i = 0; i < nBatch; i++) {
        printf("Batch %d\r\n", i);
        printf("First: %02x\r\n", *(element + i * n));
        printf("First: %02x\r\n", *(element + (i + 1) * n - 1));
    }
}

void print_sk_batch_first_last(sk_t *sk, int nBatch)
{
    for (int i = 0; i < nBatch; i++) {
        printf("Batch %d\r\n", i);

        printf("O first: %02x\r\n", sk[i].O[0]);
        printf("O last : %02x\r\n", sk[i].O[MAYO_P_v * MAYO_P_o - 1]);

        printf("p first: %02llx\r\n", (unsigned long long)sk[i].p[0]);
        printf("p last : %02llx\r\n", (unsigned long long)sk[i].p[P1_LIMBS_MAX + P2_LIMBS_MAX - 1]);

        printf("\r\n");
    }
}

int mayo2_sign(unsigned char *sm, size_t *smlen, const unsigned char *m, size_t mlen,
               const unsigned char *csk)
{
    int ret = MAYO_OK;
    const int param_sig_bytes = MAYO_P_sig_bytes;
    size_t siglen;

    memmove(sm + param_sig_bytes, m, mlen);
    ret = mayo2_sign_signature(sm, &siglen, sm + param_sig_bytes, mlen, csk);
    if (ret != MAYO_OK || siglen != (size_t)param_sig_bytes) {
        memset(sm, 0, siglen + mlen);
        return MAYO_ERR;
    }

    *smlen = siglen + mlen;
    return ret;
}

int mayo2_key_init(mayo2_key_ctx_t *ctx, const unsigned char *csk)
{
    gf16_luts_init();

    const int p1_vecs = MAYO_P_v * (MAYO_P_v + 1) / 2;
    const int p2_vecs = MAYO_P_v * MAYO_P_o;

    const int packed_bytes = MAYO_P_P1_bytes + MAYO_P_P2_bytes;

    const int p1_limbs = p1_vecs * MAYO_P_m_vec_limbs;

    const int S_stride = MAYO_P_pk_seed_bytes + MAYO_P_O_bytes;

    unsigned char *d_S;
    uint64_t *d_rkeys;

    cudaMalloc((void **)&ctx->d_seed_sk, MAYO_P_sk_seed_bytes);
    cudaMalloc((void **)&d_S, S_stride);
    cudaMalloc((void **)&ctx->d_O, MAYO_P_v * MAYO_P_o);
    cudaMalloc((void **)&ctx->d_P_limbs, (size_t)(P1_LIMBS_MAX + P2_LIMBS_MAX) * sizeof(uint64_t));
    cudaMalloc((void **)&d_rkeys, AES128_SK_EXP_WORDS * sizeof(uint64_t));

    assert(((uintptr_t)ctx->d_P_limbs) % alignof(uint64_t) == 0);

    cudaMemcpy(ctx->d_seed_sk, csk, MAYO_P_sk_seed_bytes, cudaMemcpyHostToDevice);

    shake256<<<1, 25>>>(d_S, S_stride, ctx->d_seed_sk, MAYO_P_sk_seed_bytes, 0);

    int in_bytes_per_sig = (MAYO_P_v * MAYO_P_o + 1) / 2;
    int blocks_per_sig = (in_bytes_per_sig + THREADS - 1) / THREADS;

    decode<<<blocks_per_sig, THREADS>>>(d_S + MAYO_P_pk_seed_bytes, ctx->d_O, MAYO_P_v * MAYO_P_o,
                                        blocks_per_sig, S_stride, MAYO_P_v * MAYO_P_o);

    aes128_key_expand_batch<<<1, 1>>>(d_S, S_stride, d_rkeys, 1);

    const int groups_per_batch = (packed_bytes + 63) / 64;
    int blocks_aes = (groups_per_batch + THREADS - 1) / THREADS;

    unsigned char *d_packed;
    cudaMalloc((void **)&d_packed, (size_t)groups_per_batch * 64);
    aes128_ctr4x_batch<<<blocks_aes, THREADS>>>(d_packed, packed_bytes, d_rkeys, groups_per_batch, 1);
    {
        const int nv = p1_vecs + p2_vecs;
        const int total = nv * MAYO_P_m_vec_limbs * 8;
        unpack_vecs_single<<<(total + THREADS - 1) / THREADS, THREADS>>>(d_packed, ctx->d_P_limbs, nv);
    }
    cudaFree(d_packed);

    P1P1t_times_O<<<MAYO_P_v, MAYO_P_m_vec_limbs * 32>>>(ctx->d_P_limbs, ctx->d_O, ctx->d_P_limbs + p1_limbs);

    ctx->d_L = ctx->d_P_limbs + p1_limbs;

    cudaMalloc((void **)&ctx->d_P1_planes, (size_t)p1_vecs * MAYO_P_m_vec_limbs * sizeof(uint64_t));

    {
        int blocks = (p1_vecs + THREADS - 1) / THREADS;
        pack_P1_to_planes_kernel<<<blocks, THREADS>>>(ctx->d_P_limbs, ctx->d_P1_planes, p1_vecs);
    }

    cudaMalloc((void **)&ctx->d_L_planes, (size_t)p2_vecs * MAYO_P_m_vec_limbs * sizeof(uint64_t));

    {
        int blocks = (p2_vecs + THREADS - 1) / THREADS;
        pack_P1_to_planes_kernel<<<blocks, THREADS>>>(ctx->d_P_limbs + p1_limbs, ctx->d_L_planes, p2_vecs);
    }

    cudaFree(d_S);
    cudaFree(d_rkeys);

    cudaError_t err = cudaDeviceSynchronize();

    return (err == cudaSuccess) ? MAYO_OK : MAYO_ERR;
}

void mayo2_key_destroy(mayo2_key_ctx_t *ctx)
{
    cudaFree(ctx->d_seed_sk);
    cudaFree(ctx->d_O);
    cudaFree(ctx->d_P_limbs);
    cudaFree(ctx->d_P1_planes);
    cudaFree(ctx->d_L_planes);

    ctx->d_seed_sk = NULL;
    ctx->d_O = NULL;
    ctx->d_P_limbs = NULL;
    ctx->d_L = NULL;
    ctx->d_P1_planes = NULL;
    ctx->d_L_planes = NULL;
}

void mayo2_workspace_init(mayo2_sign_workspace_t *ws, size_t mlen)
{
    const int A_out_bytes_per_batch = MAYO_P_m * MAYO_P_A_cols;
    const int x_stride = MAYO_P_k * MAYO_P_n;
    const int s_stride = MAYO_P_k * MAYO_P_n;

    ws->mlen = mlen;

    cudaMallocHost((void **)&ws->h_msg, mlen * BATCH);
    cudaMallocHost((void **)&ws->h_s, s_stride * BATCH);

    cudaMalloc((void **)&ws->d_tmp,
               (MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1) * BATCH);
    cudaMalloc((void **)&ws->d_digest, MAYO_P_digest_bytes * BATCH);
    cudaMalloc((void **)&ws->d_m, mlen * BATCH);
    cudaMalloc((void **)&ws->d_salt, MAYO_P_salt_bytes * BATCH);
    cudaMalloc((void **)&ws->d_tenc, MAYO_P_m_bytes * BATCH);
    cudaMalloc((void **)&ws->d_t, MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_V, (MAYO_P_k * MAYO_P_v_bytes + MAYO_P_r_bytes) * BATCH);
    cudaMalloc((void **)&ws->d_Vdec, MAYO_P_v * MAYO_P_k * BATCH);
    cudaMalloc((void **)&ws->d_VL, MAYO_P_k * MAYO_P_o * MAYO_P_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&ws->d_Pv, MAYO_P_v * MAYO_P_k * MAYO_P_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&ws->d_A, MAYO_P_k * MAYO_P_k * MAYO_P_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&ws->d_y, MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_A_out, A_out_bytes_per_batch * BATCH);
    cudaMalloc((void **)&ws->d_r, MAYO_P_k * MAYO_P_o * BATCH);
    cudaMalloc((void **)&ws->d_x, x_stride * BATCH);
    cudaMalloc((void **)&ws->d_sol_batch, sizeof(int));
    cudaMalloc((void **)&ws->d_s, s_stride * BATCH);

    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;
#ifdef R3_TENC_STRIDE
    cudaMalloc((void **)&ws->d_tenc2, (size_t)R3_TENC_STRIDE * BATCH);
#else
    ws->d_tenc2 = NULL;
#endif
    cudaMalloc((void **)&ws->d_tmp2, (size_t)tmp_bytes * BATCH);
    cudaMalloc((void **)&ws->d_t2, (size_t)MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_map, BATCH * sizeof(int));
    cudaMalloc((void **)&ws->d_map2, BATCH * sizeof(int));
    cudaMalloc((void **)&ws->d_solved, BATCH * sizeof(int));
    cudaMalloc((void **)&ws->d_count, sizeof(int));
    cudaMalloc((void **)&ws->d_sig, (size_t)MAYO_P_sig_bytes * BATCH);
    cudaMallocHost((void **)&ws->h_sig, (size_t)MAYO_P_sig_bytes * BATCH);
    ws->distinct = mayo_distinct_msgs();
}

void mayo2_workspace_destroy(mayo2_sign_workspace_t *ws)
{
    cudaFreeHost(ws->h_msg);
    cudaFreeHost(ws->h_s);

    cudaFree(ws->d_tmp);
    cudaFree(ws->d_m);
    cudaFree(ws->d_salt);
    cudaFree(ws->d_tenc);
    cudaFree(ws->d_t);
    cudaFree(ws->d_V);
    cudaFree(ws->d_digest);
    cudaFree(ws->d_Vdec);
    cudaFree(ws->d_VL);
    cudaFree(ws->d_Pv);
    cudaFree(ws->d_A);
    cudaFree(ws->d_y);
    cudaFree(ws->d_A_out);
    cudaFree(ws->d_r);
    cudaFree(ws->d_x);
    cudaFree(ws->d_sol_batch);
    cudaFree(ws->d_s);
    cudaFree(ws->d_tmp2);
    cudaFree(ws->d_t2);
    cudaFree(ws->d_tenc2);
    cudaFree(ws->d_map);
    cudaFree(ws->d_map2);
    cudaFree(ws->d_solved);
    cudaFree(ws->d_count);
    cudaFree(ws->d_sig);
    cudaFreeHost(ws->h_sig);
}

__global__ void compact_unsolved(int nb, const int *solved, const int *map_in, int *map_out,
                                 const unsigned char *tmp_in, unsigned char *tmp_out,
                                 const unsigned char *t_in, unsigned char *t_out,
                                 const unsigned char *tenc_in, unsigned char *tenc_out, int tenc_stride,
                                 int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb || solved[i]) {
        return;
    }

    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;
    int j = atomicAdd(count, 1);
    map_out[j] = map_in ? map_in[i] : i;
    for (int b = 0; b < tmp_bytes; b++) {
        tmp_out[(size_t)j * tmp_bytes + b] = tmp_in[(size_t)i * tmp_bytes + b];
    }
    for (int b = 0; b < MAYO_P_m; b++) {
        t_out[(size_t)j * MAYO_P_m + b] = t_in[(size_t)i * MAYO_P_m + b];
    }
    if (tenc_in) {
        for (int b = 0; b < tenc_stride; b++) {
            tenc_out[(size_t)j * tenc_stride + b] = tenc_in[(size_t)i * tenc_stride + b];
        }
    }
}

// sig = encode(s) || salt, one thread per signature byte.
__global__ void encode_sigs(const unsigned char *s, const unsigned char *salt, unsigned char *sig)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int sb = MAYO_P_sig_bytes, kn = MAYO_P_k * MAYO_P_n;
    const int body = MAYO_P_sig_bytes - MAYO_P_salt_bytes;
    if (tid >= BATCH * sb) {
        return;
    }

    int b = tid / sb, i = tid % sb;
    const unsigned char *s_b = s + (size_t)b * kn;
    unsigned char v = 0;
    if (i < body) {
        int e = 2 * i;
        if (e < kn) {
            v = s_b[e] & 0xF;
        }
        if (e + 1 < kn) {
            v |= (s_b[e + 1] & 0xF) << 4;
        }
    } else {
        v = salt[(size_t)b * MAYO_P_salt_bytes + i - body];
    }
    sig[tid] = v;
}

int mayo2_sign_with_ctx(mayo2_key_ctx_t *ctx, mayo2_sign_workspace_t *ws, unsigned char *sig, size_t *siglen,
                        const unsigned char *m, size_t mlen, float *elapsed_ms_out)
{
    int ret = MAYO_OK;

    const int A_out_bytes_per_batch = MAYO_P_m * MAYO_P_A_cols;
    const int ko = MAYO_P_k * MAYO_P_o;
    const int x_stride = MAYO_P_k * MAYO_P_n;
    const int s_stride = MAYO_P_k * MAYO_P_n;

    unsigned char *h_msg = ws->h_msg;
    unsigned char *h_s = ws->h_s;
    unsigned char *d_tmp = ws->d_tmp;
    unsigned char *d_m = ws->d_m;
    unsigned char *d_salt = ws->d_salt;
    unsigned char *d_tenc = ws->d_tenc;
    unsigned char *d_t = ws->d_t;
    unsigned char *d_V = ws->d_V;
    unsigned char *d_digest = ws->d_digest;
    unsigned char *d_Vdec = ws->d_Vdec;
    unsigned char *d_y = ws->d_y;
    unsigned char *d_A = ws->d_A;
    unsigned char *d_A_out = ws->d_A_out;
    unsigned char *d_x = ws->d_x;
    unsigned char *d_r = ws->d_r;
    unsigned char *d_s = ws->d_s;
    uint64_t *d_VL = ws->d_VL;
    uint64_t *d_Pv = ws->d_Pv;
    int *d_sol_batch = ws->d_sol_batch;

    uint64_t *d_P_batch_alias = ctx->d_P_limbs;
    unsigned char *d_O_batch_alias = ctx->d_O;
    uint64_t *d_L = ctx->d_L;

    float gpu_t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < BATCH; i++) {
        for (int j = 0; j < mlen; j++) {
            h_msg[j + i * mlen] = m[j];
        }
        if (ws->distinct) { // slot 0 keeps the base message
            for (int j = 0; j < 4 && j < (int)mlen; j++) {
                h_msg[j + i * mlen] ^= (unsigned char)(i >> (8 * j));
            }
        }
    }

    if (!mayo_bench_include_copies()) {
        cudaMemcpy(d_m, h_msg, mlen * BATCH, cudaMemcpyHostToDevice);
    }
    cudaMemset(d_A_out, 0, A_out_bytes_per_batch * BATCH);
    cudaMemset(d_x, 0, x_stride * BATCH);

    cudaEventRecord(start);
    if (mayo_bench_include_copies()) {
        cudaMemcpy(d_m, h_msg, mlen * BATCH, cudaMemcpyHostToDevice);
    }

    shake256<<<BATCH, 25>>>(d_digest, MAYO_P_digest_bytes, d_m, mlen, 0);

    int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;
    int total_tmp = BATCH * tmp_bytes;
    int blocks_tmp = (total_tmp + THREADS - 1) / THREADS;

    build_tmp_for_salt_kernel<<<blocks_tmp, THREADS>>>(d_tmp, d_digest, ctx->d_seed_sk, 0);

    shake256<<<BATCH, 25>>>(d_salt, MAYO_P_salt_bytes, d_tmp,
                            MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes,
                            MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1);

    int total_salt = BATCH * MAYO_P_salt_bytes;
    int blocks_salt = (total_salt + THREADS - 1) / THREADS;

    insert_salt_in_tmp_kernel<<<blocks_salt, THREADS>>>(d_tmp, d_salt);

    shake256<<<BATCH, 25>>>(d_tenc, MAYO_P_m_bytes, d_tmp, MAYO_P_digest_bytes + MAYO_P_salt_bytes,
                            tmp_bytes);

    int blocks = (((MAYO_P_m + 1) / 2) + THREADS - 1) / THREADS;
    decode<<<blocks * BATCH, THREADS>>>(d_tenc, d_t, MAYO_P_m, blocks, MAYO_P_m_bytes, MAYO_P_m);

    // Retry queue
    unsigned char *tmp_w = d_tmp, *tmp_n = ws->d_tmp2;
    unsigned char *t_w = d_t, *t_n = ws->d_t2;
#ifdef R3_TENC_STRIDE
    unsigned char *tenc_w = d_tenc, *tenc_n = ws->d_tenc2;
#endif
    int *map_w = NULL, *map_n = ws->d_map;
    int nb = BATCH;

    for (int ctr = 0; ctr <= 255 && nb > 0; ctr++) {
        int blocks_ctr = (nb + THREADS - 1) / THREADS;
        set_ctr_in_tmp_kernel<<<blocks_ctr, THREADS>>>(tmp_w, ctr);
#ifdef R3_TENC_STRIDE
        if (ctr > 0) {
            r3_lambda_gen<<<(nb * R3_LAMBDA_CHUNKS + THREADS - 1) / THREADS, THREADS>>>(tenc_w, ws->d_lambda,
                                                                                        nb);
        }
#endif

        shake256<<<nb, 25>>>(d_V, MAYO_P_k * MAYO_P_v_bytes + MAYO_P_r_bytes, tmp_w, tmp_bytes, tmp_bytes);

        int V_blocks = (((MAYO_P_v + 1) / 2) + THREADS - 1) / THREADS;
        int V_bytes_per_batch = MAYO_P_k * MAYO_P_v_bytes + MAYO_P_r_bytes;
        int Vdec_per_batch = MAYO_P_k * MAYO_P_v;
        decode_v_kbatch<<<V_blocks * nb * MAYO_P_k, THREADS>>>(
            d_V, d_Vdec, MAYO_P_v, V_blocks, V_bytes_per_batch, Vdec_per_batch, MAYO_P_v_bytes, MAYO_P_v);

        int total_VL_bs = nb * MAYO_P_k * MAYO_P_o; // one thread per output vector
        mul_add_mat_x_m_mat_bs<MAYO_P_o, false><<<(total_VL_bs + THREADS - 1) / THREADS, THREADS>>>(
            d_Vdec, ctx->d_L_planes, d_VL, MAYO_P_k * MAYO_P_v, 0, MAYO_P_k * MAYO_P_o * MAYO_P_m_vec_limbs);

        int total_Pv_bs = nb * MAYO_P_v * MAYO_P_k;
        P1_times_Vt_bs<<<(total_Pv_bs + THREADS - 1) / THREADS, THREADS>>>(
            ctx->d_P1_planes, d_Vdec, d_Pv, 0, MAYO_P_k * MAYO_P_v, MAYO_P_v * MAYO_P_k * MAYO_P_m_vec_limbs);

        int total_A_bs = nb * MAYO_P_k * MAYO_P_k;
        mul_add_mat_x_m_mat_bs<MAYO_P_k, true><<<(total_A_bs + THREADS - 1) / THREADS, THREADS>>>(
            d_Vdec, d_Pv, (uint64_t *)d_A, MAYO_P_k * MAYO_P_v, MAYO_P_v * MAYO_P_k * MAYO_P_m_vec_limbs,
            MAYO_P_k * MAYO_P_k * MAYO_P_m_vec_limbs);
#ifdef R3_TENC_STRIDE
        r3_lambda_fold_sign<<<(nb * (MAYO_P_k + MAYO_P_o) + THREADS - 1) / THREADS, THREADS>>>(
            ws->d_lambda, d_Vdec, ctx->d_O, d_VL, (uint64_t *)d_A, nb);
#endif

        compute_rhs_finegrain<<<nb, MAYO_P_m_vec_limbs>>>((uint64_t *)d_A, t_w, d_y);
        compute_A_fused_bs<<<nb, THREADS>>>(d_VL, d_A_out);

        int r_blocks = (((MAYO_P_k * MAYO_P_o + 1) / 2) + THREADS - 1) / THREADS;
        decode<<<r_blocks * nb, THREADS>>>(d_V + MAYO_P_k * MAYO_P_v_bytes, d_r, MAYO_P_k * MAYO_P_o,
                                           r_blocks, MAYO_P_k * MAYO_P_v_bytes + MAYO_P_r_bytes,
                                           MAYO_P_k * MAYO_P_o);

        int total_prepare = nb * MAYO_P_m;
        if (nb * ko > total_prepare) {
            total_prepare = nb * ko;
        }
        sample_solution_prepare<<<(total_prepare + THREADS - 1) / THREADS, THREADS>>>(d_r, d_x, d_A_out, d_y);

        cudaMemsetAsync(ws->d_solved, 0, nb * sizeof(int));
        cudaMemsetAsync(d_sol_batch, 0x7F, sizeof(int));
        sample_solution_finish<<<nb, MAYO_P_solver_threads>>>(d_A_out, d_x, d_sol_batch, ws->d_solved);

        build_s<<<(nb * MAYO_P_k * MAYO_P_n + THREADS - 1) / THREADS, THREADS>>>(d_O_batch_alias, d_Vdec, d_x,
                                                                                 d_s, 0, map_w, nb);

        cudaMemsetAsync(ws->d_count, 0, sizeof(int));
#ifdef R3_TENC_STRIDE
        compact_unsolved<<<(nb + THREADS - 1) / THREADS, THREADS>>>(nb, ws->d_solved, map_w, map_n, tmp_w,
                                                                    tmp_n, t_w, t_n, tenc_w, tenc_n,
                                                                    R3_TENC_STRIDE, ws->d_count);
        unsigned char *tenc_s = tenc_w;
        tenc_w = tenc_n;
        tenc_n = tenc_s;
#else
        compact_unsolved<<<(nb + THREADS - 1) / THREADS, THREADS>>>(
            nb, ws->d_solved, map_w, map_n, tmp_w, tmp_n, t_w, t_n, NULL, NULL, 0, ws->d_count);
#endif
        cudaMemcpy(&nb, ws->d_count, sizeof(int), cudaMemcpyDeviceToHost);

        unsigned char *swp = tmp_w;
        tmp_w = tmp_n;
        tmp_n = swp;
        swp = t_w;
        t_w = t_n;
        t_n = swp;
        map_w = map_n;
        map_n = (map_w == ws->d_map) ? ws->d_map2 : ws->d_map;
    }

    if (nb > 0) {
        ret = MAYO_ERR;
    } else {
        encode_sigs<<<(BATCH * MAYO_P_sig_bytes + THREADS - 1) / THREADS, THREADS>>>(d_s, d_salt, ws->d_sig);
        cudaMemcpy(ws->h_sig, ws->d_sig, (size_t)MAYO_P_sig_bytes * BATCH, cudaMemcpyDeviceToHost);
        memcpy(sig, ws->h_sig, MAYO_P_sig_bytes);
        *siglen = MAYO_P_sig_bytes;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_t, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (elapsed_ms_out != NULL) {
        *elapsed_ms_out = gpu_t;
    }

    return ret;
}

int mayo2_sign_batch(unsigned char *msgs, unsigned char *sigs, const unsigned char *m, size_t mlen,
                     const unsigned char *csk)
{
    mayo2_key_ctx_t ctx;
    int ret = mayo2_key_init(&ctx, csk);
    if (ret != MAYO_OK) {
        return ret;
    }

    mayo2_sign_workspace_t ws;
    mayo2_workspace_init(&ws, mlen);
    ws.distinct = 1;

    unsigned char sig[MAYO_P_sig_bytes];
    size_t siglen = 0;
    ret = mayo2_sign_with_ctx(&ctx, &ws, sig, &siglen, m, mlen, NULL);
    if (ret == MAYO_OK) {
        memcpy(msgs, ws.h_msg, mlen * BATCH);
        memcpy(sigs, ws.h_sig, (size_t)MAYO_P_sig_bytes * BATCH);
    }
    mayo2_workspace_destroy(&ws);
    mayo2_key_destroy(&ctx);
    return ret;
}

int mayo2_sign_signature(unsigned char *sig, size_t *siglen, const unsigned char *m, size_t mlen,
                         const unsigned char *csk)
{
    mayo2_key_ctx_t ctx;

    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);
    cudaEventRecord(init_start);

    int ret = mayo2_key_init(&ctx, csk);

    cudaEventRecord(init_stop);
    cudaEventSynchronize(init_stop);

    float key_init_ms = 0.0f;
    cudaEventElapsedTime(&key_init_ms, init_start, init_stop);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);

    printf("Key expansion (once per csk): %.6f ms\r\n", key_init_ms);

    if (ret != MAYO_OK) {
        return ret;
    }

    mayo2_sign_workspace_t ws;
    mayo2_workspace_init(&ws, mlen);

    float throughput[REPEATS + 1];
    float sumThput = 0;

    for (int rep = 0; rep <= REPEATS; rep++) {
        float gpu_t = 0.0f;

        ret = mayo2_sign_with_ctx(&ctx, &ws, sig, siglen, m, mlen, &gpu_t);

        printf("GPU signing time per signature: %.6f ms\n", gpu_t / BATCH);
        throughput[rep] = BATCH / gpu_t;
    }

    mayo2_workspace_destroy(&ws);
    mayo2_key_destroy(&ctx);

    for (int i = 0; i < REPEATS; i++) {
        float sig_per_sec = throughput[i + 1] * 1000.0f;
        printf("Throughput %d: %.6f sig/s\r\n", i + 1, sig_per_sec);
        sumThput += sig_per_sec;
    }

    printf("Avg Throughput: %.6f sig/s\r\n", sumThput / REPEATS);

    return ret;
}

__global__ void decode(const unsigned char *m, unsigned char *mdec, int mdeclen, int blocks_per_sig,
                       int input_stride, int output_stride)
{
    int sig_id = blockIdx.x / blocks_per_sig;

    int local_block = blockIdx.x % blocks_per_sig;
    int local_idx = local_block * blockDim.x + threadIdx.x;

    int packed_bytes = (mdeclen + 1) / 2;

    if (local_idx >= packed_bytes) {
        return;
    }

    const unsigned char *m_sig = m + sig_id * input_stride;

    unsigned char *mdec_sig = mdec + sig_id * output_stride;

    unsigned char byte = m_sig[local_idx];

    int out0 = 2 * local_idx;
    int out1 = 2 * local_idx + 1;

    if (out0 < mdeclen) {
        mdec_sig[out0] = byte & 0x0F;
    }

    if (out1 < mdeclen) {
        mdec_sig[out1] = byte >> 4;
    }
}

__global__ void decode_v_kbatch(const unsigned char *m, unsigned char *mdec, int mdeclen, int blocks_per_sig,
                                int input_stride, int output_stride, int k_in_stride, int k_out_stride)
{
    int per_k_blocks = gridDim.x / MAYO_P_k;

    int k_id = blockIdx.x / per_k_blocks;
    int rem = blockIdx.x % per_k_blocks;

    int sig_id = rem / blocks_per_sig;

    int local_block = rem % blocks_per_sig;
    int local_idx = local_block * blockDim.x + threadIdx.x;

    int packed_bytes = (mdeclen + 1) / 2;

    if (local_idx >= packed_bytes) {
        return;
    }

    const unsigned char *m_sig = m + k_id * k_in_stride + sig_id * input_stride;

    unsigned char *mdec_sig = mdec + k_id * k_out_stride + sig_id * output_stride;

    unsigned char byte = m_sig[local_idx];

    int out0 = 2 * local_idx;
    int out1 = 2 * local_idx + 1;

    if (out0 < mdeclen) {
        mdec_sig[out0] = byte & 0x0F;
    }

    if (out1 < mdeclen) {
        mdec_sig[out1] = byte >> 4;
    }
}

__global__ void unpack_m_vecs(const unsigned char *in, uint64_t *out, int total_vecs)
{
    int m_vec_limbs = (MAYO_P_m + 15) / 16;

    int in_bytes_per_vec = MAYO_P_m / 2;
    int out_bytes_per_vec = m_vec_limbs * sizeof(uint64_t);

    int in_bytes_per_sig = total_vecs * in_bytes_per_vec;
    int out_bytes_per_sig = total_vecs * out_bytes_per_vec;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_bytes = BATCH * out_bytes_per_sig;

    if (tid >= total_bytes) {
        return;
    }

    int sig_id = tid / out_bytes_per_sig;
    int local_tid = tid % out_bytes_per_sig;

    int vec_index = local_tid / out_bytes_per_vec;
    int byte_index = local_tid % out_bytes_per_vec;

    unsigned char *out_bytes = (unsigned char *)out;

    if (byte_index < in_bytes_per_vec) {
        out_bytes[sig_id * out_bytes_per_sig + vec_index * out_bytes_per_vec + byte_index] =
            in[sig_id * in_bytes_per_sig + vec_index * in_bytes_per_vec + byte_index];
    } else {
        out_bytes[sig_id * out_bytes_per_sig + vec_index * out_bytes_per_vec + byte_index] = 0;
    }
}

__global__ void P1P1t_times_O(const uint64_t *__restrict__ P1, const unsigned char *__restrict__ O,
                              uint64_t *__restrict__ acc)
{
    int sig_id = blockIdx.x / MAYO_P_v;
    int row = blockIdx.x % MAYO_P_v;

    int limb = threadIdx.x / 32;
    int k = threadIdx.x % 32;

    bool active = (k < MAYO_P_o);

    int p_limbs_per_sig = P1_LIMBS_MAX + P2_LIMBS_MAX;
    int o_bytes_per_sig = MAYO_P_v * MAYO_P_o;

    const uint64_t *P1_sig = P1 + sig_id * p_limbs_per_sig;

    const unsigned char *O_sig = O + sig_id * o_bytes_per_sig;

    uint64_t *acc_sig = acc + sig_id * p_limbs_per_sig;

    uint64_t local = 0;

    for (int col = 0; col < MAYO_P_v; col++) {
        if (col == row) {
            continue;
        }

        int r;
        int c;

        if (row < col) {
            r = row;
            c = col;
        } else {
            r = col;
            c = row;
        }

        int p1_index = upper_triangular_index(r, c, MAYO_P_v);

        uint64_t p1_val = 0;

        if (k == 0) {
            p1_val = P1_sig[p1_index * MAYO_P_m_vec_limbs + limb];
        }

        p1_val = __shfl_sync(0xFFFFFFFF, p1_val, 0);

        if (active) {
            uint8_t a = O_sig[col * MAYO_P_o + k];

            local ^= gf16_vec_mul(p1_val, a);
        }
    }

    if (active) {
        uint64_t *acc_entry = acc_sig + (row * MAYO_P_o + k) * MAYO_P_m_vec_limbs + limb;

        *acc_entry ^= local;
    }
}

__device__ __forceinline__ int upper_triangular_index(int r, int c, int param_v)
{
    return r * param_v - (r * (r - 1)) / 2 + (c - r);
}

void mayo_secure_clear(void *mem, size_t size)
{
    typedef void *(*memset_t)(void *, int, size_t);
    static volatile memset_t memset_func = memset;
    memset_func(mem, 0, size);
}

template <int BS_MAT_COLS>
__global__ void mul_add_mat_x_m_mat(const unsigned char *__restrict__ mat,
                                    const uint64_t *__restrict__ bs_mat, uint64_t *__restrict__ acc,
                                    int mat_stride, int bs_mat_stride, int acc_stride)
{
    constexpr int m_vec_limbs = MAYO_P_m_vec_limbs;
    constexpr int mat_rows = MAYO_P_k;
    constexpr int mat_cols = MAYO_P_v;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int outputs_per_batch = mat_rows * BS_MAT_COLS * m_vec_limbs;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int limb = local % m_vec_limbs;

    int tmp = local / m_vec_limbs;
    int k = tmp % BS_MAT_COLS;
    int r = tmp / BS_MAT_COLS;

    const unsigned char *mat_b = mat + batch * mat_stride;

    const uint64_t *bs_mat_b = bs_mat + batch * bs_mat_stride;

    uint64_t *acc_b = acc + batch * acc_stride;

    uint64_t sum = 0;

    for (int c = 0; c < mat_cols; c++) {
        uint8_t a = mat_b[r * mat_cols + c];

        uint64_t b = bs_mat_b[m_vec_limbs * (c * BS_MAT_COLS + k) + limb];

        sum ^= gf16_vec_mul(b, a);
    }

    acc_b[m_vec_limbs * (r * BS_MAT_COLS + k) + limb] = sum;
}

__global__ void P1_times_Vt(const uint64_t *P1, const unsigned char *Vdec, uint64_t *Pv, int p_limbs_per_sig,
                            int Vdec_stride, int Pv_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int outputs_per_batch = MAYO_P_v * MAYO_P_k * MAYO_P_m_vec_limbs;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int limb = local % MAYO_P_m_vec_limbs;

    int tmp = local / MAYO_P_m_vec_limbs;
    int k = tmp % MAYO_P_k;
    int r = tmp / MAYO_P_k;

    const uint64_t *P1_b = P1 + batch * p_limbs_per_sig;

    const unsigned char *Vdec_b = Vdec + batch * Vdec_stride;

    uint64_t *Pv_b = Pv + batch * Pv_stride;

    uint64_t sum = 0;

    for (int c = r; c < MAYO_P_v; c++) {
        int p1_index = upper_triangular_index(r, c, MAYO_P_v);

        uint64_t p1_val = P1_b[p1_index * MAYO_P_m_vec_limbs + limb];

        uint8_t v = Vdec_b[k * MAYO_P_v + c];

        sum ^= gf16_vec_mul(p1_val, v);
    }

    Pv_b[(r * MAYO_P_k + k) * MAYO_P_m_vec_limbs + limb] = sum;
}

__global__ void compute_rhs_finegrain(const uint64_t *vPv, const unsigned char *t, unsigned char *y)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    const int param_k = MAYO_P_k;
    const int param_m = MAYO_P_m;
    const int m_vec_limbs = MAYO_P_m_vec_limbs;

    if (tid >= m_vec_limbs) {
        return;
    }

    const int top_pos = ((param_m - 1) % 16) * 4;

    const uint64_t *vPv_b = vPv + batch * param_k * param_k * m_vec_limbs;

    const unsigned char *t_b = t + batch * param_m;

    unsigned char *y_b = y + batch * param_m;

    __shared__ uint64_t temp[M_VEC_LIMBS_MAX];
    __shared__ unsigned char top;

    temp[tid] = 0;

    __syncthreads();

    for (int i = param_k - 1; i >= 0; i--) {
        for (int j = i; j < param_k; j++) {
            if (tid == m_vec_limbs - 1) {
                top = (unsigned char)((temp[m_vec_limbs - 1] >> top_pos) & 0xF);
            }

            __syncthreads();

            uint64_t old_limb = temp[tid];
            uint64_t carry_in = 0;

            if (tid > 0) {
                carry_in = temp[tid - 1] >> 60;
            }

            __syncthreads();

            temp[tid] = (old_limb << 4) ^ carry_in;

            __syncthreads();

            if (tid == 0) {
                unsigned char *temp_bytes = (unsigned char *)temp;

                for (int jj = 0; jj < F_TAIL_LEN; jj++) {
                    unsigned char val = mul_f(top, d_f_tail_64[jj]);

                    if ((jj & 1) == 0) {
                        temp_bytes[jj / 2] ^= val;
                    } else {
                        temp_bytes[jj / 2] ^= val << 4;
                    }
                }
            }

            __syncthreads();

            uint64_t a = vPv_b[(i * param_k + j) * m_vec_limbs + tid];

            uint64_t b = 0;

            if (i != j) {
                b = vPv_b[(j * param_k + i) * m_vec_limbs + tid];
            }

            temp[tid] ^= a ^ b;

            __syncthreads();
        }
    }

    for (int idx = tid; idx < param_m; idx += m_vec_limbs) {
        unsigned char *temp_bytes = (unsigned char *)temp;

        unsigned char temp_i = (temp_bytes[idx / 2] >> (4 * (idx & 1))) & 0xF;

        y_b[idx] = t_b[idx] ^ temp_i;
    }
}

__global__ void pack_P1_to_planes_kernel(const uint64_t *d_P_limbs, uint64_t *d_P1_planes, int p1_vecs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= p1_vecs) {
        return;
    }

    const uint64_t *pack = d_P_limbs + (size_t)tid * MAYO_P_m_vec_limbs;
    uint64_t *planes = d_P1_planes + (size_t)tid * MAYO_P_m_vec_limbs;

    hyb_pack_to_planes(pack, planes);
}

// Bitsliced P1 * Vt
__global__ void P1_times_Vt_bs(const uint64_t *P1_planes, const unsigned char *Vdec, uint64_t *Pv,
                               int p_limbs_per_sig, int Vdec_stride, int Pv_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int outputs_per_batch = MAYO_P_v * MAYO_P_k;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int k = local % MAYO_P_k;
    int r = local / MAYO_P_k;

    const uint64_t *P1_b = P1_planes + batch * p_limbs_per_sig;
    const unsigned char *Vdec_b = Vdec + batch * Vdec_stride;
    uint64_t *Pv_b = Pv + batch * Pv_stride;

    uint64_t acc[MAYO_P_m_vec_limbs];
#pragma unroll
    for (int w = 0; w < MAYO_P_m_vec_limbs; w++) {
        acc[w] = 0;
    }

    for (int c = r; c < MAYO_P_v; c++) {
        int p1_index = upper_triangular_index(r, c, MAYO_P_v);

        const uint64_t *pv = P1_b + p1_index * MAYO_P_m_vec_limbs;

        uint8_t v = Vdec_b[k * MAYO_P_v + c];

        hyb_mul_add(acc, pv, v);
    }

    uint64_t *out = Pv_b + (r * MAYO_P_k + k) * MAYO_P_m_vec_limbs;
#pragma unroll
    for (int w = 0; w < MAYO_P_m_vec_limbs; w++) {
        out[w] = acc[w];
    }
}

// Bitsliced mat * bs_mat for the vPv path
template <int BS_MAT_COLS, bool PACK_OUT>
__global__ void
mul_add_mat_x_m_mat_bs(const unsigned char *__restrict__ mat, const uint64_t *__restrict__ bs_mat_planes,
                       uint64_t *__restrict__ acc_planes, int mat_stride, int bs_mat_stride, int acc_stride)
{
    constexpr int m_vec_limbs = MAYO_P_m_vec_limbs;
    constexpr int mat_rows = MAYO_P_k;
    constexpr int mat_cols = MAYO_P_v;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int outputs_per_batch = mat_rows * BS_MAT_COLS;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int k = local % BS_MAT_COLS;
    int r = local / BS_MAT_COLS;

    const unsigned char *mat_b = mat + batch * mat_stride;
    const uint64_t *bs_mat_b = bs_mat_planes + batch * bs_mat_stride;
    uint64_t *acc_b = acc_planes + batch * acc_stride;

    uint64_t acc[MAYO_P_m_vec_limbs];
#pragma unroll
    for (int w = 0; w < MAYO_P_m_vec_limbs; w++) {
        acc[w] = 0;
    }

    for (int c = 0; c < mat_cols; c++) {
        uint8_t a = mat_b[r * mat_cols + c];

        const uint64_t *b = bs_mat_b + m_vec_limbs * (c * BS_MAT_COLS + k);

        hyb_mul_add(acc, b, a);
    }

    uint64_t *out = acc_b + m_vec_limbs * (r * BS_MAT_COLS + k);
    if (PACK_OUT) {
        uint64_t packed[MAYO_P_m_vec_limbs];
        hyb_planes_to_pack(acc, packed);
#pragma unroll
        for (int w = 0; w < MAYO_P_m_vec_limbs; w++) {
            out[w] = packed[w];
        }
    } else {
#pragma unroll
        for (int w = 0; w < MAYO_P_m_vec_limbs; w++) {
            out[w] = acc[w];
        }
    }
}

// Bitsliced RHS
__global__ void compute_rhs_finegrain_bs(const uint64_t *vPv, const unsigned char *t, unsigned char *y)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    const int param_k = MAYO_P_k;
    const int param_m = MAYO_P_m;
    const int m_vec_limbs = MAYO_P_m_vec_limbs;

    if (tid >= m_vec_limbs) {
        return;
    }

    const uint64_t *vPv_b = vPv + batch * param_k * param_k * m_vec_limbs;

    const unsigned char *t_b = t + batch * param_m;
    unsigned char *y_b = y + batch * param_m;

    __shared__ uint64_t planes_s[MAYO_P_m_vec_limbs];
    __shared__ unsigned char top_bits[MAYO_P_m_vec_limbs];

    planes_s[tid] = 0;

    __syncthreads();

    for (int i = param_k - 1; i >= 0; i--) {
        for (int j = i; j < param_k; j++) {
            // capture the nibble that will overflow the shift
            top_bits[tid] = (unsigned char)((planes_s[tid] >> 63) & 1);

            __syncthreads();

            unsigned char top =
                (unsigned char)(top_bits[0] | (top_bits[1] << 1) | (top_bits[2] << 2) | (top_bits[3] << 3));

            // shift the whole vector up by one nibble (one element)
            planes_s[tid] <<= 1;

            __syncthreads();

            // reduce: fold top * tail into coefficients 0..F_TAIL_LEN-1
            for (int jj = 0; jj < F_TAIL_LEN; jj++) {
                unsigned char s = gf16_mul_ct(top, d_f_tail_64[jj]);

                planes_s[tid] ^= ((uint64_t)(-(int64_t)((s >> tid) & 1))) & (1ULL << jj);
            }

            __syncthreads();

            // accumulate vPv[i][j] + vPv[j][i] (bit-plane words)
            uint64_t a = vPv_b[(i * param_k + j) * m_vec_limbs + tid];

            uint64_t b = 0;

            if (i != j) {
                b = vPv_b[(j * param_k + i) * m_vec_limbs + tid];
            }

            planes_s[tid] ^= a ^ b;

            __syncthreads();
        }
    }

    // extract per-coefficient nibbles back to packed bytes
    for (int idx = tid; idx < param_m; idx += m_vec_limbs) {
        unsigned char temp_i =
            (unsigned char)(((planes_s[0] >> idx) & 1) | (((planes_s[1] >> idx) & 1) << 1) |
                            (((planes_s[2] >> idx) & 1) << 2) | (((planes_s[3] >> idx) & 1) << 3));

        y_b[idx] = t_b[idx] ^ temp_i;
    }
}

__global__ void compute_A_fused_bs(const uint64_t *VtL_planes, unsigned char *A_out)
{
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;

    const int param_k = MAYO_P_k;
    const int param_o = MAYO_P_o;
    const int m_vec_limbs = MAYO_P_m_vec_limbs;

    const int A_width = ((MAYO_P_o * MAYO_P_k + 15) / 16) * 16;
    const int pairs = (MAYO_P_k + 1) * MAYO_P_k / 2;
    const int A_cols = MAYO_P_k * MAYO_P_o + 1;

    __shared__ uint64_t A_work[((MAYO_P_o * MAYO_P_k + 15) / 16) * 16 *
                               ((MAYO_P_m + (MAYO_P_k + 1) * MAYO_P_k / 2 + 15) / 16)];
    __shared__ uint64_t tscratch[((MAYO_P_o * MAYO_P_k + 15) / 16) * 16 *
                                 ((MAYO_P_m + (MAYO_P_k + 1) * MAYO_P_k / 2 + 15) / 16)];
    __shared__ uint64_t VL_packed[MAYO_P_k * MAYO_P_o * MAYO_P_m_vec_limbs];

    const int A_work_limbs = A_width * ((MAYO_P_m + pairs + 15) / 16);

    for (int i = tid; i < A_work_limbs; i += blockDim.x) {
        A_work[i] = 0;
    }

    {
        const uint64_t *VtL_b = VtL_planes + batch * param_k * param_o * m_vec_limbs;

        for (int idx = tid; idx < param_k * param_o; idx += blockDim.x) {
            hyb_planes_to_pack(VtL_b + idx * m_vec_limbs, VL_packed + idx * m_vec_limbs);
        }
    }

    __syncthreads();

    {
        const int total_build = pairs * param_o * m_vec_limbs * 2;
        const uint64_t *VtL_b = VL_packed;

        for (int idx = tid; idx < total_build; idx += blockDim.x) {
            int local = idx;

            int side = local % 2;
            local /= 2;

            int limb = local % m_vec_limbs;
            local /= m_vec_limbs;

            int c = local % param_o;
            local /= param_o;

            int pair_id = local;

            int i_found = 0;
            int j_found = 0;
            int count = 0;

            for (int i = 0; i < param_k; i++) {
                for (int j = param_k - 1; j >= i; j--) {
                    if (count == pair_id) {
                        i_found = i;
                        j_found = j;
                    }
                    count++;
                }
            }

            int i = i_found;
            int j = j_found;

            if (side == 1 && i == j) {
                continue;
            }

            int bits_to_shift = (4 * pair_id) % 64;
            int words_to_shift = (4 * pair_id) / 64;

            int source_row;
            int target_col;

            if (side == 0) {
                source_row = j;
                target_col = param_o * i + c;
            } else {
                source_row = i;
                target_col = param_o * j + c;
            }

            uint64_t value = VtL_b[(source_row * param_o + c) * m_vec_limbs + limb];

            int pos = target_col + (limb + words_to_shift) * A_width;

            atomicXor((unsigned long long *)&A_work[pos], (unsigned long long)(value << bits_to_shift));

            if (bits_to_shift > 0) {
                int pos_hi = target_col + (limb + words_to_shift + 1) * A_width;

                atomicXor((unsigned long long *)&A_work[pos_hi],
                          (unsigned long long)(value >> (64 - bits_to_shift)));
            }
        }
    }

    __syncthreads();

    {
        const int transpose_blocks = A_work_limbs / 16;
        const int transpose_total = transpose_blocks * 16;

        for (int idx = tid; idx < transpose_total; idx += blockDim.x) {
            int blk = idx / 16;
            int w = idx % 16;

            const uint64_t *in = A_work + blk * 16;

            uint64_t out = 0;

            for (int i = 0; i < 16; i++) {
                out |= ((in[i] >> (4 * w)) & 0xFULL) << (4 * i);
            }

            tscratch[idx] = out;
        }

        __syncthreads();

        for (int idx = tid; idx < transpose_total; idx += blockDim.x) {
            A_work[idx] = tscratch[idx];
        }
    }

    __syncthreads();

    {
        const int c_blocks = A_width / 16;
        const int total_reduce = c_blocks * pairs;
        const uint64_t low_bit_in_nibble = 0x1111111111111111ULL;

        for (int idx = tid; idx < total_reduce; idx += blockDim.x) {
            int local = idx;

            int pair_row = local % pairs;
            local /= pairs;

            int c_block = local;

            int c = c_block * 16;
            int r = MAYO_P_m + pair_row;

            size_t pos = (r / 16) * A_width + c + (r % 16);
            uint64_t word = A_work[pos];

            uint64_t t0 = word & low_bit_in_nibble;
            uint64_t t1 = (word >> 1) & low_bit_in_nibble;
            uint64_t t2 = (word >> 2) & low_bit_in_nibble;
            uint64_t t3 = (word >> 3) & low_bit_in_nibble;

            for (int t = 0; t < F_TAIL_LEN; t++) {
                uint64_t value = t0 * bs_tail_tab[4 * t + 0] ^ t1 * bs_tail_tab[4 * t + 1] ^
                                 t2 * bs_tail_tab[4 * t + 2] ^ t3 * bs_tail_tab[4 * t + 3];

                int rr = r + t - MAYO_P_m;

                size_t dst = (rr / 16) * A_width + c + (rr % 16);

                atomicXor((unsigned long long *)&A_work[dst], (unsigned long long)value);
            }
        }
    }

    __syncthreads();

    {
        const int useful_cols = A_cols - 1;
        const int total_decode = MAYO_P_m * useful_cols;

        unsigned char *A_out_b = A_out + batch * MAYO_P_m * A_cols;

        for (int idx = tid; idx < total_decode; idx += blockDim.x) {
            int local = idx;

            int col = local % useful_cols;
            local /= useful_cols;

            int row = local;

            int c_base = (col / 16) * 16;
            int c_off = col % 16;

            int row_block = (row / 16) * 16;
            int row_off = row % 16;

            size_t pos = row_block * (A_width / 16) + c_base + row_off;
            uint64_t word = A_work[pos];

            A_out_b[row * A_cols + col] = (word >> (4 * c_off)) & 0xF;
        }
    }
}

__global__ void sample_solution_prepare(const unsigned char *r, unsigned char *x, unsigned char *A,
                                        const unsigned char *y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int ko = MAYO_P_k * MAYO_P_o;
    const int m = MAYO_P_m;
    const int A_cols = MAYO_P_k * MAYO_P_o + 1;
    const int x_stride = MAYO_P_k * MAYO_P_n;

    int total_x = BATCH * ko;
    int total_rows = BATCH * m;

    if (tid < total_x) {
        int batch = tid / ko;
        int idx = tid % ko;

        x[batch * x_stride + idx] = r[batch * ko + idx];
    }

    if (tid < total_rows) {
        int batch = tid / m;
        int row = tid % m;

        unsigned char *A_b = A + batch * m * A_cols;

        const unsigned char *r_b = r + batch * ko;

        const unsigned char *y_b = y + batch * m;

        A_b[row * A_cols + (A_cols - 1)] = 0;

        unsigned char acc = 0;

        for (int col = 0; col < ko; col++) {
            unsigned char a = A_b[row * A_cols + col] & 0x0F;
            unsigned char b = r_b[col] & 0x0F;

            acc ^= gf16_mul_ct(a, b);
        }

        A_b[row * A_cols + (A_cols - 1)] = (y_b[row] ^ acc) & 0x0F;
    }
}

__device__ void EF_device_cooperative(unsigned char *A, int tid)
{
    const int nrows = MAYO_P_m;
    const int ncols = MAYO_P_A_cols;
    const int row_len = (ncols + 15) / 16;

    __shared__ uint64_t packed_A[((MAYO_P_k * MAYO_P_o + 1 + 15) / 16) * MAYO_P_m];
    __shared__ uint64_t pivot_row2[(MAYO_P_k * MAYO_P_o + 1 + 15) / 16];
    __shared__ int pivot_row_idx_s;
    __shared__ uint64_t pivot_is_zero_s;
    __shared__ uint32_t ballot_per_warp_s[8];

    if (tid < nrows) {
        ef_pack_m_vec_device(A + tid * ncols, packed_A + tid * row_len, ncols);
    }

    if (tid == 0) {
        pivot_row_idx_s = 0;
    }

    __syncthreads();

    for (int pivot_col = 0; pivot_col < ncols; pivot_col++) {
        int pivot_row_lower_bound = pivot_col + nrows - ncols;

        if (pivot_row_lower_bound < 0) {
            pivot_row_lower_bound = 0;
        }

        int pivot_row_upper_bound = pivot_col;

        if (pivot_row_upper_bound > nrows - 1) {
            pivot_row_upper_bound = nrows - 1;
        }

        int pivot_row_idx = pivot_row_idx_s;

        int row_search_upper = pivot_row_upper_bound + 32;

        if (row_search_upper > nrows - 1) {
            row_search_upper = nrows - 1;
        }

        int window_size = row_search_upper - pivot_row_lower_bound + 1;
        int base = pivot_row_idx - pivot_row_lower_bound;

        bool in_window = (tid < window_size);
        int row = pivot_row_lower_bound + tid;

        uint64_t is_nonzero_mask = 0;

        if (in_window) {
            unsigned char b = m_extract_element_device(packed_A + row * row_len, pivot_col);

            is_nonzero_mask = ct_compare_64_device((uint64_t)b, 0ULL);
        }

        uint64_t at_or_after_base_mask = ~ct_64_is_greater_than_device((uint64_t)base, (uint64_t)tid);

        bool eligible_bit = in_window && ((at_or_after_base_mask & is_nonzero_mask) != 0);

        uint32_t warp_id = tid >> 5;
        uint32_t ballot = __ballot_sync(0xFFFFFFFFu, eligible_bit);

        if ((tid & 31) == 0) {
            ballot_per_warp_s[warp_id] = ballot;
        }

        __syncthreads();

        int winner_tid = base;
        {
            const int nwarps = (blockDim.x + 31) / 32; // public
            for (int w = nwarps - 1; w >= 0; w--) {
                const int f = __ffs(ballot_per_warp_s[w]);
                const int take = -(int)(f != 0);
                winner_tid = (take & (32 * w + f - 1)) | (~take & winner_tid);
            }
        }

        if (tid < row_len) {
            uint64_t acc = 0;

            for (int i = 0; i < window_size; i++) {
                int r = pivot_row_lower_bound + i;

                uint64_t in_range_mask = (~ct_64_is_greater_than_device((uint64_t)base, (uint64_t)i)) &
                                         (~ct_64_is_greater_than_device((uint64_t)i, (uint64_t)winner_tid));

                acc ^= in_range_mask & packed_A[r * row_len + tid];
            }

            unsigned char my_nibble = (unsigned char)((acc >> (4 * (pivot_col % 16))) & 0xF);

            int owner_lane = pivot_col / 16;

            unsigned char pivot = __shfl_sync((1u << row_len) - 1, my_nibble, owner_lane);

            unsigned char inverse = gf16_inv_ct(pivot);

            pivot_row2[tid] = gf16_vec_mul(acc, inverse);

            if (tid == 0) {
                uint64_t pivot_is_zero = ~ct_compare_64_device((uint64_t)pivot, 0);

                pivot_is_zero_s = pivot_is_zero;

                pivot_row_idx_s = pivot_row_idx + (int)(1 & ~pivot_is_zero);
            }
        }

        __syncthreads();

        uint64_t pivot_is_zero = pivot_is_zero_s;

        if (tid >= pivot_row_lower_bound && tid <= pivot_row_upper_bound) {
            uint64_t do_copy = ~ct_compare_64_device((uint64_t)tid, (uint64_t)pivot_row_idx) & ~pivot_is_zero;

            uint64_t do_not_copy = ~do_copy;

            for (int col = 0; col < row_len; col++) {
                packed_A[tid * row_len + col] =
                    (do_not_copy & packed_A[tid * row_len + col]) ^ (do_copy & pivot_row2[col]);
            }
        }

        __syncthreads();

        if (tid >= pivot_row_lower_bound && tid < nrows) {
            unsigned char below_pivot = (unsigned char)(tid > pivot_row_idx);

            unsigned char elt_to_elim = m_extract_element_device(packed_A + tid * row_len, pivot_col);

            vec_mul_add_u64_device(row_len, pivot_row2, below_pivot * elt_to_elim, packed_A + tid * row_len);
        }

        __syncthreads();
    }

    if (tid < nrows) {
        unsigned char temp[MAYO_P_k * MAYO_P_o + 1 + 15];

        ef_unpack_m_vec_device(row_len, packed_A + tid * row_len, temp);

        for (int j = 0; j < ncols; j++) {
            A[tid * ncols + j] = temp[j];
        }
    }

    __syncthreads();
}

__global__ void sample_solution_finish(unsigned char *A, unsigned char *x, int *sol_batch_out, int *solved)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    const int m = MAYO_P_m;
    const int ko = MAYO_P_k * MAYO_P_o;
    const int A_cols = MAYO_P_k * MAYO_P_o + 1;
    const int x_stride = MAYO_P_k * MAYO_P_n;

    unsigned char *A_b = A + batch * m * A_cols;

    unsigned char *x_b = x + batch * x_stride;

    EF_device_cooperative(A_b, tid);

    __shared__ unsigned char u_sh[MAYO_P_A_cols];
    __shared__ int win_col0_s;
    __shared__ int win_len_s;
    __shared__ unsigned char full_rank_s;

    if (tid == 0) {
        unsigned char full_rank = 0;

        for (int i = 0; i < A_cols - 1; i++) {
            full_rank |= A_b[(m - 1) * A_cols + i];
        }

        full_rank_s = full_rank;
    }

    __syncthreads();

    if (full_rank_s == 0) {
        return;
    }

    for (int row = m - 1; row >= 0; row--) {
        if (tid == 0) {
            unsigned char finished = 0;

            int col_upper_bound = row + (32 / (m - row));

            if (col_upper_bound > ko) {
                col_upper_bound = ko;
            }

            win_col0_s = row;
            win_len_s = col_upper_bound - row + 1;

            for (int col = row; col <= col_upper_bound; col++) {
                unsigned char correct_column = ct_compare_8_device(A_b[row * A_cols + col], 0) & ~finished;

                unsigned char u = correct_column & A_b[row * A_cols + A_cols - 1];

                x_b[col] ^= u;
                u_sh[col - row] = u;

                finished |= correct_column;
            }
        }

        __syncthreads();

        int win_col0 = win_col0_s;
        int win_len = win_len_s;
        int num_groups = (row + 7) / 8;

        if (tid < num_groups) {
            int i = tid * 8;

            uint64_t acc = 0;

            for (int c = 0; c < win_len; c++) {
                int col = win_col0 + c;

                uint64_t tmp = 0;
#pragma unroll
                for (int j8 = 0; j8 < 8; j8++) {
                    if (i + j8 < m) {
                        tmp ^= ((uint64_t)A_b[(i + j8) * A_cols + col]) << (8 * j8);
                    }
                }

                acc ^= mul_fx8_device(u_sh[c], tmp);
            }

#pragma unroll
            for (int j8 = 0; j8 < 8; j8++) {
                if (i + j8 < m) {
                    A_b[(i + j8) * A_cols + A_cols - 1] ^= (acc >> (8 * j8)) & 0xF;
                }
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(sol_batch_out, batch);
        if (solved) {
            solved[batch] = 1;
        }
    }
}

__global__ void build_s(const unsigned char *O, const unsigned char *Vdec, const unsigned char *x,
                        unsigned char *s, int O_stride, const int *map, int nsig)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int k = MAYO_P_k;
    const int n = MAYO_P_n;
    const int o = MAYO_P_o;
    const int v = MAYO_P_v;

    const int Vdec_stride = MAYO_P_k * MAYO_P_v;
    const int x_stride = MAYO_P_k * MAYO_P_n;
    const int s_stride = MAYO_P_k * MAYO_P_n;

    int total = nsig * k * n;

    if (tid >= total) {
        return;
    }

    int batch = tid / (k * n);
    int local = tid % (k * n);

    int i = local / n;   // 0..k-1
    int pos = local % n; // 0..n-1

    const unsigned char *O_b = O + batch * O_stride;

    const unsigned char *Vdec_b = Vdec + batch * Vdec_stride;

    const unsigned char *x_b = x + batch * x_stride;

    unsigned char *s_b = s + (size_t)(map ? map[batch] : batch) * s_stride;

    const unsigned char *x_i = x_b + i * o;

    if (pos < v) {
        unsigned char Ox = 0;

        for (int j = 0; j < o; j++) {
            unsigned char a = O_b[pos * o + j] & 0x0F;
            unsigned char b = x_i[j] & 0x0F;

            Ox ^= gf16_mul_ct(a, b);
        }

        unsigned char vi = Vdec_b[i * v + pos] & 0x0F;

        s_b[i * n + pos] = (vi ^ Ox) & 0x0F;
    } else {
        int oil_pos = pos - v;

        s_b[i * n + pos] = x_i[oil_pos] & 0x0F;
    }
}

__global__ void pack_sk_kernel(sk_t *d_sk, const unsigned char *d_O, const uint64_t *d_P_limbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int O_elems = MAYO_P_v * MAYO_P_o;
    const int P_elems = P1_LIMBS_MAX + P2_LIMBS_MAX;

    const int elems_per_sig = O_elems + P_elems;
    const int total_elems = BATCH * elems_per_sig;

    if (tid >= total_elems) {
        return;
    }

    int sig_id = tid / elems_per_sig;
    int local = tid % elems_per_sig;

    if (local < O_elems) {
        d_sk[sig_id].O[local] = d_O[sig_id * O_elems + local];
    } else {
        int p_idx = local - O_elems;

        d_sk[sig_id].p[p_idx] = d_P_limbs[sig_id * P_elems + p_idx];
    }
}

__global__ void build_tmp_for_salt_kernel(unsigned char *d_tmp, const unsigned char *d_digest,
                                          const unsigned char *d_seed_sk, int seed_sk_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;

    int total = BATCH * tmp_bytes;

    if (tid >= total) {
        return;
    }

    int batch_id = tid / tmp_bytes;
    int local = tid % tmp_bytes;

    unsigned char value = 0;

    if (local < MAYO_P_digest_bytes) {
        value = d_digest[batch_id * MAYO_P_digest_bytes + local];
    } else if (local < MAYO_P_digest_bytes + MAYO_P_salt_bytes) {
        value = 1;
    } else if (local < MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes) {
        int sk_idx = local - MAYO_P_digest_bytes - MAYO_P_salt_bytes;

        value = d_seed_sk[batch_id * seed_sk_stride + sk_idx];
    } else {
        value = 0;
    }

    d_tmp[batch_id * tmp_bytes + local] = value;
}

__global__ void insert_salt_in_tmp_kernel(unsigned char *d_tmp, const unsigned char *d_salt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total = BATCH * MAYO_P_salt_bytes;
    if (tid >= total) {
        return;
    }

    int batch_id = tid / MAYO_P_salt_bytes;
    int j = tid % MAYO_P_salt_bytes;

    int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;

    d_tmp[batch_id * tmp_bytes + MAYO_P_digest_bytes + j] = d_salt[batch_id * MAYO_P_salt_bytes + j];
}

__global__ void set_ctr_in_tmp_kernel(unsigned char *d_tmp, int ctr)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= BATCH) {
        return;
    }

    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes + MAYO_P_sk_seed_bytes + 1;

    d_tmp[b * tmp_bytes + tmp_bytes - 1] = (unsigned char)ctr;
}

typedef struct mayo2_verify_ctx {
    uint64_t *d_P_limbs; // P1 || P2 || P3
} mayo2_verify_ctx_t;

typedef struct mayo2_verify_workspace {
    size_t mlen;

    unsigned char *d_msg;    // mlen * BATCH
    unsigned char *d_sig;    // MAYO_P_sig_bytes * BATCH
    unsigned char *d_digest; // MAYO_P_digest_bytes * BATCH
    unsigned char *d_tmp;    // (digest + salt) * BATCH
    unsigned char *d_tenc;   // MAYO_P_m_bytes * BATCH
    unsigned char *d_t;      // MAYO_P_m * BATCH
    unsigned char *d_s;      // MAYO_P_k * MAYO_P_n * BATCH
    uint64_t *d_PS;          // n * k * m_vec_limbs * BATCH
    uint64_t *d_SPS;         // k * k * m_vec_limbs * BATCH
    unsigned char *d_y;      // m * BATCH
    unsigned char *d_zero;   // m * BATCH
    int *d_results;          // BATCH
    int *h_results;
} mayo2_verify_workspace_t;

int mayo2_verify_key_init(mayo2_verify_ctx_t *ctx, const unsigned char *cpk);
void mayo2_verify_key_destroy(mayo2_verify_ctx_t *ctx);
void mayo2_verify_workspace_init(mayo2_verify_workspace_t *ws, size_t mlen);
void mayo2_verify_workspace_destroy(mayo2_verify_workspace_t *ws);
int mayo2_verify_batch(mayo2_verify_ctx_t *ctx, mayo2_verify_workspace_t *ws, const unsigned char *msgs,
                       const unsigned char *sigs, size_t mlen, int *results, float *elapsed_ms_out);

// tmp[batch] = digest[batch] || salt(sig[batch])
__global__ void verify_build_tmp(const unsigned char *d_digest, const unsigned char *d_sig,
                                 unsigned char *d_tmp)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes;

    int total = BATCH * tmp_bytes;

    if (tid >= total) {
        return;
    }

    int batch = tid / tmp_bytes;
    int local = tid % tmp_bytes;

    unsigned char value;

    if (local < MAYO_P_digest_bytes) {
        value = d_digest[batch * MAYO_P_digest_bytes + local];
    } else {
        int salt_off = local - MAYO_P_digest_bytes;
        value = d_sig[batch * MAYO_P_sig_bytes + (MAYO_P_sig_bytes - MAYO_P_salt_bytes) + salt_off];
    }

    d_tmp[batch * tmp_bytes + local] = value;
}

// PS[batch][r][c] = (P * s^t)[r][c] rows r <  v
__global__ void verify_calc_PS(const uint64_t *__restrict__ d_P, const unsigned char *__restrict__ d_s,
                               uint64_t *__restrict__ d_PS)
{
    const int n = MAYO_P_n;
    const int v = MAYO_P_v;
    const int o = MAYO_P_o;
    const int k = MAYO_P_k;
    const int ml = MAYO_P_m_vec_limbs;

    const int outputs_per_batch = n * k * ml;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int limb = local % ml;
    int tmp = local / ml;
    int c = tmp % k;
    int r = tmp / k;

    const unsigned char *s = d_s + batch * (k * n);

    uint64_t sum = 0;

    if (r < v) {
        for (int j = r; j < v; j++) {
            int p1 = upper_triangular_index(r, j, v);
            uint64_t p1v = d_P[p1 * ml + limb];
            uint8_t sv = s[c * n + j];
            sum ^= gf16_vec_mul(p1v, sv);
        }
        for (int j = 0; j < o; j++) {
            uint64_t p2v = d_P[P1_LIMBS_MAX + (r * o + j) * ml + limb];
            uint8_t sv = s[c * n + v + j];
            sum ^= gf16_vec_mul(p2v, sv);
        }
    } else {
        int rr = r - v;
        for (int j = r; j < n; j++) {
            int p3 = upper_triangular_index(rr, j - v, o);
            uint64_t p3v = d_P[P1_LIMBS_MAX + P2_LIMBS_MAX + p3 * ml + limb];
            uint8_t sv = s[c * n + j];
            sum ^= gf16_vec_mul(p3v, sv);
        }
    }

    d_PS[batch * outputs_per_batch + (r * k + c) * ml + limb] = sum;
}

// SPS[batch][row][col] = sum_j s[row][j] * PS[j][col]
__global__ void verify_calc_SPS(const unsigned char *__restrict__ d_s, const uint64_t *__restrict__ d_PS,
                                uint64_t *__restrict__ d_SPS)
{
    const int n = MAYO_P_n;
    const int k = MAYO_P_k;
    const int ml = MAYO_P_m_vec_limbs;

    const int outputs_per_batch = k * k * ml;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int limb = local % ml;
    int tmp = local / ml;
    int col = tmp % k;
    int row = tmp / k;

    const unsigned char *s = d_s + batch * (k * n);
    const uint64_t *PS = d_PS + batch * (n * k * ml);

    uint64_t sum = 0;

    for (int j = 0; j < n; j++) {
        uint8_t sv = s[row * n + j];
        uint64_t ps = PS[(j * k + col) * ml + limb];
        sum ^= gf16_vec_mul(ps, sv);
    }

    d_SPS[batch * outputs_per_batch + (row * k + col) * ml + limb] = sum;
}

// result[batch] = 0 if y == t (valid), 1 otherwise (invalid)
__global__ void verify_compare(const unsigned char *__restrict__ d_y, const unsigned char *__restrict__ d_t,
                               int *__restrict__ d_results)
{
    const int m = MAYO_P_m;

    int batch = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= m) {
        return;
    }

    const unsigned char *y = d_y + batch * m;
    const unsigned char *t = d_t + batch * m;

    __shared__ int s_bad[MAYO_P_m];

    s_bad[tid] = (y[tid] != t[tid]);

    __syncthreads();

    if (tid == 0) {
        int bad = 0;
        for (int i = 0; i < m; i++) {
            bad |= s_bad[i];
        }
        d_results[batch] = bad;
    }
}

// Expand the compressed public key once
int mayo2_verify_key_init(mayo2_verify_ctx_t *ctx, const unsigned char *cpk)
{
    const int packed_bytes = MAYO_P_P1_bytes + MAYO_P_P2_bytes;
    const int groups_per_batch = (packed_bytes + 63) / 64;

    unsigned char *d_seed_pk;
    uint64_t *d_rkeys;

    cudaMalloc((void **)&ctx->d_P_limbs, (P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX) * sizeof(uint64_t));
    cudaMalloc((void **)&d_seed_pk, MAYO_P_pk_seed_bytes);
    cudaMalloc((void **)&d_rkeys, AES128_SK_EXP_WORDS * sizeof(uint64_t));

    cudaMemcpy(d_seed_pk, cpk, MAYO_P_pk_seed_bytes, cudaMemcpyHostToDevice);

    aes128_key_expand_batch<<<1, 1>>>(d_seed_pk, MAYO_P_pk_seed_bytes, d_rkeys, 1);

    int blocks_aes = (groups_per_batch + THREADS - 1) / THREADS;
    {
        unsigned char *d_packed;
        cudaMalloc((void **)&d_packed, (size_t)groups_per_batch * 64 + MAYO_P_P3_bytes);
        aes128_ctr4x_batch<<<blocks_aes, THREADS>>>(d_packed, packed_bytes, d_rkeys, groups_per_batch, 1);
        const int p12_vecs = MAYO_P_v * (MAYO_P_v + 1) / 2 + MAYO_P_v * MAYO_P_o;
        const int p3_vecs = MAYO_P_o * (MAYO_P_o + 1) / 2;
        int total = p12_vecs * MAYO_P_m_vec_limbs * 8;
        unpack_vecs_single<<<(total + THREADS - 1) / THREADS, THREADS>>>(d_packed, ctx->d_P_limbs, p12_vecs);
        unsigned char *d_p3 = d_packed + (size_t)groups_per_batch * 64;
        cudaMemcpy(d_p3, cpk + MAYO_P_pk_seed_bytes, MAYO_P_P3_bytes, cudaMemcpyHostToDevice);
        total = p3_vecs * MAYO_P_m_vec_limbs * 8;
        unpack_vecs_single<<<(total + THREADS - 1) / THREADS, THREADS>>>(
            d_p3, ctx->d_P_limbs + P1_LIMBS_MAX + P2_LIMBS_MAX, p3_vecs);
        cudaDeviceSynchronize();
        cudaFree(d_packed);
    }

    cudaFree(d_seed_pk);
    cudaFree(d_rkeys);

    if (mayo_tc_ps_mode() != 0) {
        cudaFree(g_tc_PA);
        cudaMalloc((void **)&g_tc_PA, TC_PA_WORDS * sizeof(uint32_t));
        tc_build_key_frags<<<(unsigned)((TC_PA_WORDS + THREADS - 1) / THREADS), THREADS>>>(ctx->d_P_limbs,
                                                                                           g_tc_PA);
    }

    cudaError_t err = cudaDeviceSynchronize();

    return (err == cudaSuccess) ? MAYO_OK : MAYO_ERR;
}

void mayo2_verify_key_destroy(mayo2_verify_ctx_t *ctx)
{
    cudaFree(ctx->d_P_limbs);
    ctx->d_P_limbs = NULL;
    cudaFree(g_tc_PA);
    g_tc_PA = NULL;
}

static void verify_PS_dispatch(const mayo2_verify_ctx_t *ctx, mayo2_verify_workspace_t *ws, int blocks_ps)
{
    const int mode = mayo_tc_ps_mode();
    const unsigned tc_blocks = (unsigned)TC_BLOCKS, tc_threads = TC_WPB * 32;
    if (mode == 1) {
        verify_calc_PS_tc<<<tc_blocks, tc_threads>>>(g_tc_PA, ws->d_s, ws->d_PS);
        return;
    }
    verify_calc_PS<<<blocks_ps, THREADS>>>(ctx->d_P_limbs, ws->d_s, ws->d_PS);
    static int checked = 0;
    if (mode == 2 && !checked) {
        checked = 1;
        const size_t words = (size_t)MAYO_P_n * MAYO_P_k * MAYO_P_m_vec_limbs * BATCH;
        uint64_t *d_alt;
        cudaMalloc((void **)&d_alt, words * sizeof(uint64_t));
        cudaMemset(d_alt, 0xA5, words * sizeof(uint64_t));
        verify_calc_PS_tc<<<tc_blocks, tc_threads>>>(g_tc_PA, ws->d_s, d_alt);
        cudaError_t e = cudaDeviceSynchronize();
        uint64_t *h_ref = (uint64_t *)malloc(words * sizeof(uint64_t));
        uint64_t *h_alt = (uint64_t *)malloc(words * sizeof(uint64_t));
        cudaMemcpy(h_ref, ws->d_PS, words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_alt, d_alt, words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        size_t bad = 0, first = words;
        for (size_t i = 0; i < words; i++) {
            if (h_ref[i] != h_alt[i]) {
                if (!bad) {
                    first = i;
                }
                bad++;
            }
        }
        printf("[TC-PS CHECK] %s: %zu of %zu limbs differ", cudaGetErrorString(e), bad, words);
        if (bad) {
            printf(" (first at %zu: ref %016llx tc %016llx)", first, (unsigned long long)h_ref[first],
                   (unsigned long long)h_alt[first]);
        }
        printf("\r\n");
        free(h_ref);
        free(h_alt);
        cudaFree(d_alt);
    }
}

void mayo2_verify_workspace_init(mayo2_verify_workspace_t *ws, size_t mlen)
{
    ws->mlen = mlen;

    cudaMalloc((void **)&ws->d_msg, mlen * BATCH);
    cudaMalloc((void **)&ws->d_sig, MAYO_P_sig_bytes * BATCH);
    cudaMalloc((void **)&ws->d_digest, MAYO_P_digest_bytes * BATCH);
    cudaMalloc((void **)&ws->d_tmp, (MAYO_P_digest_bytes + MAYO_P_salt_bytes) * BATCH);
    cudaMalloc((void **)&ws->d_tenc, MAYO_P_m_bytes * BATCH);
    cudaMalloc((void **)&ws->d_t, MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_s, MAYO_P_k * MAYO_P_n * BATCH);
    cudaMalloc((void **)&ws->d_PS, MAYO_P_n * MAYO_P_k * MAYO_P_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&ws->d_SPS, MAYO_P_k * MAYO_P_k * MAYO_P_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&ws->d_y, MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_zero, MAYO_P_m * BATCH);
    cudaMalloc((void **)&ws->d_results, sizeof(int) * BATCH);
    cudaMallocHost((void **)&ws->h_results, sizeof(int) * BATCH);
}

void mayo2_verify_workspace_destroy(mayo2_verify_workspace_t *ws)
{
    cudaFree(ws->d_msg);
    cudaFree(ws->d_sig);
    cudaFree(ws->d_digest);
    cudaFree(ws->d_tmp);
    cudaFree(ws->d_tenc);
    cudaFree(ws->d_t);
    cudaFree(ws->d_s);
    cudaFree(ws->d_PS);
    cudaFree(ws->d_SPS);
    cudaFree(ws->d_y);
    cudaFree(ws->d_zero);
    cudaFree(ws->d_results);
    cudaFreeHost(ws->h_results);
}

int mayo2_verify_batch(mayo2_verify_ctx_t *ctx, mayo2_verify_workspace_t *ws, const unsigned char *msgs,
                       const unsigned char *sigs, size_t mlen, int *results, float *elapsed_ms_out)
{
    const int tmp_bytes = MAYO_P_digest_bytes + MAYO_P_salt_bytes;

    int blocks_tmp = (BATCH * tmp_bytes + THREADS - 1) / THREADS;

    int sig_packed_bytes = (MAYO_P_k * MAYO_P_n + 1) / 2;
    int blocks_per_sig = (sig_packed_bytes + THREADS - 1) / THREADS;

    int ps_outputs = MAYO_P_n * MAYO_P_k * MAYO_P_m_vec_limbs;
    int sps_outputs = MAYO_P_k * MAYO_P_k * MAYO_P_m_vec_limbs;
    int blocks_ps = (BATCH * ps_outputs + THREADS - 1) / THREADS;
    int blocks_sps = (BATCH * sps_outputs + THREADS - 1) / THREADS;

    if (!mayo_bench_include_copies()) {
        cudaMemcpy(ws->d_msg, msgs, mlen * BATCH, cudaMemcpyHostToDevice);
        cudaMemcpy(ws->d_sig, sigs, MAYO_P_sig_bytes * BATCH, cudaMemcpyHostToDevice);
    }

    cudaMemset(ws->d_zero, 0, MAYO_P_m * BATCH);
    cudaMemset(ws->d_results, 0, sizeof(int) * BATCH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (mayo_bench_include_copies()) {
        cudaMemcpy(ws->d_msg, msgs, mlen * BATCH, cudaMemcpyHostToDevice);
        cudaMemcpy(ws->d_sig, sigs, MAYO_P_sig_bytes * BATCH, cudaMemcpyHostToDevice);
    }

    // digest = shake256(msg)
    shake256<<<BATCH, 25>>>(ws->d_digest, MAYO_P_digest_bytes, ws->d_msg, mlen, 0);

    // tmp = digest || salt
    verify_build_tmp<<<blocks_tmp, THREADS>>>(ws->d_digest, ws->d_sig, ws->d_tmp);

    // tEnc = shake256(tmp, m_bytes)
    shake256<<<BATCH, 25>>>(ws->d_tenc, MAYO_P_m_bytes, ws->d_tmp, tmp_bytes, tmp_bytes);

    // t = decode(tEnc)
    decode<<<blocks_per_sig * BATCH, THREADS>>>(ws->d_tenc, ws->d_t, MAYO_P_m, blocks_per_sig, MAYO_P_m_bytes,
                                                MAYO_P_m);

    // s = decode(sig[0 : k*n/2 bytes])
    decode<<<blocks_per_sig * BATCH, THREADS>>>(ws->d_sig, ws->d_s, MAYO_P_k * MAYO_P_n, blocks_per_sig,
                                                MAYO_P_sig_bytes, MAYO_P_k * MAYO_P_n);

    // PS = P * s^t
    verify_PS_dispatch(ctx, ws, blocks_ps);

    // SPS = s^t * PS
    verify_calc_SPS<<<blocks_sps, THREADS>>>(ws->d_s, ws->d_PS, ws->d_SPS);

    // y = compute_rhs(SPS, 0)
    compute_rhs_finegrain<<<BATCH, MAYO_P_m_vec_limbs>>>(ws->d_SPS, ws->d_zero, ws->d_y);

    // results[batch] = (y != t)
    verify_compare<<<BATCH, MAYO_P_m>>>(ws->d_y, ws->d_t, ws->d_results);

    if (mayo_bench_include_copies()) {
        cudaMemcpy(ws->h_results, ws->d_results, sizeof(int) * BATCH, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return MAYO_ERR;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    if (elapsed_ms_out != NULL) {
        *elapsed_ms_out = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (!mayo_bench_include_copies()) {
        cudaMemcpy(ws->h_results, ws->d_results, sizeof(int) * BATCH, cudaMemcpyDeviceToHost);
    }

    if (results != NULL) {
        memcpy(results, ws->h_results, sizeof(int) * BATCH);
    }

    return MAYO_OK;
}
