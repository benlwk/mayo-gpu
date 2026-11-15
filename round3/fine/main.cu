#include <stdio.h>

#include "cuda_kernel.cuh"
#include "inputs.cuh"
#include "parameters.cuh"
#include "mayo.cuh"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);
extern "C" int crypto_sign_open(unsigned char *m, size_t *mlen, const unsigned char *sm, size_t smlen,
                                const unsigned char *pk);
#define VERIFY_ROUNDS_MAX 64

// Types mirrored from src/mayo.cu (GPU verify API)
typedef struct mayo2_verify_ctx {
    uint64_t *d_P_limbs; // P1 || P2 || P3
} mayo2_verify_ctx_t;

typedef struct mayo2_verify_workspace {
    size_t mlen;

    unsigned char *d_msg;    // mlen * BATCH
    unsigned char *d_sig;    // MAYO_P_sig_bytes * BATCH
    unsigned char *d_digest; // MAYO_P_digest_bytes * BATCH
    unsigned char *d_tmp;    // (digest + salt) * BATCH
    unsigned char *d_tenc;   // R3_TENC_STRIDE * BATCH
    unsigned char *d_lambda;
    unsigned char *d_t;    // MAYO_P_m * BATCH
    unsigned char *d_s;    // MAYO_P_k * MAYO_P_n * BATCH
    uint64_t *d_PS;        // n * k * m_vec_limbs * BATCH
    uint64_t *d_SPS;       // k * k * m_vec_limbs * BATCH
    unsigned char *d_y;    // m * BATCH
    unsigned char *d_zero; // m * BATCH
    int *d_results;        // BATCH
    int *h_results;
} mayo2_verify_workspace_t;

// GPU verify API implemented in src/mayo.cu
extern int mayo2_verify_key_init(mayo2_verify_ctx_t *ctx, const unsigned char *cpk);
extern void mayo2_verify_key_destroy(mayo2_verify_ctx_t *ctx);
extern void mayo2_verify_workspace_init(mayo2_verify_workspace_t *ws, size_t mlen);
extern void mayo2_verify_workspace_destroy(mayo2_verify_workspace_t *ws);
extern int mayo2_verify_batch(mayo2_verify_ctx_t *ctx, mayo2_verify_workspace_t *ws,
                              const unsigned char *msgs, const unsigned char *sigs, size_t mlen, int *results,
                              float *elapsed_ms_out);

extern int run_gf16_bitslice_selftest(void);

static int hex_val(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

static int parse_hex(const char *s, unsigned char *out, int maxlen)
{
    int len = (int)strlen(s);
    if (len % 2 != 0 || len / 2 > maxlen) {
        return -1;
    }
    int n = 0;
    for (int i = 0; i < len; i += 2) {
        int hi = hex_val(s[i]);
        int lo = hex_val(s[i + 1]);
        if (hi < 0 || lo < 0) {
            return -1;
        }
        out[n++] = (unsigned char)((hi << 4) | lo);
    }
    return n;
}

// Batched GPU verify benchmark
static int run_verify_benchmark(int rounds)
{
    const size_t mlen = 32;
    const int sig_bytes = MAYO_P_sig_bytes;

    if (rounds > VERIFY_ROUNDS_MAX) {
        rounds = VERIFY_ROUNDS_MAX;
    }
    if (rounds < 1) {
        rounds = 1;
    }

    if (crypto_sign_keypair(pk, sk) != 0) {
        printf("Verify setup: crypto_sign_keypair FAILED\r\n");
        return 1;
    }

    unsigned char msg[32] = {0xe};
    unsigned char sm[MAYO_P_sig_bytes + 32];
    size_t smlen = 0;

    printf("Verify setup: signing the test message (once) ...\r\n");

    int r = mayo2_sign(sm, &smlen, msg, mlen, sk);
    if (r != MAYO_OK || smlen != (size_t)(MAYO_P_sig_bytes + mlen)) {
        printf("Verify setup: sign FAILED (ret=%d)\r\n", r);
        return 1;
    }

    // batch = BATCH copies of (msg, sig)
    unsigned char *h_msgs = (unsigned char *)calloc(mlen * BATCH, 1);
    unsigned char *h_sigs = (unsigned char *)calloc(sig_bytes * BATCH, 1);
    int *h_results = (int *)calloc(BATCH, sizeof(int));

    const char *dm = getenv("MAYO_DISTINCT_MSGS");
    const int distinct = (dm != NULL && atoi(dm) > 0);
    if (distinct) {
        // BATCH distinct messages, all signed by the GPU signer
        if (mayo2_sign_batch(h_msgs, h_sigs, msg, mlen, sk) != MAYO_OK) {
            printf("Verify setup: batch sign FAILED\r\n");
            return 1;
        }
        int n_check = BATCH < 256 ? BATCH : 256, cpu_ok = 0, differ = 0;
        unsigned char *smc = (unsigned char *)malloc(sig_bytes + mlen);
        unsigned char *opened = (unsigned char *)malloc(sig_bytes + mlen);
        for (int b = 0; b < n_check; b++) {
            int slot = (b == n_check - 1) ? BATCH - 1 : b;
            memcpy(smc, h_sigs + (size_t)slot * sig_bytes, sig_bytes);
            memcpy(smc + sig_bytes, h_msgs + (size_t)slot * mlen, mlen);
            size_t olen = 0;
            if (crypto_sign_open(opened, &olen, smc, sig_bytes + mlen, pk) == 0 && olen == mlen) {
                cpu_ok++;
            }
        }
        for (int b = 1; b < BATCH; b++) {
            if (memcmp(h_sigs + (size_t)b * sig_bytes, h_sigs, sig_bytes) != 0) {
                differ++;
            }
        }
        free(smc);
        free(opened);
        printf("CPU verify of batch signatures: %d/%d OK, %d/%d differ from slot 0\r\n", cpu_ok, n_check,
               differ, BATCH - 1);
    } else {
        for (int b = 0; b < BATCH; b++) {
            memcpy(h_msgs + b * mlen, msg, mlen);
            memcpy(h_sigs + b * sig_bytes, sm, sig_bytes);
        }

        // also exercise per-slot DISTINCT inputs
        {
            int n_distinct = BATCH < 8 ? BATCH : 8;
            for (int slot = 1; slot < n_distinct; slot++) {
                unsigned char dmsg[32];
                memset(dmsg, 0, sizeof(dmsg));
                dmsg[0] = 0x0e;
                dmsg[1] = (unsigned char)slot;
                size_t dlen = 0;
                int dr = mayo2_sign(sm, &dlen, dmsg, mlen, sk);
                if (dr == MAYO_OK && dlen == (size_t)(MAYO_P_sig_bytes + mlen)) {
                    memcpy(h_msgs + slot * mlen, dmsg, mlen);
                    memcpy(h_sigs + slot * sig_bytes, sm, sig_bytes);
                }
            }
        }
    }

    mayo2_verify_ctx_t vctx;
    mayo2_verify_workspace_t vws;

    if (mayo2_verify_key_init(&vctx, pk) != MAYO_OK) {
        printf("Verify setup: key expansion FAILED\r\n");
        return 1;
    }
    mayo2_verify_workspace_init(&vws, mlen);

    printf("=== MAYO-%d GPU verify benchmark "
           "(BATCH=%d, rounds=%d, mlen=%zu) ===\r\n",
           MAYO_VARIANT, BATCH, rounds, mlen);

    float times_ms[VERIFY_ROUNDS_MAX];
    float tp_rounds[VERIFY_ROUNDS_MAX];

    for (int rd = 0; rd < rounds; rd++) {
        float ms = 0.0f;
        int vret = mayo2_verify_batch(&vctx, &vws, h_msgs, h_sigs, mlen, h_results, &ms);
        if (vret != MAYO_OK) {
            printf("Verify round %d: FAILED (ret=%d)\r\n", rd + 1, vret);
            return 1;
        }
        times_ms[rd] = ms;
        tp_rounds[rd] = BATCH / (ms / 1000.0f);
        printf("Verify round %d: %10.6f ms  ->  %12.2f verify/s\r\n", rd + 1, ms, tp_rounds[rd]);
    }

    float sum_ms = 0.0f, sum_tp = 0.0f;
    for (int rd = 0; rd < rounds; rd++) {
        sum_ms += times_ms[rd];
        sum_tp += tp_rounds[rd];
    }
    float mean_ms = sum_ms / rounds;
    float mean_tp = sum_tp / rounds;

    float sq_ms = 0.0f, sq_tp = 0.0f;
    for (int rd = 0; rd < rounds; rd++) {
        sq_ms += (times_ms[rd] - mean_ms) * (times_ms[rd] - mean_ms);
        sq_tp += (tp_rounds[rd] - mean_tp) * (tp_rounds[rd] - mean_tp);
    }
    float std_ms = (rounds > 1) ? sqrtf(sq_ms / (rounds - 1)) : 0.0f;
    float std_tp = (rounds > 1) ? sqrtf(sq_tp / (rounds - 1)) : 0.0f;

    printf("Verify mean time      : %.6f ms   (std %.6f ms)\r\n", mean_ms, std_ms);
    printf("Verify throughput     : %.2f verify/s (std %.2f)\r\n", mean_tp, std_tp);

    int all_valid = 1;
    for (int b = 0; b < BATCH; b++) {
        if (h_results[b] != 0) {
            all_valid = 0;
            break;
        }
    }
    printf("All %d signatures accepted: %s\r\n", BATCH, all_valid ? "PASS" : "FAIL");

    // negative test
    h_sigs[10] ^= 0x01;
    mayo2_verify_batch(&vctx, &vws, h_msgs, h_sigs, mlen, h_results, NULL);
    int bad_rejected = (h_results[0] == 1);
    for (int b = 1; b < BATCH; b++) {
        if (h_results[b] != 0) {
            bad_rejected = 0;
            break;
        }
    }
    h_sigs[10] ^= 0x01;

    printf("Corrupted signature rejected, others accepted (slot 0): %s\r\n", bad_rejected ? "PASS" : "FAIL");

    mayo2_verify_workspace_destroy(&vws);
    mayo2_verify_key_destroy(&vctx);

    free(h_msgs);
    free(h_sigs);
    free(h_results);

    return (all_valid && bad_rejected) ? 0 : 1;
}

// Multi-vector GPU verify check
static int run_verify_multi(const char *path)
{
    const size_t mlen = 32;
    const int sig_bytes = MAYO_P_sig_bytes;
    const int cpk_bytes = MAYO_P_cpk_bytes;

    FILE *f = fopen(path, "r");
    if (f == NULL) {
        printf("ERROR: cannot open %s\n", path);
        return 1;
    }

    char cpk_hex[2 * MAYO_P_cpk_bytes + 2];
    if (fgets(cpk_hex, sizeof(cpk_hex), f) == NULL) {
        printf("ERROR: empty vector file %s\n", path);
        fclose(f);
        return 1;
    }
    cpk_hex[strcspn(cpk_hex, "\r\n")] = '\0';

    unsigned char *cpk = (unsigned char *)calloc(cpk_bytes, 1);
    if (parse_hex(cpk_hex, cpk, cpk_bytes) != cpk_bytes) {
        printf("ERROR: bad cpk hex in %s\n", path);
        fclose(f);
        free(cpk);
        return 1;
    }

    unsigned char *h_msgs = (unsigned char *)calloc(mlen * BATCH, 1);
    unsigned char *h_sigs = (unsigned char *)calloc(sig_bytes * BATCH, 1);
    int *h_results = (int *)calloc(BATCH, sizeof(int));

    int n = 0;
    char line[2048];
    while (n < BATCH && fgets(line, sizeof(line), f) != NULL) {
        char msg_hex[128];
        char sig_hex[512];

        if (sscanf(line, "%127s %511s", msg_hex, sig_hex) != 2) {
            continue;
        }

        if (parse_hex(msg_hex, h_msgs + n * mlen, (int)mlen) != (int)mlen) {
            continue;
        }
        if (parse_hex(sig_hex, h_sigs + n * sig_bytes, sig_bytes) != sig_bytes) {
            continue;
        }
        n++;
    }
    fclose(f);

    if (n == 0) {
        printf("ERROR: no vectors parsed from %s\n", path);
        return 1;
    }

    mayo2_verify_ctx_t vctx;
    mayo2_verify_workspace_t vws;

    if (mayo2_verify_key_init(&vctx, cpk) != MAYO_OK) {
        printf("ERROR: verify key expansion FAILED\n");
        return 1;
    }
    mayo2_verify_workspace_init(&vws, mlen);

    float ms = 0.0f;
    int vret = mayo2_verify_batch(&vctx, &vws, h_msgs, h_sigs, mlen, h_results, &ms);
    if (vret != MAYO_OK) {
        printf("ERROR: verify batch FAILED (ret=%d)\n", vret);
        return 1;
    }

    printf("Verify multi: %d vectors, %.6f ms\n", n, ms);
    for (int i = 0; i < n; i++) {
        printf("VRESULT %d %d\n", i, h_results[i]);
    }

    // corrupted copies: flip first byte of each signature
    for (int i = 0; i < n; i++) {
        h_sigs[i * sig_bytes] ^= 0x01;
    }
    mayo2_verify_batch(&vctx, &vws, h_msgs, h_sigs, mlen, h_results, &ms);
    for (int i = 0; i < n; i++) {
        printf("VCORRUPT %d %d\n", i, h_results[i]);
    }
    for (int i = 0; i < n; i++) {
        h_sigs[i * sig_bytes] ^= 0x01;
    }

    mayo2_verify_workspace_destroy(&vws);
    mayo2_verify_key_destroy(&vctx);

    free(cpk);
    free(h_msgs);
    free(h_sigs);
    free(h_results);

    return 0;
}

//wklee, entry point
int main(int argc, char *argv[])
{
    int res = MAYO_OK;
    unsigned char msg[32] = {0xe};

    size_t msglen = 32;

    int do_verify = 0;
    int verify_rounds = 5;
    const char *verify_multi_path = NULL;
    int do_bs_selftest = 0;
    int sk_given = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sk") == 0 && i + 1 < argc) {
            sk_given = 1;
            if (parse_hex(argv[++i], sk, MAYO_P_csk_bytes) != MAYO_P_csk_bytes) {
                printf("ERROR: --sk must be %d bytes of hex\n", MAYO_P_csk_bytes);
                return 2;
            }
        } else if (strcmp(argv[i], "--msg") == 0 && i + 1 < argc) {
            int n = parse_hex(argv[++i], msg, (int)sizeof(msg));
            if (n <= 0) {
                printf("ERROR: --msg must be valid hex (1..%zu bytes)\n", sizeof(msg));
                return 2;
            }
            msglen = (size_t)n;
        } else if (strcmp(argv[i], "--verify") == 0) {
            do_verify = 1;
            if (i + 1 < argc && argv[i + 1][0] >= '1' && argv[i + 1][0] <= '9') {
                verify_rounds = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--verify-multi") == 0 && i + 1 < argc) {
            verify_multi_path = argv[++i];
        } else if (strcmp(argv[i], "--selftest-bs") == 0) {
            do_bs_selftest = 1;
        }
    }

    if (do_bs_selftest) {
        return run_gf16_bitslice_selftest();
    }

    if (verify_multi_path != NULL) {
        return run_verify_multi(verify_multi_path);
    }

    if (do_verify) {
        return run_verify_benchmark(verify_rounds);
    }

    size_t smlen = MAYO_P_sig_bytes + msglen;

    unsigned char *sig = (unsigned char *)calloc(smlen, 1);

    if (!sk_given) {
        crypto_sign_keypair(pk, sk);
    }

    printf("Start sign (%s)\r\n", MAYO_P_name);

    res = mayo2_sign(sig, &smlen, msg, msglen, sk);
    printf("Sign result: %d, smlen: %ld\r\n", res, (long)smlen);
    {
        unsigned char *opened = (unsigned char *)calloc(smlen + 1, 1);
        size_t olen = 0;
        int vr = crypto_sign_open(opened, &olen, sig, smlen, pk);
        int ok = (res == MAYO_OK && vr == 0 && olen == msglen && memcmp(opened, msg, msglen) == 0);
        printf("CPU verify of GPU signature: %s\r\n", ok ? "OK" : "FAIL");
        free(opened);
    }

    printf("Signature:\n");
    for (int i = 0; i < MAYO_P_sig_bytes; i++) {
        printf("%02x, ", sig[i]);
    }
    printf("\nMessage:\n");
    for (int i = 0; i < msglen; i++) {
        printf("%02x, ", sig[MAYO_P_sig_bytes + i]);
    }
    printf("\n");

    free(sig);
    return 0;
}
