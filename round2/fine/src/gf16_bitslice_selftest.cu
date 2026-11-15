// CUDA self-test for the bit-sliced GF(16) prototype

#include <cuda_runtime.h>
#include <stdio.h>

#include "gf16_bitslice.cuh"
#include "arithmetic.cuh" // reference gf16_vec_mul / gf16_mul_scalar

__device__ __forceinline__ uint64_t bs_rng_next(uint64_t &s)
{
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__global__ void gf16_bitslice_selftest_kernel(unsigned long long *d_errors)
{
    uint64_t s = 1 + blockIdx.x * 4099ULL + threadIdx.x * 0x9E3779B97F4A7C15ULL;

    unsigned long long local_err = 0;

    // scalar multiply cross-check: all 256 pairs
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            if (gf16_bs_mul4((uint8_t)i, (uint8_t)j) != gf16_mul_scalar((uint8_t)i, (uint8_t)j)) {
                local_err++;
            }
        }
    }

    for (int it = 0; it < 128; it++) {
        uint64_t pack[4];
        for (int l = 0; l < 4; l++) {
            pack[l] = bs_rng_next(s);
        }
        uint8_t a = (uint8_t)(bs_rng_next(s) & 0xF);

        // reference: nibble-sliced gf16_vec_mul per limb
        uint64_t ref[4];
        for (int l = 0; l < 4; l++) {
            ref[l] = gf16_vec_mul(pack[l], a);
        }

        // bitsliced path: pack -> planes -> mul -> pack
        uint64_t got[4];
        gf16_bitslice_vec_mul_packed(pack, a, got);

        for (int l = 0; l < 4; l++) {
            if (ref[l] != got[l]) {
                local_err++;
            }
        }

        // pack -> planes -> pack identity
        uint64_t planes[4], back[4];
        gf16_bitslice_pack_to_planes(pack, planes);
        gf16_bitslice_planes_to_pack(planes, back);
        for (int l = 0; l < 4; l++) {
            if (back[l] != pack[l]) {
                local_err++;
            }
        }

        // planes -> pack -> planes identity
        uint64_t rplanes[4], rpack[4], rplanes2[4];
        for (int j = 0; j < 4; j++) {
            rplanes[j] = bs_rng_next(s);
        }
        gf16_bitslice_planes_to_pack(rplanes, rpack);
        gf16_bitslice_pack_to_planes(rpack, rplanes2);
        for (int j = 0; j < 4; j++) {
            if (rplanes2[j] != rplanes[j]) {
                local_err++;
            }
        }
    }

    if (local_err != 0) {
        atomicAdd(d_errors, local_err);
    }
}

int run_gf16_bitslice_selftest(void)
{
    unsigned long long *d_errors = NULL;
    unsigned long long h_errors = 0;

    cudaError_t err = cudaMalloc(&d_errors, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        printf("gf16 bitslice selftest: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemset(d_errors, 0, sizeof(unsigned long long));

    gf16_bitslice_selftest_kernel<<<32, 256>>>(d_errors);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("gf16 bitslice selftest: kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_errors);
        return 1;
    }

    cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_errors);

    const unsigned long long total_checks = 256 + 128ULL * 12ULL * 32ULL * 256ULL;

    printf("GF16 bit-sliced self-test (device): %s\n", h_errors == 0 ? "PASS" : "FAIL");
    printf("  %llu checks, %llu errors\n", total_checks, h_errors);

    return h_errors == 0 ? 0 : 1;
}
