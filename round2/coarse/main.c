// Include C++ header files.
#include <stdio.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/inputs.cuh"
#include "include/parameters.cuh"

int main()
{
    run_mayo_sign();

    // CUDA signature verification check (GPU vs CPU reference).
    run_mayo_verify();

    // // Print out result

    return 0;
}
