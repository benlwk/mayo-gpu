// SPDX-License-Identifier: Apache-2.0

#ifndef FIPS202_H
#define FIPS202_H

#include <stddef.h>

__global__ void shake128(unsigned char *output, size_t outputByteLen,
                         const unsigned char *input, size_t inputByteLen);
__host__ __device__ void shake256_device(unsigned char *output,
                                         size_t outputByteLen,
                                         const unsigned char *input,
                                         size_t inputByteLen);
__global__ void shake256(unsigned char *output, size_t outputByteLen,
                         const unsigned char *input, size_t inputByteLen);

#endif
