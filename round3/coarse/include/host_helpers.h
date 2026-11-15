#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void compute_host_P1_P2_and_L(const unsigned char *cpk, uint64_t *P1, uint64_t *L);

#ifdef __cplusplus
}
#endif
