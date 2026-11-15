#ifndef R3_SHIM_H
#define R3_SHIM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int r3_expand_sk(int variant, const unsigned char *csk, uint64_t *P1L, int n_limbs, unsigned char *O,
                 int n_O);
int r3_expand_pk(int variant, const unsigned char *cpk, uint64_t *P, int n_limbs);
#ifdef __cplusplus
}
#endif
#endif
