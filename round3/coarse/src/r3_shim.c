#include <stdint.h>
#include <string.h>
#include <mayo.h>
#include "r3_shim.h"

static const mayo_params_t *r3_sel(int v)
{
    switch (v) {
        case 1:
            return &MAYO_1;
        case 2:
            return &MAYO_2;
        case 3:
            return &MAYO_3;
        default:
            return &MAYO_5;
    }
}

int r3_expand_sk(int variant, const unsigned char *csk, uint64_t *P1L, int n_limbs, unsigned char *O, int n_O)
{
    static sk_t esk;
    memset(&esk, 0, sizeof(esk));
    int r = mayo_expand_sk(r3_sel(variant), csk, &esk);
    memcpy(P1L, esk.p, (size_t)n_limbs * sizeof(uint64_t));
    memcpy(O, esk.O, (size_t)n_O);
    return r;
}

int r3_expand_pk(int variant, const unsigned char *cpk, uint64_t *P, int n_limbs)
{
    static uint64_t epk[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX + MAYO_MVEC_SLACK];
    memset(epk, 0, sizeof(epk));
    int r = mayo_expand_pk(r3_sel(variant), cpk, epk);
    memcpy(P, epk, (size_t)n_limbs * sizeof(uint64_t));
    return r;
}
