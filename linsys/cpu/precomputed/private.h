#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "csparse.h"
#include "external/amd/amd.h"
#include "external/qdldl/qdldl.h"
#include "glbopts.h"
#include "linsys.h"
#include "scs_matrix.h"

struct SCS_LIN_SYS_WORK {
  scs_int m, n;       /* linear system dimensions */
  ScsMatrix *kkt, *L; /* KKT, and factorization matrix L resp. */
  scs_float *Dinv;    /* inverse diagonal matrix of factorization */
  scs_int *perm;      /* permutation of KKT matrix for factorization */
  scs_float *bp;      /* workspace memory for solves */
  scs_int *diag_r_idxs;
  scs_int factorizations;
  /* ldl factorization workspace */
  scs_float *D, *fwork;
  scs_int *etree, *iwork, *Lnz, *bwork;
  scs_float *diag_p;
  /* storage for precomputed factorizations*/
  scs_int num_precomputed_scales;
  scs_float* precomputed_scales;//length is num_precomputed_scales. Must be sorted in increasing order!
  ScsMatrix** precomputed_Ls;//length is num_precomputed_scales. Order corresponds to precomputed_scales
  scs_float** precomputed_Ds;//length is num_precomputed_scales. Order corresponds to precomputed_scales
  scs_float** precomputed_Dinvs;//length is num_precomputed_scales. Order corresponds to precomputed_scales
  
};

#ifdef __cplusplus
}
#endif
#endif
