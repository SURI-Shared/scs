#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#include "glbopts.h"
#include "linsys.h"
#include "scs_matrix.h"
#include "cholmod.h"

#ifdef __cplusplus
extern "C" {
#endif
#define CHOLMOD_STYPE_UNSYMMETRIC 0;
#define CHOLMOD_STYPE_SYMMETRIC_UPPER 1;
#define CHOLDMOD_STYPE_SYMMETRIC_LOWER -1;
cholmod_sparse* interpret_ScsMatrix_as_cholmod_sparse(ScsMatrix*,int,int);
ScsMatrix* interpret_cholmod_sparse_as_ScsMatrix(cholmod_sparse*);
void populate_cholmod_dense_with_column_vector(scs_float*,size_t,cholmod_dense*);
struct SCS_LIN_SYS_WORK {
    size_t nrows;
    size_t nprimals;
    size_t nduals;

    cholmod_common* internal;
    cholmod_sparse *kkt;//matrix to factorize
    cholmod_factor* L;//LDL^T factorization of kkt
    cholmod_dense* solution;//storage for the solutions to the linear systems; automatically resized by solver
    cholmod_dense* Y;//workspace for linear solver; automatically resized by solver
    cholmod_dense* E;//workspace for linear solver; automatically resized by solver

    scs_int *diag_r_idxs;
    scs_float *diag_p;

    scs_int factorizations;
};

#ifdef __cplusplus
}
#endif
#endif
