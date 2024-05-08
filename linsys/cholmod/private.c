#include "private.h"
#include "cholmod.h"

cholmod_sparse* interpret_ScsMatrix_as_cholmod_sparse(ScsMatrix* in,int stype,int sorted){
    cholmod_sparse* out=(cholmod_sparse*) scs_malloc(sizeof(cholmod_sparse));
    out->nrow=in->m;
    out->ncol=in->n;
    out->nzmax=in->p[in->n];
    out->p=in->p;
    out->i=in->i;
    out->nz=NULL;
    out->packed=1;
    out->x=in->x;
    out->z=NULL;
    out->stype=stype;
    out->itype=CHOLMOD_INT;
    out->xtype=CHOLMOD_REAL;
    out->dtype=CHOLMOD_DOUBLE;
    out->sorted=sorted;

    return out;

}

ScsMatrix* interpret_cholmod_sparse_as_ScsMatrix(cholmod_sparse* in){
    ScsMatrix* out=(ScsMatrix*) scs_malloc(sizeof(ScsMatrix));
    out->m=in->nrow;
    out->n=in->ncol;
    out->p=in->p;
    out->i=in->i;
    out->x=in->x;
    return out;
}

void populate_cholmod_dense_with_column_vector(scs_float* vector,size_t nelements,cholmod_dense* out){
    out->nrow=nelements;
    out->ncol=1;
    out->nzmax=nelements;
    out->d=nelements;
    out->x=vector;
    out->z=NULL;
    out->xtype=CHOLMOD_REAL;
    out->dtype=CHOLMOD_DOUBLE;
}
/**
 * Initialize `ScsLinSysWork` structure and perform any necessary preprocessing.
 *
 *  @param  A          `A` data matrix, `m x n`.
 *  @param  P          `P` data matrix, `n x n`.
 *  @param  diag_r     `R > 0` diagonal entries of length `m + n`.
 *  @return            Linear system solver workspace.
 *
 */
ScsLinSysWork *scs_init_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                     const scs_float *diag_r){
    ScsLinSysWork* p=(ScsLinSysWork *)scs_calloc(1, sizeof(ScsLinSysWork));
    p->internal=(cholmod_common*)scs_malloc(sizeof(cholmod_common));
    int success=cholmod_start(p->internal);
    if(!success){
        return SCS_NULL;
    }
    p->internal->supernodal=CHOLMOD_SIMPLICIAL;
    // p->internal->print=5;
    // cholmod_print_common("KKT Common:",p->internal);
    p->nprimals=A->n;
    p->nduals=A->m;
    p->nrows=A->m+A->n;
    p->factorizations=0;
    p->diag_p = (scs_float *)scs_calloc(A->n, sizeof(scs_float));
    p->diag_r_idxs = (scs_int *)scs_calloc(p->nrows, sizeof(scs_int));
    p->solution=NULL;
    p->Y=NULL;
    p->E=NULL;

    scs_int use_upper_triangular_part=1;
    ScsMatrix *kkt = SCS(form_kkt)(A, P, p->diag_p, diag_r, p->diag_r_idxs, use_upper_triangular_part);
    if (!kkt) {
        return SCS_NULL;
    }

    p->kkt=interpret_ScsMatrix_as_cholmod_sparse(kkt,use_upper_triangular_part,0);
    scs_free(kkt);//don't need the ScsMatrix struct anymore. The data in the pointers is now owned by p->kkt.
    p->L=cholmod_analyze(p->kkt,p->internal);
    success=cholmod_factorize(p->kkt,p->L,p->internal);
    if(!success){
        return SCS_NULL;
    }
    // cholmod_print_common("KKT Common:",p->internal);
    return p;
}

/**
 * Frees `ScsLinSysWork` structure and associated allocated memory.
 *
 *  @param  w    Linear system private workspace.
 */
void scs_free_lin_sys_work(ScsLinSysWork *p){
    if(p){
        cholmod_free_sparse(&p->kkt,p->internal);
        cholmod_free_factor(&p->L,p->internal);
        if(p->solution){
            cholmod_free_dense(&p->solution,p->internal);
        }
        if(p->Y){
            cholmod_free_dense(&p->Y,p->internal);
        }
        if(p->E){
            cholmod_free_dense(&p->E,p->internal);
        }
        cholmod_free_work(p->internal);

        scs_free(p->diag_p);
        scs_free(p->diag_r_idxs);

        scs_free(p);
    }
}

/**
 * Solves the linear system as required by SCS at each iteration:
 * \f[
 *    \begin{bmatrix}
 *    (R_x + P) & A^\top \\
 *     A   &  -R_y \\
 *    \end{bmatrix} x = b
 *  \f]
 *
 *  for `x`, where `diag(R_x, R_y) = R`. Overwrites `b` with result.
 *
 *  @param  w    Linear system private workspace.
 *  @param  b    Right hand side, contains solution at the end.
 *  @param  s    Contains warm-start (may be NULL).
 *  @param  tol  Tolerance required for the system solve.
 *  @return status != 0 indicates failure.
 *
 * TODO: this copies into the memory b pointed to when passed in. Should be possible to just change where b points (watch out for memory leaks)
 */
scs_int scs_solve_lin_sys(ScsLinSysWork *w, scs_float *b, const scs_float *s,
                          scs_float tol){
    cholmod_dense B;
    populate_cholmod_dense_with_column_vector(b,w->nrows,&B);
    int success=cholmod_solve2(CHOLMOD_A,w->L,&B,NULL,&w->solution,NULL,&w->Y,&w->E,w->internal);
    if(!success){
        return 1;//solve failed
    }
    memcpy(b,w->solution->x,sizeof(scs_float)*w->nrows);
    return 0;
}
/**
 *  Update the linsys workspace when `R` is changed. The sparsity pattern of the KKT matrix is unchanged, so we reuse the prior numerical factorization.
 *
 *  @param  p             Linear system private workspace.
 *  @param  new_diag_r    Updated `diag_r`, diagonal entries of R.
 *
 */
void scs_update_lin_sys_diag_r(ScsLinSysWork *p, const scs_float *new_diag_r){
    scs_int i;
    for (i = 0; i < p->nprimals; ++i) {
        /* top left is R_x + P, bottom right is -R_y */
        ((scs_float *)p->kkt->x)[p->diag_r_idxs[i]] = p->diag_p[i] + new_diag_r[i];
    }
    for (i = p->nprimals; i < p->nrows; ++i) {
        /* top left is R_x + P, bottom right is -R_y */
        ((scs_float *)p->kkt->x)[p->diag_r_idxs[i]] = -new_diag_r[i];
    }

    int success=cholmod_factorize(p->kkt,p->L,p->internal);
    if(!success){
        scs_printf("Error in LDL factorization when updating.\n");
        return;
    }
}

/**
 * Name of the linear solver.
 *
 * @return name of method.
 */
const char *scs_get_lin_sys_method(void){
    return "sparse-direct-cholmod";
}