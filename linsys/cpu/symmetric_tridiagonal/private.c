#include "private.h"

const char *scs_get_lin_sys_method() {
  return "symmetric-tridiagonal";
}

void scs_free_lin_sys_work(ScsLinSysWork *p) {
  if (p) {
    scs_free(p->D);
    scs_free(p->Lsubdiag);
    scs_free(p->ATAdiag);
    scs_free(p->ATAsubdiag);
    if(p->Pdiag){//if no P, then these are both NULL
      scs_free(p->Pdiag);
      scs_free(p->Psubdiag);
    }
    scs_free(p);
  }
}

/* can call only once between calls to form_symmetric_tridiagonal*/
static scs_int ldl_factor(ScsLinSysWork *p) {
#if VERBOSITY > 0
  scs_printf("numeric factorization\n");
#endif
  scs_int factor_status=LAPACKE(pttrf)(p->n,p->D,p->Lsubdiag);
#if VERBOSITY > 0
  scs_printf("finished numeric factorization.\n");
#endif
  if (factor_status < 0) {
    scs_printf("Error in LDL factorization. ");
    if(factor_status==-1){
      scs_printf("%i is an illegal value for the size of a SPD tridiagonal matrix\n",p->n);
    }else if(factor_status==-2){
      scs_printf("Array containing diagonal elements of SPD tridiagonal matrix had an illegal value\n");
    }else if(factor_status==-3){
      scs_printf("Array containing subdiagonal elements of SPD tridiagonal matrix had an illegal value\n");
    }else{
      scs_printf("Unrecognized error code %i from LAPACKE_dpttrf\n",factor_status);
    }
  }else if(factor_status>0){
    scs_printf("Leading principal minor of order %i is not positive\n",factor_status);
  }
  p->factorizations++;
  return factor_status;
}

void form_symmetric_tridiagonal(ScsLinSysWork* p){
  //diagonal is R_x+Pdiag+R_y^{-1}ATAdiag
  scs_float r_y=p->diag_r[p->n];//by assumption all of the diagonal elements of R_y, which are diag_r[n:n+m], are identical
  for(scs_int i=0;i<p->n;i++){
    p->D[i]=p->diag_r[i]+p->ATAdiag[i]/r_y;
    if(p->Pdiag){//may have 0 P, in which case p->Pdiag is NULL
      p->D[i]+=p->Pdiag[i];
    }
  }

  //subdiagonal is Psubdiag+ATAsubdiag/r_y
  for(scs_int i=0;i<p->n-1;i++){
    p->Lsubdiag[i]=p->ATAsubdiag[i]/r_y;
    if(p->Psubdiag){//may have 0 P, in which case p->Psubdiag is NULL
      p->Lsubdiag[i]+=p->Psubdiag[i];
    }
  }
}

void scs_update_lin_sys_diag_r(ScsLinSysWork *p, const scs_float *diag_r) {
  scs_int i, ldl_status;
  p->diag_r=diag_r;
  //populate D and Lsubdiag with the tridiagonal matrix we need to factor
  form_symmetric_tridiagonal(p);
  //populate D and Lsubdiag with the LDL' factorization
  ldl_status=ldl_factor(p);
  if (ldl_status != 0) {
    scs_printf("Error in LDL factorization when updating.\n");
    /* TODO: this is broken somehow */
    /* SCS(free_lin_sys_work)(p); */
    return;
  }
}
//assumes diagonal and subdiagonal are already allocated and initialized to 0s
static void get_symmetric_diagonal_and_subdiagonal(const ScsMatrix *M, scs_float* diagonal, scs_float* subdiagonal){
  for (scs_int j = 0; j < M->n; j++) { /* cols */
      for (scs_int h = M->p[j]; h < M->p[j + 1]; h++) {
        scs_int i = M->i[h]; /* row */
        if (i > j) { /* only upper triangular part is needed, and for example the quadratic cost will only have the upper part specified anyway */
          break;
        }
        if (i == j) {
          /* M has diagonal element */
          diagonal[j] = M->x[h];
        }
        if (i+1==j) {
          /* M has subdiagonal element (technically, this is a superdiagonal element but M is symmetric)*/
          subdiagonal[i]=M->x[h];
        }
      }
    }
}

ScsLinSysWork *scs_init_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                     const scs_float *diag_r) {
  ScsLinSysWork *p = (ScsLinSysWork *)scs_calloc(1, sizeof(ScsLinSysWork));
  scs_int n_plus_m = A->n + A->m, ldl_status, ldl_prepare_status;
  p->m = A->m;
  p->n = A->n;
  p->D = (scs_float *)scs_calloc(A->n,sizeof(scs_float));
  p->Lsubdiag = (scs_float*)scs_calloc(A->n-1,sizeof(scs_float));
  p->ATAdiag = (scs_float*)scs_calloc(A->n,sizeof(scs_float));
  p->ATAsubdiag = (scs_float*)scs_calloc(A->n-1,sizeof(scs_float));
  if(P){
    p->Pdiag=(scs_float*)scs_calloc(A->n,sizeof(scs_float));
    p->Psubdiag=(scs_float*)scs_calloc(A->n-1,sizeof(scs_float));
  }else{
    p->Pdiag=NULL;
    p->Psubdiag=NULL;
  }
  p->factorizations = 0;
  p->diag_r = diag_r;

  //populate diagonal and subdiagonal of P
  if (P) {
    get_symmetric_diagonal_and_subdiagonal(P,p->Pdiag,p->Psubdiag);
  }

  //compute ATA
  compute_symmetric_tridiagonal_ATA(A,p->ATAdiag,p->ATAsubdiag);

  //form the matrix we need to factorize
  form_symmetric_tridiagonal(p);

  //factorize
  ldl_status = ldl_factor(p);
  if (ldl_status != 0) {
    scs_printf("Error in LDL initial factorization.\n");
    /* TODO: this is broken somehow */
    /* SCS(free_lin_sys_work)(p); */
    return SCS_NULL;
  }
  return p;
}

scs_int scs_solve_lin_sys(ScsLinSysWork *p, scs_float *b, const scs_float *s,
                          scs_float tol) {
  /* returns solution to linear system */
  /* Ax = b with solution stored in b */

  //solve for x
  LAPACKE(pttrs)(LAPACK_COL_MAJOR,p->n,1,p->D,p->Lsubdiag,xrhs,p->n);

  //solve for y
  return 0;
}