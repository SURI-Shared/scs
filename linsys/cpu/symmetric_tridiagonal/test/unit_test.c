/* Taken from http://www.jera.com/techinfo/jtns/jtn002.html */
#include <stdio.h>

#include "minunit.h"
#include "problem_utils.h"
#include "scs.h"
#include "private.h"

/* Include Tests */
#include "tridiagonal.h"

int tests_run = 0;

/* decrement tests_run since mu_unit will increment it, so this cancels */
#define _SKIP(problem)                                                         \
  char *problem(void) {                                                        \
    scs_printf("skipped\n");                                                   \
    tests_run--;                                                               \
    return 0;                                                                  \
  }

static const char *all_tests(void) {
  mu_run_test(A_to_tridiag);
  mu_run_test(ATA);
  return 0;
}
int main(void) {
  const char *result = all_tests();
  if (result != 0) {
    scs_printf("%s\n", result);
    scs_printf("TEST FAILED!\n");
  } else {
    scs_printf("ALL TESTS PASSED\n");
  }
  scs_printf("Tests run: %d\n", tests_run);

  return result != 0;
}

scs_float* dense_from_symmetric_tridiagonal(const scs_float* diagonal, const scs_float* subdiagonal, scs_int n){
    //allocate dense storage
    scs_float* dense=(scs_float*) scs_calloc(n*n,sizeof(scs_float));

    //populate dense storage
    for(scs_int i=0;i<n;i++){
        dense[i*n+i]=diagonal[i];
        if(i<n-1){
            dense[i*n+i+1]=subdiagonal[i];
            dense[(i+1)*n+i]=subdiagonal[i];
        }
    }
    return dense;
}

ScsMatrix* sparse_from_symmetric_tridiagonal(const scs_float* diagonal, const scs_float* subdiagonal, scs_int n){
    //allocate storage
    ScsMatrix* sparse=(ScsMatrix*) scs_calloc(1,sizeof(ScsMatrix));
    scs_int nnz=n+2*n-2;
    sparse->x=(scs_float*) scs_calloc(nnz,sizeof(scs_float));
    sparse->m=n;
    sparse->n=n;
    sparse->i=(scs_int*) scs_calloc(nnz,sizeof(scs_int));
    sparse->p=(scs_int*) scs_calloc(n+1,sizeof(scs_int));

    //populate storage
    scs_int data_index=0;
    for(scs_int i=0;i<n;i++){
        sparse->p[i]=data_index;
        if(i>0){
            //add super diagonal
            sparse->x[data_index]=subdiagonal[i];
            sparse->i[data_index]=i-1;
            data_index++;
        }
        //add diagonal
        sparse->x[data_index]=diagonal[i];
        sparse->i[data_index]=i;
        data_index++;
        if(i<n-1){
            //add sub diagonal
            sparse->x[data_index]=subdiagonal[i];
            sparse->i[data_index]=i-1;
            data_index++;
        }
    }
    sparse->p[n]=data_index;
    return sparse;
}

int check_diag_and_subdiag(const scs_float* expected_diagonal, const scs_float* expected_subdiagonal, scs_int n, const scs_float* obtained_diagonal, const scs_float* obtained_subdiagonal){
    int ok=1;
    for(scs_int i=0;i<n;i++){
        if(expected_diagonal[i]!=obtained_diagonal[i]){
            scs_printf("Diagonal[%i]=%f, should be %f\n",i,obtained_diagonal[i],expected_diagonal[i]);
            ok=0;
        }
    }
    for(scs_int i=0;i<n-1;i++){
        if(expected_subdiagonal[i]!=obtained_subdiagonal[i]){
            scs_printf("Subiagonal[%i]=%f, should be %f\n",i,obtained_subdiagonal[i],expected_subdiagonal[i]);
            ok=0;
        }
    }
    return ok;
}

int verify_get_symmetric_diagonal_and_subdiagonal(const scs_float* expected_diagonal, const scs_float* expected_subdiagonal, scs_int n){
    ScsMatrix* sparse=sparse_from_symmetric_tridiagonal(expected_diagonal,expected_subdiagonal,n);

    scs_float* obtained_diagonal=(scs_float*) scs_calloc(n,sizeof(scs_float));
    scs_float* obtained_subdiagonal=(scs_float*) scs_calloc(n-1,sizeof(scs_float));
    get_symmetric_diagonal_and_subdiagonal(sparse,obtained_diagonal,obtained_subdiagonal);

    SCS(cs_spfree)(sparse);

    scs_int ok= check_diag_and_subdiag(expected_diagonal,expected_subdiagonal,n,obtained_diagonal,obtained_subdiagonal);

    scs_free(obtained_diagonal);
    scs_free(obtained_subdiagonal);
    return ok;
}

int verify_compute_symmetric_tridiagonal_ATA(const ScsMatrix* A, const scs_float* expected_ATAdiag, const scs_float* expected_ATAsubdiag){
    scs_int n=A->n;
    scs_float* obtained_diagonal=(scs_float*) scs_calloc(n,sizeof(scs_float));
    scs_float* obtained_subdiagonal=(scs_float*) scs_calloc(n-1,sizeof(scs_float));

    compute_symmetric_tridiagonal_ATA(A,obtained_diagonal,obtained_subdiagonal);

    scs_int ok=check_diag_and_subdiag(expected_ATAdiag,expected_ATAsubdiag,n,obtained_diagonal,obtained_subdiagonal);

    scs_free(obtained_diagonal);
    scs_free(obtained_subdiagonal);
    return ok;

}