#include "unit_test_functions.h"
#include "problem_utils.h"
#include "csparse.h"

static const char* A_to_tridiag(void){
    int ok;
    ran_start(100);
    scs_int trials=20;
    scs_int n=20;
    scs_float* expected_d=(scs_float*) scs_calloc(n,sizeof(scs_float));
    scs_float* expected_subd=(scs_float*) scs_calloc(n-1,sizeof(scs_float));

    //generate random symmetric tridiagonal matrices
    for(scs_int i=0;i<trials;i++){
        for(scs_int j=0;j<n;j++){
            expected_d[j]=rand_scs_float();
        }
        for(scs_int j=0;j<n-1;j++){
            expected_subd[j]=rand_scs_float();
        }
        ok=verify_get_symmetric_diagonal_and_subdiagonal(expected_d,expected_subd,n);
        if(!ok){
            scs_free(expected_d);
            scs_free(expected_subd);
            return "INCORRECT_DIAGONAL_OR_SUBDIAGONAL";
        }
    }
    scs_free(expected_d);
    scs_free(expected_subd);
    return 0;
}

static const char* ATA_diagonal(void){
    int ok;
    ran_start(100);
    scs_int n=20;
    scs_int trials=20;

    //check diagonal A
    scs_float* diagonal=(scs_float*) scs_calloc(n,sizeof(scs_float));
    scs_float* diag_squared=(scs_float*) scs_calloc(n,sizeof(scs_float));
    scs_float* zeros=(scs_float*) scs_calloc(n,sizeof(scs_float));
    for(scs_int i=0;i<trials;i++){
        for(scs_int j=0;j<n;j++){
            diagonal[j]=rand_scs_float();
            diag_squared[j]=diagonal[j]*diagonal[j];
        }
        ScsMatrix* A=sparse_from_symmetric_tridiagonal(diagonal,zeros,n);
        ok=verify_compute_symmetric_tridiagonal_ATA(A,diag_squared,zeros);
        if(!ok){
            scs_free(diagonal);
            scs_free(diag_squared);
            scs_free(zeros);
            return "INCORRECT_WHEN_A_IS_DIAGONAL";
        }else{
            SCS(cs_spfree)(A);
        }
    }

    scs_free(diagonal);
    scs_free(diag_squared);
    scs_free(zeros);

    return 0;

}

static const char* ATA_not_diagonal(void){
    int ok;
    scs_int n=10;

    //check for a particular non-diagonal A sparsity pattern known to produce tridiagonal A'A
    scs_int nnz=22;
    scs_float Ax[]={-1., -1., -1., -1.,  1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.};
    scs_int Arows[]={9, 12, 15, 18,  1, 10, 13,  4, 13, 16,  7, 16, 19,  2, 11, 14,  5,14, 17,  8, 17, 20};
    scs_int Acoloffsets[]={0,  1,  2,  3,  4,  7, 10, 13, 16, 19, 22};
    ScsMatrix* A=(ScsMatrix*) scs_calloc(1,sizeof(ScsMatrix));
    A->x=Ax;
    A->i=Arows;
    A->p=Acoloffsets;
    A->m=21;
    A->n=10;
    scs_float diagonal[]={1,1,1,1,3,3,3,3,3,3};
    scs_float subdiagonal[]={0,0,0,0,-1,-1,0,-1,-1};

    ok=verify_compute_symmetric_tridiagonal_ATA(A,diagonal,subdiagonal);
    if(!ok){
        scs_free(A);
        return "INCORRECT_WHEN_A_IS_DIAGONAL";
    }else{
        scs_free(A);
    }

    return 0;

}