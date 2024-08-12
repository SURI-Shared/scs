#ifndef SYMTRIDIAG_UNITTEST_H_GUARD
#define SYMTRIDIAG_UNITTEST_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "scs.h"

int verify_get_symmetric_diagonal_and_subdiagonal(const scs_float* expected_diagonal, const scs_float* expected_subdiagonal, scs_int n);
int verify_compute_symmetric_tridiagonal_ATA(const ScsMatrix* A, const scs_float* expected_ATAdiag, const scs_float* expected_ATAsubdiag);
ScsMatrix* sparse_from_symmetric_tridiagonal(const scs_float* diagonal, const scs_float* subdiagonal, scs_int n);

#ifdef __cplusplus
}
#endif
#endif