#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "csparse.h"
#include "glbopts.h"
#include "linsys.h"
#include "scs_matrix.h"
#include <lapacke.h>

/*Linear system solver for the special case where:
1. A' A is tridiagonal
2. P is symmetric with no non-zero elements outside the tridiagonal
3. R_y is of the form r_y*I

3 holds for SCS v3.2.6 provided there are either all zero cones or no zero cones,
since R_y is computed as 1/scale*I for non-zero cones and 1/(1000*scale)*I for zero cones.

At each iteration, SCS wants to solve the linear system 

[(R_x + P)  A'; A -R_y] [x; y] = [z_x; z_y]

Which can be rearranged as:
(1): (R_x + P + A'A/r_y)x = z_x + A'z_y/r_y
(2): y = (Ax-z_y)/r_y

R_x and R_y are diagonal, so if A'A is tridiagonal and P has no non-zero elements outside the tridiagonal (1) can be solved using the tridiagonal matrix algorithm
which for n variables requires O(n) operations.

In particular, we can cheaply compute an LDL' factorization using LAPACK's dpttrf where D is diagonal and L is unit diagonal plus elements in the sub diagonal.
The factorization is recomputed when R changes.
*/

struct SCS_LIN_SYS_WORK {
  scs_int m, n;       /* linear system dimensions */
  scs_float *D; /*after factorization, the n diagonal elements of D in the LDL' factorization*/
  scs_float *Lsubdiag; /*after factorization, the n-1 elements of the subdiagonal of L in the LDL' factorization*/
  scs_int factorizations;
  scs_float *ATAdiag; /*the n elements of the diagonal of A'A*/
  scs_float *ATAsubdiag; /*the n-1 elements of the subdiagonal of A'A*/
  scs_float *Pdiag; /*the n elements of the diagonal of P (NULL if no P)*/
  scs_float *Psubdiag; /*the n-1 elements of the subdiagonal of P (NULL if no P)*/
  scs_float* scaled_zy_space; /*workspace to temporarily store the m elements of z_y/r_y*/
  const scs_float *diag_r; /*diagonal scaling matrix, does NOT own this memory*/
  const ScsMatrix *A; /*constraint matrix mxn, does NOT own this memory*/
};

#ifndef SFLOAT
#define LAPACKE(x) LAPACKE_d##x
#else
#define LAPACKE(x) LAPACKE_s##x
#endif

#ifdef __cplusplus
}
#endif
#endif
