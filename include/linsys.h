#ifndef LINSYS_H_GUARD
#define LINSYS_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "glbopts.h"
#include "scs.h"

/* This is the API that any new linear system solver must implement */

/* Struct containing linear system workspace. Implemented by linear solver. */
/* This typedef is in scs.h */
/* typedef struct SCS_LIN_SYS_WORK ScsLinSysWork; */

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
                                     const scs_float *diag_r);

/**
 * Frees `ScsLinSysWork` structure and associated allocated memory.
 *
 *  @param  w    Linear system private workspace.
 */
void scs_free_lin_sys_work(ScsLinSysWork *w);

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
 */
scs_int scs_solve_lin_sys(ScsLinSysWork *w, scs_float *b, const scs_float *s,
                          scs_float tol);
/**
 *  Update the linsys workspace when `R` is changed. For example, a
 *  direct method for solving the linear system might need to update the
 *  factorization of the matrix.
 *
 *  @param  w             Linear system private workspace.
 *  @param  new_diag_r    Updated `diag_r`, diagonal entries of R.
 *
 */
void scs_update_lin_sys_diag_r(ScsLinSysWork *w, const scs_float *new_diag_r);

/**
 * Name of the linear solver.
 *
 * @return name of method.
 */
const char *scs_get_lin_sys_method(void);

//These methods are only to be implemented by solvers capable of precomputing much of the linear system work for known scales

/**
 * Initialize `ScsLinSysWork` structure and perform any necessary preprocessing.
 *
 *  @param  A          `A` data matrix, `m x n`.
 *  @param  P          `P` data matrix, `n x n`.
 *  @param  rho_x      Primal constraint scaling factor.
 *  @param  cone_work  Cone workspace
 *  @param  scales     dual scaling factors to precompute for
 *  @param  num_precomputed_scales length of the scales array
 *  @param  initial_scale index in scales of the dual scaling factor to start with
 *  @return            Linear system solver workspace.
 *
 */
ScsLinSysWork *scs_init_precomputed_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                     const scs_float rho_x,const ScsConeWork* cone_work,scs_float* scales,scs_int num_precomputed_scales, scs_int initial_scale);

/**
 *  Update the linsys workspace when `R` is changed to a value we have precomputed things for. For example, a
 *  direct method for solving the linear system might need to update the
 *  factorization of the matrix to the precomputed factorization for this value of R
 *
 *  @param  w             Linear system private workspace.
 *  @param  new_diag_r    Updated `diag_r`, diagonal entries of R.
 *  @param  new_scales_index The index of the scale factor that produced new_diag_r
 *
 */
void scs_precomputed_update_lin_sys_diag_r(ScsLinSysWork *w, const scs_float *new_diag_r, const scs_int new_scales_index);

/**
 *  Select the scale from the allowed set that is closest to the requested scale
 * 
 *  @param  requested_scale Scale factor SCS wants to use
 *  @param  w               Linear system private workspace
 *  @param  new_scale       scale from the allowed set that is closest to the requested scale
 * 
 *  @return integer index of the new_scale
 */
scs_int scs_get_closest_allowed_scale(scs_float requested_scale, const ScsLinSysWork* w, scs_float* new_scale);
#ifdef __cplusplus
}
#endif

#endif
