#include "scs_matrix.h"
#include "linalg.h"
#include "linsys.h"
#include "util.h"
#include <stdio.h>

__global__ void _cuda_accum_by_a(scs_float *y, Scs_int *Ap, Scs_int *Ai, Scs_float *Ax, scs_float *x) {
    scs_int j= blockIdx.x*blockDim.x + threadIdx.x;
    scs_int p = blockIdx.y*blockDim.x + threadIdx.x + Ap[j];
    scs_int i = Ai[p];
    y[i] += Ax[p] * x[j];
  }

void SCS(accum_by_a)(const ScsMatrix *A, const scs_float *x, const scs_data *d, scs_float *y) {
    /*y += A*x
      A in column compressed format
      */
    scs_int n = A->n;
    scs_int *Ap = A->p;
    scs_int rows = Ap[n];
    scs_int *Ai = A->i;
    scs_float *Ax = A->x;
    cudaMalloc(&y_dev,d->m*d->n *sizeof(scs_float));
    _cuda_accum_by_a<<<1, dim3(d->m, d->n)>>>(y_dev, Ap, Ai, x);
    cudaMemcpy(y_dev,y,d->m*d->n*sizeof(scs_float),cudaMemcpyHostToDevice);
  }

  __global__ void _cuda_accum_by_atrans(scs_float *y, Scs_int *Ap, Scs_int *Ai, Scs_float *Ax, scs_float *x) {
    scs_int j= blockIdx.x*blockDim.x + threadIdx.x;
    scs_int p = blockIdx.y*blockDim.x + threadIdx.x + Ap[j];
    scs_int i = Ai[p];
    y[j] += Ax[p] * x[i];
  }

void SCS(accum_by_atrans)(const ScsMatrix *A, const scs_float *x, const scs_data *d,
                          scs_float *y) {
  /* y += A'*x
     A in column compressed format
     parallelizes over columns (rows of A')
   */
    scs_int n = A->n;
    scs_int *Ap = A->p;
    scs_int *Ai = A->i;
    scs_float *Ax = A->x;
    cudaMalloc(&y_dev,d->m*d->n *sizeof(scs_float));
    _cuda_accum_by_atrans<<<1, dim3(d->m, d->n)>>>(y_dev, Ap, Ai, x);
    cudaMemcpy(y_dev,y,d->m*d->n*sizeof(scs_float),cudaMemcpyHostToDevice);
}

__global__ void _cuda_accum_by_p(scs_float *y, Scs_int *Pp, Scs_int *Pi, Scs_float *Px, scs_float *x) {
    scs_int j= blockIdx.x*blockDim.x + threadIdx.x;
    scs_int p = blockIdx.y*blockDim.x + threadIdx.x + Pp[j];
    scs_int i = Pi[p];
    if (i != j) y[i] += Px[p] * x[j];
  }

/* Since P is upper triangular need to be clever here */
void SCS(accum_by_p)(const ScsMatrix *P, const scs_float *x, const scs_data *d, scs_float *y) {
    /* returns y += P x */
    scs_int p, j, i;
    scs_int n = P->n;
    scs_int *Pp = P->p;
    scs_int *Pi = P->i;
    scs_float *Px = P->x;
    /* y += P_upper x but skip diagonal entries*/
    cudaMalloc(&y_dev,d->m*d->n *sizeof(scs_float));
    _cuda_accum_by_p<<<1, dim3(d->m, d->n)>>>(y_dev, Pp, Pi, x);
    cudaMemcpy(y_dev,y,d->m*d->n*sizeof(scs_float),cudaMemcpyHostToDevice);
    /* y += P_lower x */
    SCS(accum_by_atrans)(P, x, y);
  }

__global__ void _cuda_compute_rsk(scs_float *rski, ScsWork *w) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  rski[i] *= (w->v[i] + w->u[i] - 2 * w->u_t[i]) * w->diag_r[i];
}

void SCS(compute_rsk)(ScsWork *w) {
    scs_float rski_dev;
    scs_int l = w->d->m + w->d->n + 1;
    cudaMalloc(&rski_dev,l*sizeof(scs_float));
    _cuda_compute_rsk<<<(l+255)/256, 256>>>(rski_dev, w);
    cudaMemcpy(rski_dev,w->rsk,l*sizeof(scs_float),cudaMemcpyHostToDevice);
  }

__global__ void _cuda_update_dual_vars(scs_float *vi_dev, ScsWork *w) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  w->v[i] += w->stgs->alpha * (w->u[i] - w->u_t[i]);
}

void SCS(update_dual_vars)(ScsWork *w) {
  scs_float vi_dev;
  scs_int i, l = w->d->n + w->d->m + 1;
  cudaMalloc(&vi_dev,l*sizeof(scs_float));
  _cuda_update_dual_vars<<<(l+255)/256, 256>>>(vi_dev, w);
  cudaMemcpy(vi_dev,w->v,l*sizeof(scs_float),cudaMemcpyHostToDevice);
}