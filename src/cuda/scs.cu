#include "scs.h"
#include "aa.h"
#include "ctrlc.h"
#include "glbopts.h"
#include "linalg.h"
#include "linsys.h"
#include "normalize.h"
#include "rw.h"
#include "scs_matrix.h"
#include "scs_work.h"
#include "util.h"

__global__ void _cuda_compute_rsk(scs_float *rski, ScsWork *w) {
    scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
    rski[i] *= (w->v[i] + w->u[i] - 2 * w->u_t[i]) * w->diag_r[i];
  }

static void compute_rsk(ScsWork *w) {
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

  static void update_dual_vars(ScsWork *w) {
    scs_float vi_dev;
    scs_int i, l = w->d->n + w->d->m + 1;
    cudaMalloc(&vi_dev,l*sizeof(scs_float));
    _cuda_update_dual_vars<<<(l+255)/256, 256>>>(vi_dev, w);
    cudaMemcpy(vi_dev,w->v,l*sizeof(scs_float),cudaMemcpyHostToDevice);
  }