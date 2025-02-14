#include "linalg.h"
#include "scs_blas.h"
#include <math.h>
#include <stdio.h>
#include "scs_blas.h"

__global__ void _cuda_scale_array_kernel(scs_float *a, const scs_float b, scs_int len) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  a[i] *= b;
}

/* a *= b */
void SCS(scale_array)(scs_float *a, const scs_float b, scs_int len) {
  scs_float a_dev;
  cudaMalloc(&a_dev,len*sizeof(scs_float));
  _cuda_scale_array_kernel<<<(len+255)/256, 256>>>(a_dev, b);
  cudaMemcpy(a_dev,a,len*sizeof(scs_float),cudaMemcpyHostToDevice);
}

__global__ void _cuda_add_scaled_array_kernel(scs_float *a, const scs_float *b, const scs_float sc) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  a[i] += sc * b[i];
}


/* axpy a += sc*b */
void SCS(add_scaled_array)(scs_float *a, const scs_float *b, scs_int n,
                           const scs_float sc) {
  scs_float a_dev;
  cudaMalloc(&a_dev,n*sizeof(scs_float));
  _cuda_add_scaled_array_kernel<<<(n+255)/256, 256>>>(a, b, sc);
  cudaMemcpy(a_dev,a,n*sizeof(scs_float),cudaMemcpyHostToDevice);
}

__global__ void _entrywise_prod_kernel(scs_float *v, const scs_float *a, const scs_float *b) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  v[i] = a[i] * b[i]
}

// /* ||v||_2^2 */
// scs_float SCS(norm_sq)(const scs_float *v, scs_int len) {
//   scs_float prod;
//   cudaMalloc(&prod, len*sizeof(scs_float));
//   scs_float sum;
//   scs_int i;
//   scs_float nmsq = 0.0;
//   for (i = 0; i < len; ++i) {
//     nmsq += v[i] * v[i];
//   }
//   return nmsq;
// }

// /* ||v||_2 */
// scs_float SCS(norm_2)(const scs_float *v, scs_int len) {
//   return SQRTF(SCS(norm_sq)(v, len));
// }

#ifdef __cplusplus
extern "C" {
#endif

scs_float BLAS(nrm2)(blas_int *n, const scs_float *x, blas_int *incx);
scs_float BLAS(dot)(const blas_int *n, const scs_float *x, const blas_int *incx,
                    const scs_float *y, const blas_int *incy);
void BLAS(axpy)(blas_int *n, const scs_float *a, const scs_float *x,
                blas_int *incx, scs_float *y, blas_int *incy);
void BLAS(scal)(const blas_int *n, const scs_float *sa, scs_float *sx,
                const blas_int *incx);
blas_int BLASI(amax)(blas_int *n, const scs_float *x, blas_int *incx);

/* Possibly not working correctly on all platforms.
scs_float BLAS(lange)(const char *norm, const blas_int *m, const blas_int *n,
                      const scs_float *a, blas_int *lda, scs_float *work);
*/

#ifdef __cplusplus
}
#endif

/* ||v||_2^2 */
scs_float SCS(norm_sq)(const scs_float *v, scs_int len) {
  scs_float nrm = SCS(norm_2)(v, len);
  return nrm * nrm;
}

/* ||v||_2 */
scs_float SCS(norm_2)(const scs_float *v, scs_int len) {
  blas_int bone = 1;
  blas_int blen = (blas_int)len;
  return BLAS(nrm2)(&blen, v, &bone);
}

/* x'*y */
scs_float SCS(dot)(const scs_float *x, const scs_float *y, scs_int len) {
  blas_int bone = 1;
  blas_int blen = (blas_int)len;
  return BLAS(dot)(&blen, x, &bone, y, &bone);
}
