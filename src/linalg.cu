#include "linalg.h"
#include "scs_blas.h"
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
__global__ void _cuda_scale_array_kernel(scs_float *a, const scs_float b, scs_int len) {
  scs_int i= blockIdx.x*blockDim.x + threadIdx.x;
  a[i] *= b;
}

/* a *= b */
__global__ void scale_kernel(scs_float *a, scs_float b, scs_int len) {
  scs_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
      a[i] *= b;
  }
}

void SCS(scale_array)(scs_float *a, const scs_float b, scs_int len) {
  scs_float *d_a;
  const size_t size = len * sizeof(scs_float);
  cudaMalloc(&d_a, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  const int block_size = 256;
  const int grid_size = (len + block_size - 1) / block_size;
  scale_kernel<<<grid_size, block_size>>>(d_a, b, len);
  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
}

__global__ void add_scaled_kernel(
  scs_float *a, 
  const scs_float *b, 
  const scs_float sc, 
  scs_int len
) {
  scs_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
      a[i] += sc * b[i];
  }
}

void SCS(add_scaled_array)(
  scs_float *a, 
  const scs_float *b, 
  scs_int n, 
  const scs_float sc
) {
  scs_float *d_a, *d_b;
  const size_t size = n * sizeof(scs_float);
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 
  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  add_scaled_kernel<<<grid_size, block_size>>>(d_a, d_b, sc, n);
  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
}


__global__ void norm_sq_kernel(const scs_float *v, scs_int len, scs_float *result) {
  extern __shared__ scs_float shared_data[];
  
  scs_int tid = threadIdx.x;
  scs_int i = blockIdx.x * blockDim.x + tid;
  scs_float sum = 0.0;

  // Each thread accumulates partial sum
  while (i < len) {
      sum += v[i] * v[i];
      i += blockDim.x * gridDim.x;
  }
  
  shared_data[tid] = sum;
  __syncthreads();

  // Parallel reduction within block
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) {
          shared_data[tid] += shared_data[tid + s];
      }
      __syncthreads();
  }

  // Store block result
  if (tid == 0) {
      result[blockIdx.x] = shared_data[0];
  }
}

scs_float SCS(norm_sq)(const scs_float *v, scs_int len) {
  scs_float *d_v, *d_block_results;
  scs_float *h_block_results;
  scs_float final_sum = 0.0;

  // Allocate device memory
  cudaMalloc(&d_v, len * sizeof(scs_float));
  scs_int threadsPerBlock = 256;
  scs_int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 65535) blocksPerGrid = 65535;  // Max grid size
  
  cudaMalloc(&d_block_results, blocksPerGrid * sizeof(scs_float));
  h_block_results = (scs_float*)malloc(blocksPerGrid * sizeof(scs_float));

  // Copy data to device
  cudaMemcpy(d_v, v, len * sizeof(scs_float), cudaMemcpyHostToDevice);

  // Launch kernel with dynamic shared memory
  norm_sq_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(scs_float)>>>(
      d_v, len, d_block_results
  );

  // Copy partial results back
  cudaMemcpy(h_block_results, d_block_results, 
             blocksPerGrid * sizeof(scs_float), cudaMemcpyDeviceToHost);

  // Final reduction on host
  for (scs_int i = 0; i < blocksPerGrid; i++) {
      final_sum += h_block_results[i];
  }

  // Cleanup
  free(h_block_results);
  cudaFree(d_v);
  cudaFree(d_block_results);

  return final_sum;
}

/* ||v||_2 */
scs_float SCS(norm_2)(const scs_float *v, scs_int len) {
  return SQRTF(SCS(norm_sq)(v, len));
}
__global__ void dot_product_kernel(const scs_float *x, const scs_float *y, scs_int len, scs_float *result) {
  extern __shared__ scs_float shared_data[];
  
  scs_int tid = threadIdx.x;
  scs_int i = blockIdx.x * blockDim.x + tid;
  scs_float sum = 0.0;

  // Each thread computes partial dot product
  while (i < len) {
      sum += x[i] * y[i];
      i += blockDim.x * gridDim.x;
  }
  
  shared_data[tid] = sum;
  __syncthreads();

  // Block-level reduction
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) {
          shared_data[tid] += shared_data[tid + s];
      }
      __syncthreads();
  }

  // Store block result
  if (tid == 0) {
      result[blockIdx.x] = shared_data[0];
  }
}

scs_float SCS(dot)(const scs_float *x, const scs_float *y, scs_int len) {
  scs_float *d_x, *d_y, *d_block_results;
  scs_float *h_block_results;
  scs_float final_sum = 0.0;

  // Allocate device memory
  cudaMalloc(&d_x, len * sizeof(scs_float));
  cudaMalloc(&d_y, len * sizeof(scs_float));
  
  scs_int threadsPerBlock = 256;
  scs_int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 65535) blocksPerGrid = 65535;
  
  cudaMalloc(&d_block_results, blocksPerGrid * sizeof(scs_float));
  h_block_results = (scs_float*)malloc(blocksPerGrid * sizeof(scs_float));

  // Copy input data to device
  cudaMemcpy(d_x, x, len * sizeof(scs_float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, len * sizeof(scs_float), cudaMemcpyHostToDevice);

  // Launch kernel
  dot_product_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(scs_float)>>>(
      d_x, d_y, len, d_block_results
  );

  // Copy partial results back
  cudaMemcpy(h_block_results, d_block_results, 
             blocksPerGrid * sizeof(scs_float), cudaMemcpyDeviceToHost);

  // Final host-side reduction
  for (scs_int i = 0; i < blocksPerGrid; i++) {
      final_sum += h_block_results[i];
  }

  // Cleanup
  free(h_block_results);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_block_results);

  return final_sum;
}

}


scs_float SCS(norm_diff)(const scs_float *a, const scs_float *b, scs_int len) {
  scs_float nm_diff = 0.0, tmp;
  scs_int i;
  for (i = 0; i < len; ++i) {
    tmp = (a[i] - b[i]);
    nm_diff += tmp * tmp;
  }
  return SQRTF(nm_diff);
}

scs_float SCS(norm_inf_diff)(const scs_float *a, const scs_float *b,
                             scs_int len) {
  scs_float tmp, max = 0.0;
  scs_int i;
  for (i = 0; i < len; ++i) {
    tmp = ABS(a[i] - b[i]);
    if (tmp > max) {
      max = tmp;
    }
  }
  return max;
}

scs_float SCS(norm_inf)(const scs_float *a, scs_int len) {
  scs_float tmp, max = 0.0;
  scs_int i;
  for (i = 0; i < len; ++i) {
    tmp = ABS(a[i]);
    if (tmp > max) {
      max = tmp;
    }
  }
  return max;
}



scs_float SCS(mean)(const scs_float *x, scs_int n) {
  scs_int i;
  scs_float mean = 0.;
  for (i = 0; i < n; ++i) {
    mean += x[i];
  }
  return mean / n;
}