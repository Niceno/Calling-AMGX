#include <cstdio>
#include <cuda_runtime.h>

/*============================================================================*/
extern "C" void cuda_alloc_double_(void ** ptr, const int & N) {
/*----------------------------------------------------------------------------*/

  cudaError_t err = cudaMalloc(ptr, N * sizeof(double));
  if(err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n",
            cudaGetErrorString(err));
  }
}

/*============================================================================*/
extern "C" void cuda_alloc_double_copyin_(void **        ptr,
                                          const double * val_host,
                                          const int    & N) {
/*----------------------------------------------------------------------------*/

  cudaError_t err = cudaMalloc(ptr, N * sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc (double) failed: %s\n",
            cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(*ptr, val_host, N * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy H->D (double) failed: %s\n",
            cudaGetErrorString(err));
  }
}

/*============================================================================*/
extern "C" void cuda_alloc_int_(void ** ptr, const int & N) {
/*----------------------------------------------------------------------------*/

  cudaError_t err = cudaMalloc(ptr, N * sizeof(int));
  if(err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n",
            cudaGetErrorString(err));
  }
}

/*============================================================================*/
extern "C" void cuda_alloc_int_copyin_(void **     ptr,
                                       const int * val_host,
                                       const int & N) {
/*----------------------------------------------------------------------------*/

  cudaError_t err = cudaMalloc(ptr, N * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc (int) failed: %s\n",
            cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(*ptr, val_host, N * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy H->D (int) failed: %s\n",
            cudaGetErrorString(err));
  }
}
