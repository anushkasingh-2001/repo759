#include <cuda_runtime.h>
#include "vscale.cuh"

// b[i] = a[i] * b[i], each thread does at most one multiply
__global__ void vscale(const float* a, float* b, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  //thread index must be less than the size of the elements (n)
  if (i < n) {
    b[i] = a[i] * b[i];
  }
}

