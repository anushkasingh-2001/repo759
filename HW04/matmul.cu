// matmul.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
  size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t N = n * n;
  if (tid >= N) return;

  size_t row = tid / n;
  size_t col = tid % n;

  float acc = 0.0f;
  size_t a_base = row * n;
  for (size_t k = 0; k < n; ++k) {
    acc += A[a_base + k] * B[k * n + col];
  }
  C[tid] = acc;
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
  if (n == 0) return;

  size_t N = n * n;
  unsigned int tpb = threads_per_block;
  unsigned int blocks = (unsigned int)((N + tpb - 1) / tpb);

  matmul_kernel<<<blocks, tpb>>>(A, B, C, n);

  // Minimal required error check (recommended)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}
