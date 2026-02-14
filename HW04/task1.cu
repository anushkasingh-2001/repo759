// task1.cu
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

static void die_if_cuda_error(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./task1 n threads_per_block\n";
    return 1;
  }

  size_t n = (size_t)std::strtoull(argv[1], nullptr, 10);
  unsigned int tpb = (unsigned int)std::strtoul(argv[2], nullptr, 10);

  if (n == 0 || tpb == 0) {
    std::cerr << "n and threads_per_block must be positive.\n";
    return 1;
  }

  const size_t N = n * n;

  // Host matrices
  std::vector<float> A(N), B(N);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  // Device matrices
  float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
  die_if_cuda_error(cudaMalloc(&A_d, N * sizeof(float)), "cudaMalloc A_d failed");
  die_if_cuda_error(cudaMalloc(&B_d, N * sizeof(float)), "cudaMalloc B_d failed");
  die_if_cuda_error(cudaMalloc(&C_d, N * sizeof(float)), "cudaMalloc C_d failed");

  die_if_cuda_error(cudaMemcpy(A_d, A.data(), N * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy A->A_d failed");
  die_if_cuda_error(cudaMemcpy(B_d, B.data(), N * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy B->B_d failed");

  matmul(A_d, B_d, C_d, n, tpb);  // warm up pf kernel (as did in previous assignment for fair comparison)
  cudaDeviceSynchronize();     // wait for warm-up kernel to finish

  // CUDA events timing
  cudaEvent_t start, stop;
  die_if_cuda_error(cudaEventCreate(&start), "cudaEventCreate start failed");
  die_if_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  die_if_cuda_error(cudaEventRecord(start), "cudaEventRecord start failed");
  matmul(A_d, B_d, C_d, n, tpb);
  die_if_cuda_error(cudaEventRecord(stop), "cudaEventRecord stop failed");
  die_if_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float ms = 0.0f;
  die_if_cuda_error(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime failed");

  // Copy back last element only
  float last = 0.0f;
  die_if_cuda_error(cudaMemcpy(&last, C_d + (N - 1), sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy last element failed");

  std::cout << std::fixed << std::setprecision(2) << last << "\n";
  std::cout << std::fixed << std::setprecision(2) << ms << "\n";

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
