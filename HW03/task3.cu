#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <iomanip>

#include "vscale.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./task3 n\n";
    return 1;
  }

  int n = std::atoi(argv[1]);
  if (n <= 0) {
    std::cerr << "n must be a positive integer\n";
    return 1;
  }

  // Host arrays
  float* hA = new float[n];
  float* hB = new float[n];

  // Random generation:
  // a in [-10, 10], b in [0, 1]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
  std::uniform_real_distribution<float> distB(0.0f, 1.0f);

  for (int i = 0; i < n; ++i) {
    hA[i] = distA(gen);
    hB[i] = distB(gen);
  }

  // Device arrays
  float *dA = nullptr, *dB = nullptr;
  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));

  cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

  // Kernel launch config: 512 threads per block
  int TPB = 512;
  
  if (argc >= 3) {
  TPB = std::atoi(argv[2]);
  if (TPB <= 0) TPB = 512; 
  }
  int blocks = (n + TPB - 1) / TPB;

  // CUDA event timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  vscale<<<blocks, TPB>>>(dA, dB, n);
  cudaDeviceSynchronize();   // warm-up launch

  cudaEventRecord(start, 0);
  vscale<<<blocks, TPB>>>(dA, dB, n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  // Copy back result
  cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Printing required outputs (each on new line)
  std::cout << std::fixed << std::setprecision(3) << ms << "\n";
  std::cout << std::defaultfloat << std::setprecision(6) << hB[0] << "\n";
  std::cout << std::defaultfloat << std::setprecision(6) << hB[n - 1] << "\n";

  
  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  delete[] hA;
  delete[] hB;

  return 0;
}
