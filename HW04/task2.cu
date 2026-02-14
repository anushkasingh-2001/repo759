// task2.cu
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cstdlib>
#include "stencil.cuh"

static void die_if_cuda_error(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./task2 n R threads_per_block\n";
    return 1;
  }

  unsigned int n   = (unsigned int)std::strtoul(argv[1], nullptr, 10);
  unsigned int R   = (unsigned int)std::strtoul(argv[2], nullptr, 10);
  unsigned int tpb = (unsigned int)std::strtoul(argv[3], nullptr, 10);

  if (n == 0 || R == 0 || tpb == 0) {
    std::cerr << "n, R, and threads_per_block must be positive integers.\n";
    return 1;
  }
  if (tpb < 2 * R + 1) {
    std::cerr << "Error: threads_per_block must be >= 2*R+1.\n";
    return 1;
  }

  // Host arrays
  std::vector<float> image(n);
  std::vector<float> mask(2 * R + 1);

  // Fill with random numbers in [-1, 1]
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (unsigned int i = 0; i < n; ++i) image[i] = dist(rng);
  for (unsigned int j = 0; j < 2 * R + 1; ++j) mask[j] = dist(rng);

  // Device arrays
  float *image_d = nullptr, *mask_d = nullptr, *output_d = nullptr;
  die_if_cuda_error(cudaMalloc(&image_d,  (size_t)n * sizeof(float)), "cudaMalloc image_d failed");
  die_if_cuda_error(cudaMalloc(&output_d, (size_t)n * sizeof(float)), "cudaMalloc output_d failed");
  die_if_cuda_error(cudaMalloc(&mask_d,   (size_t)(2 * R + 1) * sizeof(float)), "cudaMalloc mask_d failed");

  die_if_cuda_error(cudaMemcpy(image_d, image.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy image->image_d failed");
  die_if_cuda_error(cudaMemcpy(mask_d, mask.data(), (size_t)(2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy mask->mask_d failed");

  // ---- Warm-up (NOT timed) ----
  cudaFree(0); // initialize CUDA context
  stencil(image_d, mask_d, output_d, n, R, tpb);
  die_if_cuda_error(cudaDeviceSynchronize(), "Warm-up cudaDeviceSynchronize failed");
  // -----------------------------

  // Time stencil() with CUDA events
  cudaEvent_t start, stop;
  die_if_cuda_error(cudaEventCreate(&start), "cudaEventCreate start failed");
  die_if_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  die_if_cuda_error(cudaEventRecord(start), "cudaEventRecord start failed");
  stencil(image_d, mask_d, output_d, n, R, tpb);
  die_if_cuda_error(cudaEventRecord(stop), "cudaEventRecord stop failed");
  die_if_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float ms = 0.0f;
  die_if_cuda_error(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime failed");

  // Copy back only last element and print
  float last = 0.0f;
  die_if_cuda_error(cudaMemcpy(&last, output_d + (n - 1), sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy last element failed");

  std::cout << std::fixed << std::setprecision(2) << last << "\n";
  std::cout << std::fixed << std::setprecision(2) << ms << "\n";

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(image_d);
  cudaFree(mask_d);
  cudaFree(output_d);

  return 0;
}
