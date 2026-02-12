#include <cuda_runtime.h>
#include <iostream>
#include <random>

__global__ void compute_ax_plus_y(int* dA, int a) {
  int x = threadIdx.x;   // 0..7
  int y = blockIdx.x;    // 0..1

  // Storing in sequential order:
  // block 0 -> indices 0..7
  // block 1 -> indices 8..15
  int idx = y * blockDim.x + x;  // calculating the thread index

  dA[idx] = a * x + y;
}

int main() {
  // Generating random integer a in [-100, 100]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-100, 100);
  int a = dist(gen);

  // Allocating device array dA of 16 ints
  int* dA = nullptr;
  cudaMalloc(&dA, 16 * sizeof(int));

  // Launching kernel: 2 blocks, 8 threads per block
  compute_ax_plus_y<<<2, 8>>>(dA, a);
  cudaDeviceSynchronize();

  // Copy back into host array hA
  int hA[16];
  cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);

  // Print 16 sequential values separated by single space
  for (int i = 0; i < 16; ++i) {
    std::cout << hA[i];
    if (i != 15) std::cout << " ";
  }
  std::cout << "\n";

  cudaFree(dA);
  return 0;
}
