#include <cuda_runtime.h>
#include <iostream>

__global__ void factorial_kernel(int* dA) {
  int a = threadIdx.x;   // 0..7
  int n = a + 1;         // 1..8

  int f = 1;
  for (int i = 2; i <= n; ++i) f *= i;

  dA[a] = f;             // store in a-th position
}

int main() {
  // 1) allocating dA on device
  int* dA = nullptr;
  cudaMalloc(&dA, 8 * sizeof(int));

  // 2) launching kernel: 1 block, 8 threads
  factorial_kernel<<<1, 8>>>(dA);

  // waiting for kernel
  cudaDeviceSynchronize();

  // 3) copy device array to host array hA
  int hA[8];
  cudaMemcpy(hA, dA, 8 * sizeof(int), cudaMemcpyDeviceToHost);

  // 4) print 8 values, one per line
  for (int i = 0; i < 8; ++i) {
    std::cout << hA[i] << "\n";
  }

  // 5) cleanup
  cudaFree(dA);
  return 0;
}
