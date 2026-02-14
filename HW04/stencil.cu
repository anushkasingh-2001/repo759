// stencil.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "stencil.cuh"

// image[x] = 1 when x < 0 or x >= n
__device__ __forceinline__ float image_at(const float* image, int x, int n) {
  return (x < 0 || x >= n) ? 1.0f : image[x];
}

__global__ void stencil_kernel(const float* image,
                               const float* mask,
                               float* output,
                               unsigned int n,
                               unsigned int R) {
  const unsigned int t = (unsigned int)threadIdx.x;
  const unsigned int B = (unsigned int)blockDim.x;

  const int base = (int)(blockIdx.x * B);
  const int i    = base + (int)t;          // global output index for this thread

  // Shared memory layout (dynamic only):
  // [mask (2R+1)] [image tile (B+2R)] [output tile (B)]
  extern __shared__ float sh[];
  float* s_mask = sh;
  float* s_img  = s_mask + (2 * R + 1);
  float* s_out  = s_img  + (B + 2 * R);

  // 1) Load entire mask into shared
  // Assumption: B >= 2R+1, so threads 0..2R can load mask.
  if (t < 2 * R + 1) {
    s_mask[t] = mask[t];
  }

  // 2) Load the needed image elements for this block into shared:
  // s_img[0 .. B+2R-1] corresponds to global indices [base-R .. base+B+R-1]
  // Use striding so all threads participate.
  for (unsigned int off = t; off < B + 2 * R; off += B) {
    int g = base + (int)off - (int)R;
    s_img[off] = image_at(image, g, (int)n);
  }

  __syncthreads();

  // 3) Compute one output element per thread (if in range), store into shared output first
  if (i < (int)n) {
    float sum = 0.0f;
    const unsigned int center = t + R; // s_img index for image[i]
    for (int j = -(int)R; j <= (int)R; ++j) {
      sum += s_img[center + j] * s_mask[j + (int)R];
    }
    s_out[t] = sum;
  } else {
    s_out[t] = 0.0f; // defined value for out-of-range threads
  }

  __syncthreads();

  // 4) Write block output from shared to global
  if (i < (int)n) {
    output[i] = s_out[t];
  }
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block) {
  if (n == 0) return;

  const unsigned int B = threads_per_block;
  const unsigned int num_blocks = (n + B - 1) / B;

  // Dynamic shared memory (floats):
  // mask: (2R+1)
  // image tile: (B+2R)
  // output tile: (B)
  const size_t sh_floats =
      (size_t)(2 * R + 1) + (size_t)(B + 2 * R) + (size_t)B;
  const size_t sh_bytes = sh_floats * sizeof(float);

  stencil_kernel<<<num_blocks, B, sh_bytes>>>(image, mask, output, n, R);

  // Minimal error check (recommended)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}
