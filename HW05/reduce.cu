// reduce.cu
#include "reduce.cuh"
#include <cuda_runtime.h>

// Kernel 4: First Add During Load
// Each block reduces up to 2 * blockDim.x elements.
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = 2 * blockIdx.x * blockSize + tid;

    float x = 0.0f;
    if (i < n) x = g_idata[i];
    if (i + blockSize < n) x += g_idata[i + blockSize];

    sdata[tid] = x;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    if (N == 0 || threads_per_block == 0) {
        cudaDeviceSynchronize();
        return;
    }

    float *in = *input;
    float *out = *output;
    unsigned int n = N;

    while (n > 1) {
        unsigned int blocks =
            (n + (2 * threads_per_block - 1)) / (2 * threads_per_block);

        size_t shmem_bytes = threads_per_block * sizeof(float);

        reduce_kernel<<<blocks, threads_per_block, shmem_bytes>>>(in, out, n);

        // Next pass reduces the partial sums
        n = blocks;

        // Ping-pong buffers
        float *tmp = in;
        in = out;
        out = tmp;
    }

    // Final answer must be in (*input)[0]
    if (in != *input) {
        cudaMemcpy(*input, in, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Required by spec for timing purposes
    cudaDeviceSynchronize();
}