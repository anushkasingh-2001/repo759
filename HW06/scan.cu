#include "scan.cuh"

#include <cuda_runtime.h>
#include <iostream>

// Per-block inclusive scan (Hillis-Steele) in shared memory.
// Each block scans one chunk of the global input.
// If block_sums != nullptr, writes the final sum of each block chunk.
__global__ void hillis_steele(const float* input, float* output, float* block_sums, unsigned int n) {
    extern __shared__ float shmem[];
    float* prev = shmem;                  // blockDim.x floats
    float* curr = shmem + blockDim.x;     // blockDim.x floats

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Load (inactive threads use 0 so they don't affect valid outputs)
    prev[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Hillis-Steele inclusive scan within the block
    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        float v = prev[tid];
        if (tid >= offset) v += prev[tid - offset];
        curr[tid] = v;
        __syncthreads();

        // swap buffers
        float* tmp = prev;
        prev = curr;
        curr = tmp;
        __syncthreads();
    }

    if (gid < n) {
        output[gid] = prev[tid];
    }

    // Save this block's sum (last valid entry in the block segment)
    if (block_sums != nullptr && tid == 0) {
        const unsigned int block_start = blockIdx.x * blockDim.x;
        const unsigned int block_end   = min(block_start + blockDim.x, n);
        const unsigned int valid_count = (block_end > block_start) ? (block_end - block_start) : 0;
        block_sums[blockIdx.x] = (valid_count > 0) ? prev[valid_count - 1] : 0.0f;
    }
}

// Add scanned block offsets to each block (except block 0).
// block_scan is inclusive scan of block sums, so offset for block b is block_scan[b-1].
__global__ void add_block_offsets(float* output, const float* block_scan, unsigned int n) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (blockIdx.x == 0) return;

    output[gid] += block_scan[blockIdx.x - 1];
}

// Required host function: launch kernels only, no host data processing.
__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    if (n == 0 || threads_per_block == 0) return;

    const unsigned int tpb = threads_per_block;
    const unsigned int num_blocks = (n + tpb - 1) / tpb;
    const size_t shmem_bytes = 2ull * tpb * sizeof(float);

    // Single block case: one HS launch is enough
    if (num_blocks == 1) {
        hillis_steele<<<1, tpb, shmem_bytes>>>(input, output, nullptr, n);
        return;
    }

    // Assignment allows extra memory smaller than input.
    // HW says for the tested case (tpb=1024, n up to 2^16), num_blocks <= 64, so this is safe.
    float* block_sums = nullptr;
    float* block_scan = nullptr;
    cudaMallocManaged(&block_sums, num_blocks * sizeof(float));
    cudaMallocManaged(&block_scan, num_blocks * sizeof(float));

    // 1) Scan each block chunk and collect block sums
    hillis_steele<<<num_blocks, tpb, shmem_bytes>>>(input, output, block_sums, n);

    // 2) Scan block sums (fits in one block for HW benchmark sizes)
    hillis_steele<<<1, tpb, shmem_bytes>>>(block_sums, block_scan, nullptr, num_blocks);

    // 3) Add offsets to blocks 1..end
    add_block_offsets<<<num_blocks, tpb>>>(output, block_scan, n);

    cudaFree(block_sums);
    cudaFree(block_scan);
}