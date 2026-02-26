// matmul.cu
#include "matmul.cuh"
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel (templated)
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    // Dynamic shared memory stores two tiles: As and Bs
    extern __shared__ unsigned char smem[];
    T *As = reinterpret_cast<T *>(smem);
    T *Bs = As + (blockDim.x * blockDim.y);

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int row = blockIdx.y * blockDim.y + ty;
    const unsigned int col = blockIdx.x * blockDim.x + tx;

    T sum = static_cast<T>(0);

    const unsigned int tile_w = blockDim.x; // assuming square blocks (block_dim x block_dim)
    const unsigned int num_tiles = (n + tile_w - 1) / tile_w;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        const unsigned int a_col = t * tile_w + tx;
        const unsigned int b_row = t * tile_w + ty;

        // Load A tile
        if (row < n && a_col < n) {
            As[ty * blockDim.x + tx] = A[row * n + a_col];
        } else {
            As[ty * blockDim.x + tx] = static_cast<T>(0);
        }

        // Load B tile
        if (b_row < n && col < n) {
            Bs[ty * blockDim.x + tx] = B[b_row * n + col];
        } else {
            Bs[ty * blockDim.x + tx] = static_cast<T>(0);
        }

        __syncthreads();

        // Multiply tile pair
        for (unsigned int k = 0; k < tile_w; ++k) {
            sum += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

template <typename T>
static inline void matmul_impl(const T *A, const T *B, T *C, unsigned int n,
                               unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t shmem_bytes = 2 * block_dim * block_dim * sizeof(T);

    matmul_kernel<T><<<grid, block, shmem_bytes>>>(A, B, C, n);

    // Required by spec for timing purposes
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
    matmul_impl<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
    matmul_impl<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
    matmul_impl<double>(A, B, C, n, block_dim);
}