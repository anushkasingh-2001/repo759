// task2.cu
// Usage: ./task2 N threads_per_block

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include "reduce.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N threads_per_block\n";
        return EXIT_FAILURE;
    }

    const unsigned int N = static_cast<unsigned int>(std::stoul(argv[1]));
    const unsigned int threads_per_block =
        static_cast<unsigned int>(std::stoul(argv[2]));

    if (N == 0 || threads_per_block == 0) {
        std::cerr << "N and threads_per_block must be positive integers\n";
        return EXIT_FAILURE;
    }

    // Host array with random values in [-1, 1]
    std::vector<float> h_input(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (unsigned int i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
    }

    // Device input
    float *d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          static_cast<size_t>(N) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Device output for first kernel call: one partial sum per block
    const unsigned int first_blocks =
        (N + (2 * threads_per_block - 1)) / (2 * threads_per_block);

    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(first_blocks) * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    reduce(&d_input, &d_output, N, threads_per_block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum, d_input, sizeof(float), cudaMemcpyDeviceToHost));

    // Sample style in HW: print sum, then time
    std::cout << sum << "\n";
    std::cout << ms << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}