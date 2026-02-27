#include "scan.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <cstdlib>

static inline float rand_float(std::mt19937& gen) {
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(gen);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n threads_per_block\n";
        return 1;
    }

    const unsigned int n = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    const unsigned int threads_per_block = static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));

    if (n == 0 || threads_per_block == 0) {
        std::cerr << "n and threads_per_block must be positive\n";
        return 1;
    }

    float* input = nullptr;
    float* output = nullptr;
    cudaMallocManaged(&input,  n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    std::mt19937 gen(12345);
    for (unsigned int i = 0; i < n; ++i) {
        input[i] = rand_float(gen);
        output[i] = 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);


    std::cout << output[n - 1] << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(input);
    cudaFree(output);

    return 0;
}