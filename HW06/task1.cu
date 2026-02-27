#include "mmul.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <random>
#include <cstdlib>

static inline float rand_float(std::mt19937& gen) {
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(gen);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n n_tests\n";
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const int n_tests = std::atoi(argv[2]);

    if (n <= 0 || n_tests <= 0) {
        std::cerr << "n and n_tests must be positive integers\n";
        return 1;
    }

    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);

    float *A = nullptr, *B = nullptr, *C = nullptr;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    std::mt19937 gen(12345);
    for (size_t i = 0; i < elems; ++i) {
        A[i] = rand_float(gen);
        B[i] = rand_float(gen);
        C[i] = rand_float(gen);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    //  warm-up launch of kernels
    mmul(handle, A, B, C, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_tests; ++i) {
        mmul(handle, A, B, C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    
    std::cout << (total_ms / static_cast<float>(n_tests)) << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}