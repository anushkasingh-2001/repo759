// task1.cu
// Usage: ./task1 n block_dim

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "matmul.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename T>
void fill_A(std::vector<T>& A, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<T>((i + j) % 7 + 1);
        }
    }
}

template <typename T>
void fill_B(std::vector<T>& B, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            B[i * n + j] = static_cast<T>((2 * i + j) % 5 + 1);
        }
    }
}

template <typename T>
void run_case(unsigned int n, unsigned int block_dim,
              void (*matmul_fn)(const T*, const T*, T*, unsigned int, unsigned int)) {
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = count * sizeof(T);

    std::vector<T> hA(count), hB(count), hC(count, static_cast<T>(0));
    fill_A(hA, n);
    fill_B(hB, n);

    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_fn(dA, dB, dC, n, block_dim);   // matmul_* ends with cudaDeviceSynchronize()
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Sample style in HW: print first element, last element, time
    std::cout << hC[0] << "\n";
    std::cout << hC[count - 1] << "\n";
    std::cout << ms << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n block_dim\n";
        return EXIT_FAILURE;
    }

    const unsigned int n = static_cast<unsigned int>(std::stoul(argv[1]));
    const unsigned int block_dim = static_cast<unsigned int>(std::stoul(argv[2]));

    if (n == 0 || block_dim == 0) {
        std::cerr << "n and block_dim must be positive integers\n";
        return EXIT_FAILURE;
    }

    run_case<int>(n, block_dim, matmul_1);
    run_case<float>(n, block_dim, matmul_2);
    run_case<double>(n, block_dim, matmul_3);

    return 0;
}