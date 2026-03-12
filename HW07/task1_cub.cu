#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task1_cub n\n";
        return 1;
    }

    const int n = std::stoi(argv[1]);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // First call only to get temp storage size
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    float result = 0.0f;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << result << "\n";
    std::cout << ms << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
