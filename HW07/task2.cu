#include "count.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task2 n\n";
        return 1;
    }

    const size_t n = std::stoull(argv[1]);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 500);

    thrust::host_vector<int> h_in(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    if (values.empty() || counts.empty()) {
        std::cerr << "count() returned empty outputs.\n";
        return 1;
    }

    std::cout << values.back() << "\n";
    std::cout << counts.back() << "\n";
    std::cout << std::fixed << std::setprecision(6) << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
