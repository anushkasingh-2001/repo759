#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task1_thrust n\n";
        return 1;
    }

    const size_t n = std::stoull(argv[1]);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    thrust::host_vector<float> h_in(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    thrust::device_vector<float> d_in = h_in;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float sum = thrust::reduce(d_in.begin(), d_in.end(), 0.0f, thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << sum << "\n";
    std::cout << ms << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
