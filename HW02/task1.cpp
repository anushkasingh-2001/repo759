#include "scan.h"

#include <chrono>
#include <cstdlib>   // std::strtoull
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    // i) Read n from command line
    if (argc < 2) {
        std::cerr << "Usage: ./task1 n\n";
        return 1;
    }

    char* endptr = nullptr;
    unsigned long long n_ull = std::strtoull(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0' || n_ull == 0ULL) {
        std::cerr << "n must be a positive integer\n";
        return 1;
    }

    std::size_t n = static_cast<std::size_t>(n_ull);

    // ii) Create input array of n random floats in [-1.0, 1.0]
    float* arr = new float[n];
    float* out = new float[n];

    std::mt19937 rng(12345);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(rng);
    }

    // iii) Scan and time it (milliseconds)
    const auto t0 = std::chrono::high_resolution_clock::now();
    scan(arr, out, n);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // iv) Print time in ms
    std::cout << ms << "\n";
    // v) Print first output element
    std::cout << out[0] << "\n";
    // vi) Print last output element
    std::cout << out[n - 1] << "\n";

    // vii) Deallocate
    delete[] arr;
    delete[] out;

    return 0;
}
