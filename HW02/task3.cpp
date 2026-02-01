#include "matmul.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
    const unsigned int n = 1024; // >= 1000, matches sample first line style

    // A and B stored row-major in 1D
    std::vector<double> A(static_cast<std::size_t>(n) * n);
    std::vector<double> B(static_cast<std::size_t>(n) * n);
    std::vector<double> C(static_cast<std::size_t>(n) * n);

    // Fill A and B with random doubles
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Line 1: print n
    std::cout << n << "\n";

    // mmul1: print time, then last element
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul1(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << ms << "\n";
        std::cout << C[static_cast<std::size_t>(n) * n - 1] << "\n";
    }

    // mmul2
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul2(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << ms << "\n";
        std::cout << C[static_cast<std::size_t>(n) * n - 1] << "\n";
    }

    // mmul3
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul3(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << ms << "\n";
        std::cout << C[static_cast<std::size_t>(n) * n - 1] << "\n";
    }

    // mmul4
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul4(A, B, C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << ms << "\n";
        std::cout << C[static_cast<std::size_t>(n) * n - 1] << "\n";
    }

    return 0;
}
