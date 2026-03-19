#include "matmul.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n t\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    const int t = std::stoi(argv[2]);

    if (n == 0 || t < 1 || t > 20) {
        std::cerr << "n must be positive and t must be in [1, 20]\n";
        return 1;
    }

    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    omp_set_num_threads(t);

    auto start = std::chrono::high_resolution_clock::now();
    mmul(A.data(), B.data(), C.data(), n);
    auto end = std::chrono::high_resolution_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << C.front() << "\n";
    std::cout << C.back() << "\n";
    std::cout << ms << "\n";

    return 0;
}