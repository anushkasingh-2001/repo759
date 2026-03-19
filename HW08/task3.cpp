#include "msort.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./task3 n t ts\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    const int t = std::stoi(argv[2]);
    const std::size_t ts = static_cast<std::size_t>(std::stoull(argv[3]));

    if (n == 0 || t < 1 || t > 20 || ts == 0) {
        std::cerr << "n must be positive, t in [1,20], ts positive\n";
        return 1;
    }

    std::vector<int> arr(n);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (auto& x : arr) x = dist(rng);

    omp_set_num_threads(t);

    auto start = std::chrono::high_resolution_clock::now();
    msort(arr.data(), n, ts);
    auto end = std::chrono::high_resolution_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << arr.front() << "\n";
    std::cout << arr.back() << "\n";
    std::cout << std::fixed << std::setprecision(6) << ms << "\n";

    return 0;
}