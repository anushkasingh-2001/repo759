#include "convolution.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n t\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    const int t = std::stoi(argv[2]);
    const std::size_t m = 3;

    if (n == 0 || t < 1 || t > 20) {
        std::cerr << "n must be positive and t must be in [1, 20]\n";
        return 1;
    }

    std::vector<float> image(n * n), output(n * n), mask(m * m);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (auto& x : image) x = dist_img(rng);
    for (auto& x : mask) x = dist_mask(rng);

    omp_set_num_threads(t);

    auto start = std::chrono::high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), m);
    auto end = std::chrono::high_resolution_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << output.front() << "\n";
    std::cout << output.back() << "\n";
    std::cout << ms << "\n";

    return 0;
}