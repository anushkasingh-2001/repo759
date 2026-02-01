#include "convolution.h"

#include <chrono>
#include <cstdlib>   
#include <iostream>
#include <random>

static bool parse_positive_int(const char* s, std::size_t& out) {
    char* endptr = nullptr;
    unsigned long long v = std::strtoull(s, &endptr, 10);
    if (endptr == s || *endptr != '\0' || v == 0ULL) return false;
    out = static_cast<std::size_t>(v);
    return true;
}

int main(int argc, char** argv) {
    // i) Read n, ii) read m
    if (argc < 3) {
        std::cerr << "Usage: ./task2 n m\n";
        return 1;
    }

    std::size_t n = 0, m = 0;
    if (!parse_positive_int(argv[1], n) || !parse_positive_int(argv[2], m)) {
        std::cerr << "n and m must be positive integers\n";
        return 1;
    }
    if ((m % 2) == 0) {
        std::cerr << "m must be odd\n";
        return 1;
    }

    // i) Create n*n image in row-major (1D)
    float* image  = new float[n * n];
    float* output = new float[n * n];

    // ii) Create m*m mask in row-major (1D)
    float* mask = new float[m * m];

    std::mt19937 rng(12345); // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) image[i] = dist_img(rng);
    for (std::size_t i = 0; i < m * m; ++i) mask[i] = dist_mask(rng);

    // iii) Apply mask using convolve and time it
    const auto t0 = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    const auto t1 = std::chrono::high_resolution_clock::now();

    // iv) Print time taken in milliseconds
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << ms << "\n";

    // v) Print first element of output
    std::cout << output[0] << "\n";

    // vi) Print last element of output
    std::cout << output[n * n - 1] << "\n";

    // vii) Deallocate
    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}
