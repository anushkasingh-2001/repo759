#include "convolution.h"

static inline float padded_value(const float* image, std::size_t n,
                                 long long i, long long j) {
    const bool in_i = (i >= 0 && i < static_cast<long long>(n));
    const bool in_j = (j >= 0 && j < static_cast<long long>(n));

    if (in_i && in_j) {
        return image[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)];
    }

    if (in_i != in_j) return 1.0f;
    return 0.0f;
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    const long long r = static_cast<long long>((m - 1) / 2);

    #pragma omp parallel for schedule(static)
    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    const long long ix =
                        static_cast<long long>(x) + static_cast<long long>(i) - r;
                    const long long iy =
                        static_cast<long long>(y) + static_cast<long long>(j) - r;

                    sum += mask[i * m + j] * padded_value(image, n, ix, iy);
                }
            }

            output[x * n + y] = sum;
        }
    }
}