
#include "convolution.h"

// Return the padded value for image coordinate (i, j).
// Image is n x n, row-major.
// Padding rule (from HW):
// - If (i,j) is inside the image: use image[i,j]
// - If it is outside on an "edge extension" (excluding corners): pad with 1
// - If it is outside in a "corner region": pad with 0
static inline float padded_value(const float* image, std::size_t n, long long i, long long j) {
    const bool in_i = (i >= 0 && i < static_cast<long long>(n));
    const bool in_j = (j >= 0 && j < static_cast<long long>(n));

    if (in_i && in_j) {
        return image[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)];
    }

    // Exactly one coordinate is in range => edge (not corner) => 1
    // Neither coordinate is in range => corner => 0
    if (in_i != in_j) return 1.0f;
    return 0.0f;
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    if (n == 0 || m == 0) return;

    // HW says m is odd (so we have a center)
    const long long r = static_cast<long long>((m - 1) / 2);

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            double sum = 0.0;

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    const long long ix = static_cast<long long>(x) + static_cast<long long>(i) - r;
                    const long long iy = static_cast<long long>(y) + static_cast<long long>(j) - r;

                    const float imgv = padded_value(image, n, ix, iy);
                    const float w = mask[i * m + j];

                    sum += static_cast<double>(w) * static_cast<double>(imgv);
                }
            }

            output[x * n + y] = static_cast<float>(sum);
        }
    }
}
