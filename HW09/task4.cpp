#include "convolve.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }

  const std::size_t n = std::stoull(argv[1]);
  const std::size_t m = 3;

  std::vector<float> image(n * n);
  std::vector<float> output(n * n, 0.0f);
  std::vector<float> mask(m * m);

  std::mt19937 gen(42);

  // Image values: float data, same style as earlier matrix tasks
  std::uniform_real_distribution<float> image_dist(0.0f, 255.0f);

  // 3x3 mask values
  std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);

  for (std::size_t i = 0; i < n * n; i++) {
    image[i] = image_dist(gen);
  }

  for (std::size_t i = 0; i < m * m; i++) {
    mask[i] = mask_dist(gen);
  }

  auto start = std::chrono::high_resolution_clock::now();
  convolve(image.data(), output.data(), n, mask.data(), m);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << elapsed.count() << "\n";

  return 0;
}