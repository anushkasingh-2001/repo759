#include "cluster.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }

  const size_t n = std::stoull(argv[1]);
  const size_t t = std::stoull(argv[2]);

  std::vector<float> arr(n);
  std::vector<float> centers(t);
  std::vector<float> dists(t, 0.0f);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(n));

  for (size_t i = 0; i < n; i++) {
    arr[i] = dist(gen);
  }

  std::sort(arr.begin(), arr.end());

  for (size_t i = 0; i < t; i++) {
    centers[i] = static_cast<float>((2 * i + 1) * n) / static_cast<float>(2 * t);
  }

  auto start = std::chrono::high_resolution_clock::now();
  cluster(n, t, arr.data(), centers.data(), dists.data());
  auto end = std::chrono::high_resolution_clock::now();

  float max_dist = dists[0];
  size_t max_id = 0;
  for (size_t i = 1; i < t; i++) {
    if (dists[i] > max_dist) {
      max_dist = dists[i];
      max_id = i;
    }
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << max_dist << "\n";
  std::cout << max_id << "\n";
  std::cout << elapsed.count() << "\n";

  return 0;
}