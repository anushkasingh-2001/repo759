#include "montecarlo.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }

  const size_t n = std::stoull(argv[1]);
  const int t = std::stoi(argv[2]);
  const float radius = 1.0f;

  std::vector<float> x(n);
  std::vector<float> y(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-radius, radius);

  for (size_t i = 0; i < n; i++) {
    x[i] = dist(gen);
    y[i] = dist(gen);
  }

  omp_set_num_threads(t);

  auto start = std::chrono::high_resolution_clock::now();
  const int incircle = montecarlo(n, x.data(), y.data(), radius);
  auto end = std::chrono::high_resolution_clock::now();

  const double pi_est = 4.0 * static_cast<double>(incircle) / static_cast<double>(n);
  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << pi_est << "\n";
  std::cout << elapsed.count() << "\n";

  return 0;
}