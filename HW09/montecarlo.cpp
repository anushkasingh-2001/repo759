#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
  const float r2 = radius * radius;
  int incircle = 0;

#ifdef USE_SIMD
#pragma omp parallel reduction(+ : incircle)
  {
#pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
      const float xx = x[i];
      const float yy = y[i];
      incircle += ((xx * xx + yy * yy) <= r2) ? 1 : 0;
    }
  }
#else
#pragma omp parallel for reduction(+ : incircle) schedule(static)
  for (size_t i = 0; i < n; i++) {
    const float xx = x[i];
    const float yy = y[i];
    incircle += ((xx * xx + yy * yy) <= r2) ? 1 : 0;
  }
#endif

  return incircle;
}