#include "matmul.h"

// Each function computes C = A * B (n x n), stored in row-major order.
// Row-major indexing: element (i,j) is at [i*n + j].

// a) Loop order: (i, j, k)
void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
            for (unsigned int k = 0; k < n; ++k) {
                // single increment line (required)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// b) Loop order: (i, k, j)
void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    // Zero C first because we will accumulate into it
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
        }
    }

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < n; ++k) {
            const double aik = A[i * n + k];
            for (unsigned int j = 0; j < n; ++j) {
                // single increment line (required)
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

// c) Loop order: (j, k, i)
void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    // Zero C first because we will accumulate into it
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
        }
    }

    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int k = 0; k < n; ++k) {
            const double bkj = B[k * n + j];
            for (unsigned int i = 0; i < n; ++i) {
                // single increment line (required)
                C[i * n + j] += A[i * n + k] * bkj;
            }
        }
    }
}

// d) Loop order: (i, j, k), but A and B are std::vector<double>
void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
            for (unsigned int k = 0; k < n; ++k) {
                // single increment line (required)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
