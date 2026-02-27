#include "mmul.h"

#include <cuda_runtime.h>
#include <iostream>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    // Column-major GEMM: C := alpha * A * B + beta * C
    cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        A, n,
        B, n,
        &beta,
        C, n
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed\n";
    }

  
    cudaDeviceSynchronize();
}