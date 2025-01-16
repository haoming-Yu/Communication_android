#ifndef CUDA_TEST_HPP
#define CUDA_TEST_HPP

#include <iostream>
#include <cuda_runtime.h>

#define cudaCheckError(err) if (err != cudaSuccess) { printf("CUDA error: %s, line: %d, file: %s\n", cudaGetErrorString(err), __LINE__, __FILE__); }

namespace cuda_test {

__global__ void test_kernel(int* d_out);

void checkCudaError(cudaError_t err, const char* msg);

void run_cuda_test();

};

#endif