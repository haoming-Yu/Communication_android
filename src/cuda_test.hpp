#include <iostream>
#include <cuda_runtime.h>

namespace cuda_test {

__global__ void test_kernel(int* d_out);

void checkCudaError(cudaError_t err, const char* msg);

void run_cuda_test();

};
