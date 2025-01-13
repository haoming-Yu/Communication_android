#include <iostream>
#include <cuda_runtime.h>
#include "cuda_test.hpp"

namespace cuda_test {

__global__ void test_kernel(int* d_out) {
    int idx = threadIdx.x;
    d_out[idx] = idx;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void run_cuda_test() {
    const int array_size = 10;
    int h_out[array_size];
    int* d_out;

    checkCudaError(cudaMalloc((void**)&d_out, array_size * sizeof(int)), "Failed to allocate device memory");
    
    test_kernel<<<1, array_size>>>(d_out);

    checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");
    
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    checkCudaError(cudaMemcpy(h_out, d_out, array_size * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

    std::cout << "CUDA Output: ";
    for (int i = 0; i < array_size; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    checkCudaError(cudaFree(d_out), "Failed to free device memory");
}

};
