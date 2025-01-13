#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 20015
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20236
#else
#pragma diag_suppress 20015
#pragma diag_suppress 20012
#pragma diag_suppress 20013
#pragma diag_suppress 20236
#endif
#endif

#include "cuda_format.hpp"
#include "VoxelHash.h"
namespace cuda_format {

void gpuAssert(cudaError_t code, bool abort)
{
    if (code != cudaSuccess) 
    {
        ROS_ERROR("GPUassert: %s", cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

__global__ void convertToFloat3Kernel(const PointType* input, float3* positions, float3* normals, size_t numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        positions[idx] = make_float3(input[idx].x, input[idx].y, input[idx].z);
        normals[idx] = make_float3(input[idx].normal_x, input[idx].normal_y, input[idx].normal_z);
        // printf("positions: %f, %f, %f\n", positions[idx].x, positions[idx].y, positions[idx].z);
        // printf("normals: %f, %f, %f\n", normals[idx].x, normals[idx].y, normals[idx].z);
    }
}

void convertToFloat3(const PointType* input, float3* positions, float3* normals, size_t numPoints, HashData *d_hashdata) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    convertToFloat3Kernel<<<blocksPerGrid, threadsPerBlock>>>(input, positions, normals, numPoints);
    gpuAssert(cudaDeviceSynchronize());
    updatesdfframe<<<blocksPerGrid,threadsPerBlock>>>(d_hashdata,positions,normals,numPoints);
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaGetLastError());
}

}