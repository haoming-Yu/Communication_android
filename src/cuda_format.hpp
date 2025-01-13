#include <pcl/point_types.h>
#include <cuda_runtime.h>
#include <ros/ros.h>
#include "voxel_hashing/VoxelHash.h"
#define cudaCheck(ans) { cuda_format::gpuAssert((ans)); }
typedef pcl::PointXYZINormal PointType;

namespace cuda_format {
    // add cudaCheck to check CUDA errors, print error message and exit if error occurs

    void gpuAssert(cudaError_t code, bool abort=true);
    
    __global__ void convertToFloat3Kernel(const PointType* input, float3* positions, float3* normals, size_t numPoints);
    
    void convertToFloat3(const PointType* input, float3* positions, float3* normals, size_t numPoints, HashData *d_hashdata);
}

