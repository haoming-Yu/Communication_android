#include "pcd_receiver.hpp"
#include "cuda_test.hpp"
#include "cuda_format.hpp"
#include "voxel_hashing/MarchingCubesSDFUtil.h"
#include "voxel_hashing/CUDAMarchingCubesHashSDF.h"
#include "VoxelHash.h"
#include "CUDAMarchingCubesHashSDF.h"
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <termios.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

#define MAX_POINT_CLOUD_ITER 1200
#define MAX_VOXELS 200000 * 500

HashData *d_hashdata;

int count_refresh = 0;

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data);
extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data);
extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data);
extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks);

#define cudaCheckError(err) if (err != cudaSuccess) { printf("CUDA error: %s, line: %d, file: %s\n", cudaGetErrorString(err), __LINE__, __FILE__); }

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams& params)
{ 
	m_params = params;
	m_data.allocate(m_params);

	resetMarchingCubesCUDA(m_data);
}

void CUDAMarchingCubesHashSDF::destroy(void)
{
	m_data.free();
}

void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashData& hashData, const vec3f& minCorner, const vec3f& maxCorner, bool boxEnabled)
{
	resetMarchingCubesCUDA(m_data);
	float3 maxc = {maxCorner.x,maxCorner.y,maxCorner.z};
	float3 minc = {minCorner.x,minCorner.y,minCorner.z};
	m_params.m_maxCorner = maxc;
	m_params.m_minCorner = minc;
	m_params.m_boxEnabled = boxEnabled;
	m_data.updateParams(m_params);

    // get the number of occupied blocks, and save the bucket id of the occupied blocks to pass to pass 2
	extractIsoSurfacePass1CUDA(hashData, m_params, m_data);
    cudaCheckError(cudaGetLastError());
    ROS_INFO("extractIsoSurfacePass1CUDA completed");
    // to extract the triangles, we need to traverse the occupied blocks, and do the marching cubes on each voxel in the occupied blocks.
	extractIsoSurfacePass2CUDA(hashData, m_params, m_data, m_data.getNumOccupiedBlocks());
    cudaCheckError(cudaGetLastError());
    ROS_INFO("extractIsoSurfacePass2CUDA completed");
    // printf("In CUDAMarchingCubesHashSDF::extractIsoSurface, after extractIsoSurfacePass2CUDA\n");
}

void CUDAMarchingCubesHashSDF::export_ply(const std::string& filename)
{
    MarchingCubesData cpuData = m_data.copyToCPU();
    std::ofstream file_out { filename };
    if (!file_out.is_open())
        return;
    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << 3*cpuData.d_numTriangles[0] << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "element face " << cpuData.d_numTriangles[0] << std::endl;
    file_out << "property list uchar int vertex_index" << std::endl;
    file_out << "end_header" << std::endl;

    for (int v_idx = 0; v_idx < cpuData.d_numTriangles[0]; ++v_idx) {
        float3 v0 = cpuData.d_triangles[v_idx].v0.p;
        float3 v1 = cpuData.d_triangles[v_idx].v1.p;
        float3 v2 = cpuData.d_triangles[v_idx].v2.p;
        file_out << v0.x << " " << v0.y << " " << v0.z << " ";
        file_out << v1.x << " " << v1.y << " " << v1.z << " ";
        file_out << v2.x << " " << v2.y << " " << v2.z << " ";
    }

    for (int t_idx = 0; t_idx < 3*cpuData.d_numTriangles[0]; t_idx += 3) {
        file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
    }
}

void PointCloudSubscriber::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    PointCloudXYZI::Ptr tmp(new PointCloudXYZI(msg->width * msg->height, 1));
    this->received_cloud_ = tmp;
    pcl::fromROSMsg(*msg, *(this->received_cloud_));

    ROS_INFO("Received PointCloud message with %ld points", this->received_cloud_->size());

    // upload point cloud data to GPU
    PointType* d_pointCloudData = nullptr;
    size_t numPoints = this->received_cloud_->points.size();
    size_t dataSize = numPoints * sizeof(PointType);

    cudaCheck(cudaMalloc((void**)&d_pointCloudData, dataSize));
    // calculate normals
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud(this->received_cloud_);
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    // ne.setRadiusSearch(0.8);
    ne.setKSearch(100);
    ne.compute(*cloud_normals);
    // normalize normals
    for (size_t i = 0; i < numPoints; ++i) {
        float norm = sqrt(cloud_normals->points[i].normal_x * cloud_normals->points[i].normal_x +
                           cloud_normals->points[i].normal_y * cloud_normals->points[i].normal_y +
                           cloud_normals->points[i].normal_z * cloud_normals->points[i].normal_z);
        cloud_normals->points[i].normal_x /= norm;
        cloud_normals->points[i].normal_y /= norm;
        cloud_normals->points[i].normal_z /= norm;
    }

    // copy normals to point cloud data
    for (size_t i = 0; i < numPoints; ++i) {
        this->received_cloud_->points[i].normal_x = cloud_normals->points[i].normal_x;
        this->received_cloud_->points[i].normal_y = cloud_normals->points[i].normal_y;
        this->received_cloud_->points[i].normal_z = cloud_normals->points[i].normal_z;
    }
    cudaCheck(cudaMemcpy(d_pointCloudData, this->received_cloud_->points.data(), dataSize, cudaMemcpyHostToDevice));

    // free CPU memory
    this->received_cloud_->clear();
    this->received_cloud_.reset();

    // allocate output memory
    float3* d_positions = nullptr;
    float3* d_normals = nullptr;
    cudaCheck(cudaMalloc((void**)&d_positions, numPoints * sizeof(float3)));
    cudaCheck(cudaMalloc((void**)&d_normals, numPoints * sizeof(float3)));
    
    // call CUDA kernel
    cuda_format::convertToFloat3(d_pointCloudData, d_positions, d_normals, numPoints, d_hashdata);
    cudaCheck(cudaDeviceSynchronize());

    // free GPU memory
    cudaCheck(cudaFree(d_pointCloudData));
    cudaCheck(cudaFree(d_positions));
    cudaCheck(cudaFree(d_normals));

    ROS_INFO("PointCloud received and uploaded to GPU successfully\n");
    count_refresh++;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcd_receiver");
    PointCloudSubscriber pcd_receiver;

    ROS_INFO("Running CUDA test...");
    cuda_test::run_cuda_test();

    HashData hash;
    hash.allocate(true);

    cudaMalloc(&d_hashdata,sizeof(HashData));
    cudaMemcpy(d_hashdata,&hash,sizeof(HashData),cudaMemcpyHostToDevice);

    while (ros::ok() && count_refresh < MAX_POINT_CLOUD_ITER)
    {
        ros::spinOnce();
    }

    // marching cubes
    YAML::Node config = YAML::LoadFile("/home/hmy/ws_fast_lio/src/pcd_receiver/config/voxel_hashing.yaml");
    
    MarchingCubesParams mcParams = CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(
        config["marchingCubesMaxNumTriangles"].as<int>(), 
        config["SDFMarchingCubeThreshFactor"].as<int>(), 
        config["virtualVoxelSize"].as<float>(), 
        config["hashNumBuckets"].as<int>(), 
        config["SDFBlockSize"].as<int>(), 
        config["hashBucketSize"].as<int>()
    );
    CUDAMarchingCubesHashSDF marchingCubes(mcParams);
    marchingCubes.extractIsoSurface(hash, vec3f(0,0,0), vec3f(0,0,0), false);
    marchingCubes.export_ply("/home/hmy/ws_fast_lio/src/pcd_receiver/PCD/mesh/output.ply");
    ROS_INFO("Marching cubes completed");

    float count = 0;
    float* d_count;
    float3* host_voxels = new float3[MAX_VOXELS];
    float3* d_voxels;

    cudaMalloc(&d_count,sizeof(float));
    cudaMalloc(&d_voxels,MAX_VOXELS * sizeof(float3));
    cudaMemcpy(d_count,&count,sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxels,host_voxels,MAX_VOXELS * sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy(host_voxels, d_voxels, MAX_VOXELS * sizeof(float3), cudaMemcpyDeviceToHost);
    extract_pcd(d_hashdata,d_voxels,d_count);
    cudaMemcpy(&count,d_count,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_voxels, d_voxels, MAX_VOXELS * sizeof(float3), cudaMemcpyDeviceToHost);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (int i = 0; i < count; ++i) {
        pcl::PointXYZ point;
        point.x = host_voxels[i].x;
        point.y = host_voxels[i].y;
        point.z = host_voxels[i].z;
        cloud.push_back(point);
    }

    pcl::io::savePCDFileASCII("/home/hmy/ws_fast_lio/src/pcd_receiver/PCD/voxels.pcd", cloud);
    ROS_INFO("Saved %d data points to voxels.pcd", cloud.size());

    return 0;
}