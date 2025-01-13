#include <VoxelHash.h>
#include <yaml-cpp/yaml.h>

__constant__ HashParams c_hashParams;

#define SDF_BLOCK_SIZE ((c_hashParams.m_SDFBlockSize))

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif


#include <cutil_math.h>
#include "MarchingCubesSDFUtil.h"
#include "CUDAMarchingCubesHashSDF.h"

#define T_PER_BLOCK 8

#define cudaCheckError(err) if (err != cudaSuccess) { printf("CUDA error: %s, line: %d, file: %s\n", cudaGetErrorString(err), __LINE__, __FILE__); }

void extract_pcd(HashData* hash,float3* d_voxels,float* count)
{
    dim3 blockSize(1024);
    dim3 gridSize((2000000*10 + blockSize.x-1) / blockSize.x);
    extract_pcd_kernel<<<gridSize,blockSize>>>(hash,d_voxels,count);
    cudaCheckError(cudaDeviceSynchronize());
}

__global__ void extract_pcd_kernel(HashData* hash,float3* d_voxels,float* count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2000000 * 10) return;
    if (hash->d_hash[idx].ptr != FREE_ENTRY)
    {
        for (int i = 0; i < 512; i++)
        {
            if (hash->d_SDFBlocks[hash->d_hash[idx].ptr+i].weight_sum != 0)
            {
                uint3 uvoxellocalpos = hash->delinearizeVoxelIndex(i);
                int3 voxellocalpos = make_int3(uvoxellocalpos.x,uvoxellocalpos.y,uvoxellocalpos.z);
                int3 voxelpos = voxellocalpos+hash->d_hash[idx].pos;
                float3 worldpos = hash->virtualVoxelPosToWorld(voxelpos);
                float a = atomicAdd(count,1.0f) + 0.001;
                d_voxels[(int)a]=worldpos;
            }
        }
    }
}

__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
	*data.d_numOccupiedBlocks = 0;	
}
 
extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);
	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);
	cudaCheckError(cudaDeviceSynchronize());
}

__global__ void extractIsoSurfaceKernel(HashData hashData, MarchingCubesData data) 
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData);
	}
}

extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, data);
	cudaCheckError(cudaDeviceSynchronize());
}

__global__ void extractIsoSurfacePass1Kernel(HashData hashData, MarchingCubesData data)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x;

	if (bucketID < hashParams.m_hashNumBuckets*hashParams.m_hashBucketSize) {
		HashEntry& entry = hashData.d_hash[bucketID];
		if (entry.ptr != FREE_ENTRY) {
			uint addr = atomicAdd(&data.d_numOccupiedBlocks[0], 1);
			data.d_occupiedBlocks[addr] = bucketID;
			// printf("extractIsoSurfacePass1Kernel: bucketID: %d, addr: %d\n", bucketID, addr);
		}
	}
}

extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	// printf("extractIsoSurfacePass1CUDA start\n");
	const dim3 gridSize((params.m_hashNumBuckets*params.m_hashBucketSize + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);
	// printf("extractIsoSurfacePass1Kernel start\n");
	extractIsoSurfacePass1Kernel<<<gridSize, blockSize>>>(hashData, data);
	// printf("extractIsoSurfacePass1Kernel end\n");
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaDeviceSynchronize());
}

__global__ void extractIsoSurfacePass2Kernel(HashData hashData, MarchingCubesData data)
{
	uint idx = data.d_occupiedBlocks[blockIdx.x];
	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData);
	}
}

extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks)
{
	const dim3 gridSize(numOccupiedBlocks, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	if (numOccupiedBlocks) {
		extractIsoSurfacePass2Kernel << <gridSize, blockSize >> >(hashData, data);
		cudaCheckError(cudaDeviceSynchronize());
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

#define CUDA_ERROR_CHECK(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(1); \
        } \
    } while (0)

__global__ void updatesdfframe(HashData* hash, float3* worldpos, float3* normal, int numPoints) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numPoints) {
		hash->insertHashEntryElement(worldpos[idx]);
		HashEntry curr = hash->getHashEntryForWorldPos(worldpos[idx]);
		int num = hash->getNumHashEntriesPerBucket(hash->computeHashPos(worldpos[idx]));
		if (num == c_hashParams.m_hashBucketSize) {
			printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nHash bucket size is full\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
		__threadfence();
		if(curr.ptr != FREE_ENTRY)
		{
			if (isnan(normal[idx].x) || isnan(normal[idx].y) || isnan(normal[idx].z)) {
				return;
			}
			float sdf = hash->computesdf(worldpos[idx], normal[idx]);
			// printf("sdf: %f\n", sdf);
			Voxel* voxel = hash->getVoxel(worldpos[idx]);
			atomicAdd(&voxel->sdf_sum, sdf);
			atomicAdd((int*)&(voxel->weight_sum), 1);
			// printf("voxel: %f\n", voxel->sdf_sum / (int)voxel->weight_sum);
		}
	}
}

// DONE: no problems for now. initialize hash entry storage space parallelly. 
// with pose set to (0, 0, 0) and ptr set to FREE_ENTRY.
// offset is set to 0 for there are no collisions for now.
__global__ void initializeHashEntry(HashEntry* d_hash, int hashNumBuckets, int hashBucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < hashNumBuckets * hashBucketSize) {
		d_hash[idx].pos = make_int3(0, 0, 0);
		d_hash[idx].ptr = FREE_ENTRY;
		d_hash[idx].offset = 0;
	}
}

// DONE: no problems for now. initialize heap to be the same number of elements as the SDF blocks.
// d_heap is a remapping of the SDF blocks to avoid fragmentation.
// it is initialized to be the same number of elements as the SDF blocks.
// and the remapping is set to direct mapping at the beginning.
__global__ void initializeHeap(unsigned int* d_heap, unsigned int numSDFBlocks) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numSDFBlocks) {
		d_heap[idx] = idx;
	}
}

// DONE: no problems for now. initialize heap counter to be the same number of elements as the SDF blocks.
// the counter for free heap is initialized value, by one single thread.
// consumeHeap will decrement, and appendHeap will increment. -> thus d_heapCounter should be initialized to numSDFBlocks - 1.
// it means that, from d_heap[0] to d_heap[numSDFBlocks - 1] (including d_heap[numSDFBlocks - 1]), all the elements are free.
__global__ void initializeHeapCounter(unsigned int* d_heapCounter, unsigned int value) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == 0) {
		d_heapCounter[0] = value;
	}
}

// DONE: initialize hash bucket mutex parallelly.
// the mutex is used to avoid race condition when inserting hash entry.
// the mutex is initialized to be UNLOCK_ENTRY.
__global__ void initializeHashBucketMutex(int* d_hashBucketMutex, int numBuckets) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numBuckets) {
		d_hashBucketMutex[idx] = UNLOCK_ENTRY;
	}
}

// DONE: read from config file, and do initializeHashParams.
__host__ void HashData::initializeHashParams(HashParams& params, const std::string& config_file) {
	YAML::Node config = YAML::LoadFile(config_file);
	params.m_hashNumBuckets = config["hashNumBuckets"].as<int>();
	params.m_hashBucketSize = config["hashBucketSize"].as<int>();
	params.m_hashMaxCollisionLinkedListSize = config["hashMaxCollisionLinkedListSize"].as<int>();
	params.m_numSDFBlocks = config["numSDFBlocks"].as<int>();
	params.m_SDFBlockSize = config["SDFBlockSize"].as<int>();
	params.m_virtualVoxelSize = config["virtualVoxelSize"].as<float>();
	params.m_numOccupiedBlocks = config["numOccupiedBlocks"].as<int>();
	params.m_maxIntegrationDistance = config["maxIntegrationDistance"].as<float>();
	params.m_truncScale = config["truncScale"].as<float>();
	params.m_truncation = config["truncation"].as<float>();
	params.m_integrationWeightSample = config["integrationWeightSample"].as<float>();
	params.m_integrationWeightMax = config["integrationWeightMax"].as<float>();
}

// DONE: no problems for now.
__host__ void HashData::allocate(bool dataOnGPU) {
	HashParams params;
	initializeHashParams(params, "/home/hmy/ws_fast_lio/src/pcd_receiver/config/voxel_hashing.yaml");
	m_bIsOnGPU = dataOnGPU;
	if (m_bIsOnGPU) {
		// allocate memory for heap, heap counter.
		CUDA_ERROR_CHECK(cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numSDFBlocks));
		CUDA_ERROR_CHECK(cudaMalloc(&d_heapCounter, sizeof(unsigned int)));

		// initialize heap parallelly.
		int numThreads = 1024;
		int numBlocks = (params.m_numSDFBlocks + numThreads - 1) / numThreads;
		initializeHeap<<<numBlocks, numThreads>>>(d_heap, params.m_numSDFBlocks);
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());

		// initialize heap counter parallelly.
		initializeHeapCounter<<<1, 1>>>(d_heapCounter, params.m_numSDFBlocks - 1); // only one thread is needed to initialize the heap counter.
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());

		// allocate memory for hash entry storage space.
		CUDA_ERROR_CHECK(cudaMalloc(&d_hash, sizeof(HashEntry) * params.m_hashNumBuckets * params.m_hashBucketSize));

		// initialize hash entry storage space parallelly.
		numThreads = 1024;
		numBlocks = (params.m_hashNumBuckets * params.m_hashBucketSize + numThreads - 1) / numThreads;
		initializeHashEntry<<<numBlocks, numThreads>>>(d_hash, params.m_hashNumBuckets, params.m_hashBucketSize);
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());

		// allocate memory for SDF blocks.
		CUDA_ERROR_CHECK(cudaMalloc(&d_SDFBlocks, sizeof(Voxel) * params.m_SDFBlockSize * params.m_SDFBlockSize * params.m_SDFBlockSize * params.m_numSDFBlocks));

		// allocate memory for hash bucket mutex. Each bucket has a mutex to avoid race condition.
		CUDA_ERROR_CHECK(cudaMalloc(&d_hashBucketMutex, sizeof(int) * params.m_hashNumBuckets));

		// initialize hash bucket mutex parallelly.
		numThreads = 1024;
		numBlocks = (params.m_hashNumBuckets + numThreads - 1) / numThreads;
		initializeHashBucketMutex<<<numBlocks, numThreads>>>(d_hashBucketMutex, params.m_hashNumBuckets);
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
	} else {
		d_heap = new unsigned int[params.m_numSDFBlocks];
		d_heapCounter = new unsigned int[1];
		d_hash = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
		d_SDFBlocks = new Voxel[params.m_numSDFBlocks * params.m_SDFBlockSize * params.m_SDFBlockSize * params.m_SDFBlockSize];
		d_hashBucketMutex = new int[params.m_hashNumBuckets];
	}

	updateParams(params); // use this function to update the c_hashParams.
}

// DONE: no problems for now.
__host__ void HashData::updateParams(const HashParams& params) {
	if (m_bIsOnGPU) {
		size_t size;
		CUDA_ERROR_CHECK(cudaGetSymbolSize(&size, reinterpret_cast<const void*>(&c_hashParams)));
		CUDA_ERROR_CHECK(cudaMemcpyToSymbol(reinterpret_cast<const void*>(&c_hashParams), &params, size, 0, cudaMemcpyHostToDevice));
	}
}

#ifdef __CUDACC__

__device__ bool HashData::insertHashEntryElement(const float3& WorldPos) {
	uint h = computeHashPos(WorldPos);
	uint hp = h * c_hashParams.m_hashBucketSize;
	int3 pos = worldToSDFBlock(WorldPos);
	int firstEmpty = -1; // indicate the first empty hash entry in the bucket for insertion.
	for (uint j = 0; j < c_hashParams.m_hashBucketSize; j++) {
		uint i = j + hp;
		const HashEntry& curr = d_hash[i];
		if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
			return true;
		}
		// if the first empty hash entry is not found yet, and the current hash entry is free, set the first empty hash entry to the current hash entry.
		if (firstEmpty == -1 && curr.ptr == FREE_ENTRY) {
			firstEmpty = i;
		}
	}
	if (firstEmpty != -1) {	//if there is an empty entry and we haven't allocated the current entry before
		int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
		if (prevValue != LOCK_ENTRY) {	//only proceed if the bucket has been locked
			HashEntry& entry = d_hash[firstEmpty];
			entry.pos = pos;
			entry.offset = NO_OFFSET;		
			entry.ptr = consumeHeap() * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;	//memory alloc
			atomicExch(&d_hashBucketMutex[h], FREE_ENTRY);
			return true;
		}
		atomicExch(&d_hashBucketMutex[h], FREE_ENTRY);
	}
	return false;
}

// DONE: no problems for now. Calculate per-points signed distance to the surface.
__device__ float HashData::computesdf(float3 worldpos, float3 normal) {
	//printf("normal: %f %f %f\n", normal.x, normal.y, normal.z);
	int3 voxelpos = worldToVirtualVoxelPos(worldpos);
	float3 pos_center; // the center of the voxel.
	// note that the virtual voxel is already the center of the voxel, 
	// just multiply the voxelpos by the virtual voxel size as a scale.
	pos_center.x = voxelpos.x * c_hashParams.m_virtualVoxelSize;
	pos_center.y = voxelpos.y * c_hashParams.m_virtualVoxelSize;
	pos_center.z = voxelpos.z * c_hashParams.m_virtualVoxelSize;
	pos_center = worldpos - pos_center;

	return pos_center.x * normal.x + pos_center.y * normal.y + pos_center.z * normal.z;
}

// DONE: no problems for now. Compute the hash position of the voxel.
__device__ uint HashData::computeHashPos(const float3& WorldPos) const {
	int3 sdfblockPos = worldToSDFBlock(WorldPos);
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;
	int res = ((sdfblockPos.x * p0) ^ (sdfblockPos.y * p1) ^ (sdfblockPos.z * p2)) % c_hashParams.m_hashNumBuckets;
	if (res < 0) res += c_hashParams.m_hashNumBuckets;
	return (uint)res;
}

// note that the virtual voxel pos corresponds to the center point of the voxel.
__device__ int3 HashData::worldToVirtualVoxelPos(const float3& pos) const {
	const float3 p = pos / c_hashParams.m_virtualVoxelSize;
	return make_int3(p + make_float3(sign(p)) * 0.5f);
}

// DONE: do not change this function. The original implementation is correct.
// and the return value will be the SDF block index.
__device__ int3 HashData::virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const {
	// avoid duplication mapping caused by the negative rounding error.
	if (virtualVoxelPos.x < 0) virtualVoxelPos.x -= SDF_BLOCK_SIZE - 1;
	if (virtualVoxelPos.y < 0) virtualVoxelPos.y -= SDF_BLOCK_SIZE - 1;
	if (virtualVoxelPos.z < 0) virtualVoxelPos.z -= SDF_BLOCK_SIZE - 1;

	return make_int3(
		virtualVoxelPos.x / SDF_BLOCK_SIZE,
		virtualVoxelPos.y / SDF_BLOCK_SIZE,
		virtualVoxelPos.z / SDF_BLOCK_SIZE);
}

// DONE: do not change this function. The original implementation is correct.
// and the return value will be the virtual voxel position. -> the left back bottom corner of the voxel.
__device__ int3 HashData::SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const {
	return sdfBlock * SDF_BLOCK_SIZE;
}

// DONE: do not change this function. The original implementation is correct.
// use the virtual voxel index as center point, and then multiply the virtual voxel size as a scale to get world position.
__device__ float3 HashData::virtualVoxelPosToWorld(const int3& pos) const {
	return make_float3(pos) * c_hashParams.m_virtualVoxelSize;
}

// transfer from the SDF index to left back bottom corner voxel index, 
// and then transfer to world position 
// by the scale using center point correspondence of voxel and world position.
__device__ float3 HashData::SDFBlockToWorld(const int3& sdfBlock) const {
	return virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock));
}

// DONE: do not change this function. The original implementation is correct.
// transfer from the world position to the SDF block index.
__device__ int3 HashData::worldToSDFBlock(const float3& worldPos) const {
	return virtualVoxelPosToSDFBlock(worldToVirtualVoxelPos(worldPos));
}

// DONE: do not change this function. The original implementation is correct.
// transfer from the linear voxel index to the voxel index.
__device__ uint3 HashData::delinearizeVoxelIndex(uint idx) const {
	uint x = idx % SDF_BLOCK_SIZE;
	uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
	uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
	return make_uint3(x, y, z);
}

// DONE: do not change this function. The original implementation is correct.
// transfer from the voxel index to the linear voxel index.
__device__ uint HashData::linearizeVoxelPos(const int3& virtualVoxelPos) const {
	return
		virtualVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
		virtualVoxelPos.y * SDF_BLOCK_SIZE +
		virtualVoxelPos.x;
}

// DONE: do not change this function. The original implementation is correct.
// transfer from the world position to the local SDF block index.
__device__ int HashData::WorldPosToLocalSDFBlockIndex(const float3& WorldPos) const {
	int3 virtualVoxelPos = worldToVirtualVoxelPos(WorldPos);
	int3 localVoxelPos = make_int3(
		virtualVoxelPos.x % SDF_BLOCK_SIZE,
		virtualVoxelPos.y % SDF_BLOCK_SIZE,
		virtualVoxelPos.z % SDF_BLOCK_SIZE);

	if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
	if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
	if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

	return linearizeVoxelPos(localVoxelPos);
}

__device__ Voxel* HashData::getVoxel(const float3& WorldPos) const {
	HashEntry hashEntry = getHashEntryForWorldPos(WorldPos);
	if (hashEntry.ptr == FREE_ENTRY) {
		return NULL;
	}
	return &d_SDFBlocks[hashEntry.ptr + WorldPosToLocalSDFBlockIndex(WorldPos)];
}

// TODO: collision not considered for now.
__device__ HashEntry HashData::getHashEntryForWorldPos(const float3& WorldPos) const {
	uint h = computeHashPos(WorldPos);
	uint hp = h * c_hashParams.m_hashBucketSize;
	int3 sdfBlock = worldToSDFBlock(WorldPos);
	HashEntry entry;
	entry.pos = sdfBlock;
	entry.offset = 0;
	entry.ptr = FREE_ENTRY;

	for (uint j = 0; j < c_hashParams.m_hashBucketSize; j++) {
		uint i = j + hp;
		HashEntry curr = d_hash[i];
		if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
			return curr;
		}
	}
	return entry; // if not found, return the entry with FREE_ENTRY. No collision is considered for now.
}

// DONE: not used for now. But can be used to check the number of hash entries per bucket.
__device__ unsigned int HashData::getNumHashEntriesPerBucket(unsigned int bucketID) {
	unsigned int h = 0;
	for (uint i = 0; i < c_hashParams.m_hashBucketSize; i++) {
		if (d_hash[bucketID * c_hashParams.m_hashBucketSize + i].ptr != FREE_ENTRY) {
			h++;
		}
	}
	return h;
}

// DONE: consume the heap, and return the index of the SDF block.
__device__ uint HashData::consumeHeap() {
	uint addr = atomicSub(&d_heapCounter[0], 1);
	return d_heap[addr];
}

// DONE: append the heap, and set the index of the SDF block.
__device__ void HashData::appendHeap(uint ptr) {
	uint addr = atomicAdd(&d_heapCounter[0], 1);
	d_heap[addr + 1] = ptr;
}

#endif