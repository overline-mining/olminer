#include "olhash_cuda_miner_kernel.h"

#include "olhash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }

#include "olhash.cuh"

__global__ void olhash_search(volatile Search_results* g_output, uint64_t start_nonce)
{
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t distance;
    if (compute_distance(start_nonce + gid, &distance))
        return;
    uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
    if (index >= MAX_SEARCH_RESULTS)
        return;
    g_output->result[index].gid = gid;
    g_output->result[index].distance = distance;
}

void run_olhash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
                       volatile Search_results* g_output, uint64_t start_nonce)
{
    olhash_search<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
    CUDA_SAFE_CALL(cudaGetLastError());
}

void set_common_data(const std::vector<uint8_t>& _work,
                     const std::vector<uint8_t>& _miner_key,
                     const std::vector<uint8_t>& _merkle_root,
                     const std::vector<uint8_t>& _timestamp)
{
  const uint64_t miner_key_size = _miner_key.size();
  const uint64_t timestamp_size = _timestamp.size();
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_work, _work.data(), 64));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_miner_key, _miner_key.data(), _miner_key.size()));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_miner_key_length, &miner_key_size, sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_merkle_root, _merkle_root.data(), 64));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_timestamp, _timestamp.data(), 64));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_timestamp_length, &timestamp_size, sizeof(uint64_t)));
}

void set_target(uint64_t _target)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &_target, sizeof(uint64_t)));
}
