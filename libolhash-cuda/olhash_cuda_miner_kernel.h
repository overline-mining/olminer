#pragma once

#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_runtime.h"

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
#define MAX_SEARCH_RESULTS 4U

struct Search_Result
{
    // One word for gid and 8 for mix hash
    uint32_t gid;
    uint64_t distance;
    uint32_t pad[1];  // pad to size power of 2
};

struct Search_results
{
    Search_Result result[MAX_SEARCH_RESULTS];
    uint32_t count = 0;
};

void set_common_data(const std::vector<uint8_t>& _work,
                     const std::vector<uint8_t>& _miner_key,
                     const std::vector<uint8_t>& _merkle_root,
                     const std::vector<uint8_t>& _timestamp);

void set_target(uint64_t _target);

void run_olhash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile Search_results* g_output, uint64_t start_nonce);

struct cuda_runtime_error : public virtual std::runtime_error
{
    cuda_runtime_error(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)                                                              \
    do                                                                                    \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (cudaSuccess != err)                                                           \
        {                                                                                 \
            std::stringstream ss;                                                         \
            ss << "CUDA error in func " << __FUNCTION__ << " at line " << __LINE__ << ' ' \
               << cudaGetErrorString(err);                                                \
            throw cuda_runtime_error(ss.str());                                           \
        }                                                                                 \
    } while (0)
