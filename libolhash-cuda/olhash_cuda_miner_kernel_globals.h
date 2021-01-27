#pragma once

__constant__ uint8_t d_num_to_code[16] = {48,49,50,51,52,53,54,55,56,57,97,98,99,100,101,102};
__constant__ uint8_t d_work[64];
__constant__ uint8_t d_miner_key[64];
__constant__ uint8_t d_merkle_root[64];
__constant__ uint8_t d_timestamp[64];
__constant__ uint64_t d_miner_key_length;
__constant__ uint64_t d_timestamp_length;
__constant__ uint64_t d_target;

#if (__CUDACC_VER_MAJOR__ > 8)
#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))
#else
#define SHFL(x, y, z) __shfl((x), (y), (z))
#endif

#if (__CUDA_ARCH__ >= 320)
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif
