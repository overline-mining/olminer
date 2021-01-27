#include "ethash_cuda_miner_kernel_globals.h"

#include "ethash_cuda_miner_kernel.h"

#include "cuda_helper.h"

#include "blake2b.cu"
    
#define _PARALLEL_HASH 4

DEV_INLINE uint64_t cosine_distance(uint8_t soln[BLAKE2B_OUTBYTES]) {
  double acc(0), num(0), den(0), norm_w(0), norm_s(0);
  for( unsigned i = 0; i < BLAKE2B_OUTBYTES/32; ++i ) {
    num = 0.; den = 0.; norm_w = 0.; norm_s = 0.;
    for( unsigned j = 0; j < 32; ++j ) {
      unsigned w = d_work[32 * (1 - i) + j];
      unsigned s = soln[32 * i + j];
      num += w * s;
      norm_w += w * w;
      norm_s += s * s;
    }
    den = std::sqrt(norm_w) * std::sqrt(norm_s);
    acc += (1.0 - num / den);
  }
  return uint64_t(acc * 1000000000000000ULL);
}

DEV_INLINE void blake2bl_from_bytes(uint8_t in[BLAKE2B_OUTBYTES],
                                    uint8_t out[BLAKE2B_OUTBYTES]) {
  #pragma unroll
  for(unsigned j = 32; j < BLAKE2B_OUTBYTES; ++j) {
    uint8_t byte = in[j];
    out[2 * (j - 32)] = d_num_to_code[byte >> 4];
    out[2 * (j - 32) + 1] = d_num_to_code[byte & 0xf];
  }   
}

DEV_INLINE bool compute_distance(uint64_t nonce, uint64_t* distance) {
  uint8_t hash_string[4*BLAKE2B_OUTBYTES];
  uint8_t work_string[BLAKE2B_OUTBYTES];

  const unsigned nonce_hash_offset = d_miner_key_length + BLAKE2B_OUTBYTES;
  const unsigned ts_offset = nonce_hash_offset + BLAKE2B_OUTBYTES;
  const unsigned hash_string_length = ts_offset + d_timestamp_length;
  
  // bring in the parts of work to the string-to-hash
  memcpy(hash_string, d_miner_key, d_miner_key_length);
  memcpy(hash_string + d_miner_key_length, d_merkle_root, BLAKE2B_OUTBYTES);
  // skip past the area for the nonce hash to place the timestamp string
  memcpy(hash_string + ts_offset, d_timestamp, d_timestamp_length);

  //convert the nonce to a string
  work_string[0] = '0';
  uint8_t length = 0;
  uint64_t reduced_nonce = nonce;
  while(reduced_nonce > 0) {
    ++length;
    reduced_nonce /= 10ULL;
  }
  reduced_nonce = nonce;
  for(uint64_t j = length; j > 1; --j) {
    work_string[j - 1] = d_num_to_code[reduced_nonce % 10];
    reduced_nonce /= 10ULL;
  }
  work_string[0] = d_num_to_code[reduced_nonce];
  length = (length == 0) + (length > 0) * length;

  // calculate the hash of the nonce
  // reuse the work_string to hold the hash
  blake2b_state ns;
  blake2b_init_cu(&ns, BLAKE2B_OUTBYTES);
  blake2b_update_cu(&ns, work_string, length);
  blake2b_final_cu(&ns, work_string, BLAKE2B_OUTBYTES);

  // "blake2b" and stringify the nonce hash
  // place directly into string-to-hash
  blake2bl_from_bytes(work_string, hash_string + nonce_hash_offset);

  //reset blake2b state and hash hash_string into work_string
  blake2b_init_cu(&ns, BLAKE2B_OUTBYTES);
  blake2b_update_cu(&ns, hash_string, hash_string_length);
  blake2b_final_cu(&ns, work_string, BLAKE2B_OUTBYTES);

  //blake2bl the work string
  blake2bl_from_bytes(work_string, work_string);
  
  //d_work is global, no need to pass!
  distance[0] = cosine_distance(work_string);

  //printf("GPU distance -> %llu, difficulty -> %llu\n", distance[0], d_target);
  
  if(distance[0] <= d_target) {
    return true;
  }
  return false;    
}

DEV_INLINE bool compute_hash(uint64_t nonce, uint2* mix_hash)
{
    // sha3_512(header .. nonce)
    uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state);

    // Threads work together in this phase in groups of 8.
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    const int mix_idx = thread_id & 3;

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = SHFL(state[j].x, i + p, THREADS_PER_HASH);
                shuffle[j].y = SHFL(state[j].y, i + p, THREADS_PER_HASH);
            }
            switch (mix_idx)
            {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        for (uint32_t a = 0; a < ACCESSES; a += 4)
        {
            int t = bfe(a, 2u, 3u);

            for (uint32_t b = 0; b < 4; b++)
            {
                for (int p = 0; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
                    offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    // keccak_256(keccak_512(header..nonce) .. mix);
    if (cuda_swab64(keccak_f1600_final(state)) > d_target)
        return true;

    mix_hash[0] = state[8];
    mix_hash[1] = state[9];
    mix_hash[2] = state[10];
    mix_hash[3] = state[11];

    return false;
}
