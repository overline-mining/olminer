#include "olhash_cuda_miner_kernel_globals.h"

#include "olhash_cuda_miner_kernel.h"

#include "cuda_helper.h"

#include "blake2b.cu"

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
