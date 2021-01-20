#include "olhash.h"
#include <sstream>
#include <chrono>

namespace dev
{
namespace ol
{

//input is actually an ascii encoded string
h256 blake2bl_from_bytes(const bytes& data) {
  blake2b_state state;
  byte out[BLAKE2B_OUTBYTES];
  blake2b_init(&state, BLAKE2B_OUTBYTES);
  blake2b_update(&state, data.data(), data.size());
  blake2b_final(&state, out, BLAKE2B_OUTBYTES);
  // "blake2bl" is just the upper 32 bytes of blake2b
  byte const* offset = out + BLAKE2B_OUTBYTES/2;
  return h256(offset, h256::ConstructFromPointer);
}

// here we assume that left and right are 64-byte long hex-strings
// i.e. the string representation of 32 byte blake2bl output
uint64_t calc_distance(const bytes& work, const bytes& soln) {
  double acc(0), num(0), den(0), norm_w(0), norm_s(0);
  for( unsigned i = 0; i < work.size()/32; ++i ) {
    num = 0.; den = 0.; norm_w = 0.; norm_s = 0.;
    for( unsigned j = 0; j < 32; ++j ) {
      unsigned w = work[32 * (1 - i) + j];
      unsigned s = soln[32 * i + j];
      num += w * s;
      norm_w += w * w;
      norm_s += s * s;
    }
    den = std::sqrt(norm_w) * std::sqrt(norm_s);
    acc += (1.0 - num / den);
  }
  return uint64_t(acc*1000000000000000ULL);
}

uint64_t eval(const bytes& work,
              const bytes& miner_key,
              const bytes& merkle_root,
              const bytes& timestamp,
              uint64_t nonce,
              bool print_tohash) {
  std::string nonce_str = std::to_string(nonce);
  bytes nonce_bytes(nonce_str.begin(), nonce_str.end());

  std::string nonce_hash_str = blake2bl_from_bytes(nonce_bytes).hex();

  bytes nonce_hash(nonce_hash_str.begin(), nonce_hash_str.end());
  bytes tohash(miner_key.begin(), miner_key.end());
  tohash.insert(tohash.end(), merkle_root.begin(), merkle_root.end());
  tohash.insert(tohash.end(), nonce_hash.begin(), nonce_hash.end());
  tohash.insert(tohash.end(), timestamp.begin(), timestamp.end());

  if(print_tohash) {
    std::cout << "tohash -> " << std::string(tohash.begin(), tohash.end()) << ' ' << tohash.size() << std::endl;
  }
  
  std::string hash_str = blake2bl_from_bytes(tohash).hex();
  bytes guess(hash_str.begin(), hash_str.end());

  return calc_distance(work, guess);
}

search_result search(const bytes& work,
                     const bytes& miner_key,
                     const bytes& merkle_root,
                     uint64_t timestamp,
                     uint64_t difficulty,
                     uint64_t nonce,
                     size_t iterations) {

  const size_t whenStop = nonce + iterations;
  uint64_t best_nonce = 0;
  olhash_result best_result{0, 0};
  std::string stimestamp = std::to_string(timestamp);
  bytes btimestamp(stimestamp.begin(), stimestamp.end());
  for( uint64_t i_nonce = nonce; i_nonce < whenStop; ++i_nonce ) {

    uint64_t distance = eval(work, miner_key, merkle_root, btimestamp, i_nonce);
    if( distance > best_result.distance ) {
      best_result.distance = distance;
      best_result.timestamp = timestamp;
      best_nonce = i_nonce;      
    }                                 
  }

  if( best_result.distance > difficulty ) {
    return search_result{best_result, best_nonce};
  } else {
    return search_result();
  }
}

} // ol
} // dev
