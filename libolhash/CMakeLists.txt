include(FindSSE2)

#if (NOT HAVE_SSE2)
  set(SOURCES
    olhash.h
    olhash.cpp
	  blake2.h
    ref/blake2-impl.h
    ref/blake2b-ref.c
    )
#else ()
#  set(SOURCES
#    olhash.h
#    olhash.cpp
#    blake2.h
#    sse/blake2-config.h
#    sse/blake2-impl.h
#    sse/blake2b-load-sse2.h
#    sse/blake2b-load-sse41.h
#    sse/blake2b-round.h
#    sse/blake2b.c
#    )
#endif()
  

hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

add_library(olhash ${SOURCES})

#include(FindSSE2)
#if (NOT SSE2_SUPPORT_FOUND)

target_link_libraries(olhash PRIVATE devcore olminer-buildinfo Boost::system Boost::filesystem jsoncpp_lib_static OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(olhash PRIVATE ..)
