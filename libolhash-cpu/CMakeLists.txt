file(GLOB sources "*.cpp")
file(GLOB headers "*.h")

add_library(olhash-cpu ${sources} ${headers})
#target_link_libraries(olhash-cpu ethcore ethash::ethash Boost::fiber Boost::thread)
target_link_libraries(olhash-cpu olhash ethcore ethash::ethash Boost::thread)
target_include_directories(olhash-cpu PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
