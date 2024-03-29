cmake_minimum_required(VERSION 3.5)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)
include(HunterGate)

include(defaults/HunterCacheServers)

cable_configure_toolchain(DEFAULT cxx11)

set(HUNTER_CONFIGURATION_TYPES Release CACHE STRING "Build type of Hunter packages")
set(HUNTER_JOBS_NUMBER 6 CACHE STRING "Number of parallel builds used by Hunter")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.300.tar.gz"
    SHA1 "1151d539465d9cdbc880ee30f794864aec11c448"
    LOCAL
)

project(olminer)
set(PROJECT_VERSION 0.20.0)

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo)

option(ETHASHCL "Build with OpenCL mining" OFF) # NOT AVAILABLE YET DO NOT TURN ON
option(OLHASHCUDA "Build with CUDA mining" ON)
option(OLHASHCPU "Build with CPU mining" ON)
option(ETHDBUS "Build with D-Bus support" OFF)
option(APICORE "Build with API Server support" ON)
option(BINKERN "Install AMD binary kernels" ON)
option(DEVBUILD "Log developer metrics" OFF)
option(USE_SYS_OPENCL "Build with system OpenCL" OFF) # DO NOT TURN ON

# propagates CMake configuration options to the compiler
function(configureProject)
    if (ETHASHCL)
        add_definitions(-DETH_ETHASHCL)
    endif()
    if (OLHASHCUDA)
        add_definitions(-DOL_OLHASHCUDA)
    endif()
    if (OLHASHCPU)
        add_definitions(-DOL_OLHASHCPU)
    endif()
    if (ETHDBUS)
        add_definitions(-DETH_DBUS)
    endif()
    if (APICORE)
        add_definitions(-DAPI_CORE)
    endif()
    if (BINKERN)
        add_definitions(-DBIN_KERN)
    endif()
    if (DEVBUILD)
        add_definitions(-DDEV_BUILD)
    endif()
    if (USE_SYS_OPENCL)
        add_definitions(-DUSE_SYS_OPENCL)
    endif()
endfunction()

hunter_add_package(Boost COMPONENTS system filesystem thread)
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem thread)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(ethash)
find_package(ethash CONFIG REQUIRED)

configureProject()

if(APPLE)
    set(USE_SYS_OPENCL ON)
endif()

message("----------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("----------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                      ${ETHASHCL}")
message("-- OLHASHCUDA       Build CUDA components                        ${OLHASHCUDA}")
message("-- OLHASHCPU        Build CPU components (only for development)  ${OLHASHCPU}")
message("-- ETHDBUS          Build D-Bus components                       ${ETHDBUS}")
message("-- APICORE          Build API Server components                  ${APICORE}")
message("-- BINKERN          Install AMD binary kernels                   ${BINKERN}")
message("-- DEVBUILD         Build with dev logging                       ${DEVBUILD}")
message("-- USE_SYS_OPENCL   Build with system OpenCL                     ${USE_SYS_OPENCL}")
message("----------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
endif()

cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

add_subdirectory(libolhash)
add_subdirectory(libdevcore)
add_subdirectory(libethcore)
add_subdirectory(libhwmon)
add_subdirectory(libpoolprotocols)
add_subdirectory(hive)

if (ETHASHCL)
    add_subdirectory(libethash-cl)
    if (BINKERN)
        add_subdirectory(libethash-cl/kernels)
    endif()
endif ()
if (OLHASHCUDA)
    add_subdirectory(libolhash-cuda)
endif ()
if (OLHASHCPU)
    add_subdirectory(libolhash-cpu)
endif ()
if (APICORE)
    add_subdirectory(libapicore)
endif()

add_subdirectory(olminer)

if(WIN32)
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)

