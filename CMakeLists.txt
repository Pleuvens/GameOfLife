cmake_minimum_required (VERSION 3.14)
project (GPGPU LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH {CMAKE_MODULE_PATH} {PROJECT_SOURCE_DIR})
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)

include(FetchContent)

FetchContent_Declare(
    GoogleBenchmark
    URL https://github.com/google/benchmark/archive/v1.5.0.tar.gz
)

if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

add_subdirectory(src/bench)
add_subdirectory(src/gol-cpu)
add_subdirectory(src/bit-cpu)
add_subdirectory(src/gol-gpu)
