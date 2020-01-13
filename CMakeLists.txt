cmake_minimum_required (VERSION 3.14)
project (GPGPU LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH {CMAKE_MODULE_PATH} {PROJECT_SOURCE_DIR})
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(src/cpu)
add_subdirectory(src/gpu)
