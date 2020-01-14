add_library(libbit-cpu
    map.cc
    map.hh
    ../gui/callbacks.cc
)

set_target_properties(libbit-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

target_compile_options(libbit-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(libbit-cpu PRIVATE
    pthread
    tbb
    glfw
    GL
)

target_include_directories(libbit-cpu PUBLIC
    "../gui/"
)

add_executable(bit-cpu
    bit-cpu.cc
)

set_target_properties(bit-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(bit-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(bit-cpu PRIVATE
    libbit-cpu
)

target_include_directories(bit-cpu PUBLIC
    "../gui/"
)

add_executable(bit-cpu-parallel
    bit-cpu-parallel.cc
)

set_target_properties(bit-cpu-parallel PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(bit-cpu-parallel PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(bit-cpu-parallel PRIVATE
    libbit-cpu
)

target_include_directories(bit-cpu-parallel PUBLIC
    "../gui/"
)
