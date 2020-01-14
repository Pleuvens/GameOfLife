add_library(liblut-cpu
    lut-cpu.hh
    map.cc
    map.hh
    ../gui/callbacks.cc
)

set_target_properties(liblut-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

target_compile_options(liblut-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(liblut-cpu PRIVATE
    pthread
    tbb
    glfw
    GL
)

target_include_directories(liblut-cpu PUBLIC
    "../gui/"
)

add_executable(lut-cpu
    lut-cpu.cc
)

set_target_properties(lut-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(lut-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(lut-cpu PRIVATE
    liblut-cpu
)

target_include_directories(lut-cpu PUBLIC
    "../gui/"
)

add_executable(lut-cpu-parallel
    lut-cpu-parallel.cc
)

set_target_properties(lut-cpu-parallel PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(lut-cpu-parallel PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(lut-cpu-parallel PRIVATE
    liblut-cpu
)

target_include_directories(lut-cpu-parallel PUBLIC
    "../gui/"
)
