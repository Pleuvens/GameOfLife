add_executable(bench
    bench.cc
)

set_target_properties(bench PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(bench PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(bench PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(bench PRIVATE
   libgol-gpu
   libgol-cpu
   libbit-gpu
   libbit-cpu
   benchmark
)
