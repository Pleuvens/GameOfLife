add_library(libbit-gpu
    bit-gpu.cu
)

set_target_properties(libbit-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(libbit-gpu PRIVATE
    ncurses
    pthread
)

add_executable(bit-gpu
    main.cc
)

set_target_properties(bit-gpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(bit-gpu PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(bit-gpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(bit-gpu PRIVATE
    libbit-gpu
)
