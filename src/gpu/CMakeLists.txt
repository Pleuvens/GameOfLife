add_executable(gol-gpu
    gol-gpu.cu
)

set_target_properties(gol-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(gol-gpu PRIVATE
    ncurses
    pthread
)
