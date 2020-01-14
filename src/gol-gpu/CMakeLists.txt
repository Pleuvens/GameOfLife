add_library(libgol-gpu
    gol-gpu.cu
)

set_target_properties(libgol-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(libgol-gpu PRIVATE
    ncurses
    pthread
)

add_executable(gol-gpu
    main.cc
)

set_target_properties(gol-gpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(gol-gpu PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(gol-gpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(gol-gpu PRIVATE
    libgol-gpu
)
