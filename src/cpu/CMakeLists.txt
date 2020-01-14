add_library(libgol-cpu
    gol-cpu.cc
    gol-cpu.hh
    map.cc
    map.hh
)

set_target_properties(libgol-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

target_compile_options(libgol-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(libgol-cpu PRIVATE
    ncurses
    pthread
    tbb
)

add_executable(gol-cpu
    main.cc
)

set_target_properties(gol-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(gol-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(gol-cpu PRIVATE
    libgol-cpu
)

add_executable(gol-cpu-parallel
    gol-cpu-parallel.cc
)

set_target_properties(gol-cpu-parallel PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(gol-cpu-parallel PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(gol-cpu-parallel PRIVATE
    libgol-cpu
)
