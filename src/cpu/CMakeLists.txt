add_library(libgol-cpu
    gol-cpu.cc
    gol-cpu.hh
    map.cc
    map.hh
)

add_executable(gol-cpu
    main.cc
    map.cc
    map.hh
)

add_executable(gol-cpu-parallel
    gol-cpu-parallel.cc
    map.cc
    map.hh
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
    ncurses
    pthread
    tbb
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
    ncurses
    pthread
)
