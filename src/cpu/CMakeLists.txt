add_executable(gol-cpu
    gol-cpu.cc
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
