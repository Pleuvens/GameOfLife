add_library(libgol-cpu
    gol-cpu.cc
    map.cc
    map.hh
    ../gui/callbacks.cc
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
    pthread
    tbb
)

target_include_directories(libgol-cpu PUBLIC
    "../gui"
)

add_executable(gol-cpu
    main.cc
)

set_target_properties(gol-cpu PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(gol-cpu PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(gol-cpu PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(gol-cpu PRIVATE
    libgol-cpu
)


target_include_directories(gol-cpu PUBLIC
    "../gui"
)

add_executable(gol-cpu-parallel
    gol-cpu-parallel.cc
)

set_target_properties(gol-cpu-parallel PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(gol-cpu-parallel PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(gol-cpu-parallel PRIVATE
    "-Wall"
    "-Wextra"
    "-pedantic"
)

target_link_libraries(gol-cpu-parallel PRIVATE
    libgol-cpu
)

target_include_directories(gol-cpu-parallel PUBLIC
    "../gui"
)
