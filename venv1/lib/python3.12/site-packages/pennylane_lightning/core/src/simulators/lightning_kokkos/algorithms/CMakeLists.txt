cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_algorithms LANGUAGES CXX)

set(ALGORITHMS_FILES AlgorithmsKokkos.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos_algorithms STATIC ${ALGORITHMS_FILES})

target_include_directories(lightning_kokkos_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_kokkos_algorithms PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(lightning_kokkos_algorithms PUBLIC    lightning_kokkos_utils
                                                            lightning_algorithms
                                                            lightning_kokkos
                                                            lightning_kokkos_observables
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()