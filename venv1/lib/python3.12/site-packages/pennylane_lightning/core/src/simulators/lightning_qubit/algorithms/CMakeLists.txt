cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_algorithms LANGUAGES CXX)

set(ALGORITHMS_FILES AlgorithmsLQubit.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_qubit_algorithms STATIC ${ALGORITHMS_FILES})

target_include_directories(lightning_qubit_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_qubit_algorithms PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(lightning_qubit_algorithms PUBLIC     lightning_qubit_utils
                                                            lightning_algorithms
                                                            lightning_qubit
                                                            lightning_utils
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()