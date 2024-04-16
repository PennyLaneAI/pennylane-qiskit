cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_algorithms LANGUAGES CXX)

set(ALGORITHMS_FILES AlgorithmsGPU.cpp CACHE INTERNAL "" FORCE)

if(ENABLE_MPI)
    list(APPEND ALGORITHMS_FILES AlgorithmsGPUMPI.cpp)
endif()

add_library(${PL_BACKEND}_algorithms STATIC ${ALGORITHMS_FILES})

target_include_directories(${PL_BACKEND}_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_algorithms PRIVATE      lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(${PL_BACKEND}_algorithms PUBLIC       ${PL_BACKEND}_utils
                                                            lightning_algorithms
                                                            ${PL_BACKEND}
                                                            ${PL_BACKEND}_observables
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()