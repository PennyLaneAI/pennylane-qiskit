cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_observables LANGUAGES CXX)

set(OBSERVABLES_FILES ObservablesGPU.cpp CACHE INTERNAL "" FORCE)

if(ENABLE_MPI)
    list(APPEND OBSERVABLES_FILES ObservablesGPUMPI.cpp)
endif()

add_library(${PL_BACKEND}_observables STATIC ${OBSERVABLES_FILES})

target_include_directories(${PL_BACKEND}_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_observables PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(${PL_BACKEND}_observables PUBLIC  lightning_utils
                                                        lightning_gates
                                                        lightning_observables
                                                        ${PL_BACKEND}_utils
                                                        ${PL_BACKEND}_gates
                                                        ${PL_BACKEND}
                                                        )

set_property(TARGET ${PL_BACKEND}_observables PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()