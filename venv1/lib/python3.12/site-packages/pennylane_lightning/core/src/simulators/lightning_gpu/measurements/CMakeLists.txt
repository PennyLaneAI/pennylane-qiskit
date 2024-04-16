cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_measurements LANGUAGES CXX)

add_library(${PL_BACKEND}_measurements INTERFACE)

target_include_directories(${PL_BACKEND}_measurements INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PL_BACKEND}_measurements INTERFACE  lightning_compile_options
                                                            lightning_external_libs
                                                            lightning_measurements
                                                            lightning_observables
                                                            lightning_utils
                                                            ${PL_BACKEND}
                                                            ${PL_BACKEND}_utils
                                                            ${PL_BACKEND}_observables
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()