cmake_minimum_required(VERSION 3.20)

project(lightning_measurements LANGUAGES CXX)

add_library(lightning_measurements INTERFACE)

target_include_directories(lightning_measurements INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_measurements INTERFACE  lightning_compile_options
                                                        lightning_external_libs
                                                        lightning_observables
                                                        lightning_utils
                                                        )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()