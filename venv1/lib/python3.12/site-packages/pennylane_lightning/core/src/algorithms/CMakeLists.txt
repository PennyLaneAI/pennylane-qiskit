cmake_minimum_required(VERSION 3.20)

project(lightning_algorithms LANGUAGES CXX)

add_library(lightning_algorithms INTERFACE)

if(ENABLE_MPI)
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_MPI=1")
endif()

target_include_directories(lightning_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_algorithms INTERFACE    lightning_compile_options
                                                        lightning_external_libs
                                                        lightning_utils
                                                        lightning_observables
                                                        )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()