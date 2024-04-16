cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_bindings LANGUAGES CXX)

add_library(lightning_kokkos_bindings INTERFACE)

target_include_directories(lightning_kokkos_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_kokkos_bindings INTERFACE  lightning_bindings
                                                        lightning_utils
                                                        lightning_kokkos
                                                        lightning_kokkos_gates
                                                        lightning_kokkos_utils
                                                        )

set_property(TARGET lightning_kokkos_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)
