cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_observables LANGUAGES CXX)

set(OBSERVABLES_FILES ObservablesKokkos.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos_observables STATIC ${OBSERVABLES_FILES})

target_include_directories(lightning_kokkos_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_kokkos_observables PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(lightning_kokkos_observables PUBLIC  lightning_utils
                                                        lightning_gates
                                                        lightning_observables
                                                        lightning_kokkos_utils
                                                        lightning_kokkos_gates
                                                        lightning_kokkos
                                                        )

set_property(TARGET lightning_kokkos_observables PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()