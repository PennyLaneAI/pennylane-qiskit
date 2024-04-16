cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_utils LANGUAGES CXX)

add_library(lightning_kokkos_utils INTERFACE)

target_include_directories(lightning_kokkos_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_kokkos_utils INTERFACE lightning_gates lightning_utils)

set_property(TARGET lightning_kokkos_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
