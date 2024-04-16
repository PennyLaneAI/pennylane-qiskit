project(lightning_kokkos_gates)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.

add_library(lightning_kokkos_gates INTERFACE)
target_include_directories(lightning_kokkos_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_kokkos_gates INTERFACE  lightning_kokkos_utils)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
