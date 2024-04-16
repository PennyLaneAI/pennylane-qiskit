project(${PL_BACKEND}_gates)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.

add_library(${PL_BACKEND}_gates INTERFACE)
target_include_directories(${PL_BACKEND}_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PL_BACKEND}_gates INTERFACE  ${PL_BACKEND}_utils lightning_external_libs)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
