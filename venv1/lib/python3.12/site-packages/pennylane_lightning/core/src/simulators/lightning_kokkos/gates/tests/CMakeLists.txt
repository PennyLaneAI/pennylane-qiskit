cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_gates_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_kokkos_gates_tests INTERFACE)
target_link_libraries(lightning_kokkos_gates_tests INTERFACE    Catch2::Catch2
                                                                lightning_kokkos_gates
                                                                lightning_kokkos_measurements
                                                                lightning_kokkos_observables
                                                                lightning_kokkos
                                                                lightning_kokkos_utils
                                                                )

ProcessTestOptions(lightning_kokkos_gates_tests)

target_sources(lightning_kokkos_gates_tests INTERFACE runner_lightning_kokkos_gates.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_StateVectorKokkos_Generator.cpp
                    Test_StateVectorKokkos_NonParam.cpp
                    Test_StateVectorKokkos_Param.cpp
)

add_executable(lightning_kokkos_gates_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_kokkos_gates_test_runner PRIVATE lightning_kokkos_gates_tests)
catch_discover_tests(lightning_kokkos_gates_test_runner)

install(TARGETS lightning_kokkos_gates_test_runner DESTINATION bin)
