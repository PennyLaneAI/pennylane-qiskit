cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_algorithms_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_qubit_algorithms_tests INTERFACE)
target_link_libraries(lightning_qubit_algorithms_tests INTERFACE    Catch2::Catch2
                                                                    lightning_qubit
                                                                    lightning_qubit_algorithms
                                                                    lightning_qubit_observables
                                                                    )

ProcessTestOptions(lightning_qubit_algorithms_tests)

target_sources(lightning_qubit_algorithms_tests INTERFACE runner_lightning_qubit_algorithms.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_AdjointJacobianLQubit.cpp
                    Test_VectorJacobianProduct.cpp
                    )

add_executable(lightning_qubit_algorithms_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_qubit_algorithms_test_runner PRIVATE  lightning_qubit_algorithms_tests)

catch_discover_tests(lightning_qubit_algorithms_test_runner)

install(TARGETS lightning_qubit_algorithms_test_runner DESTINATION bin)
