cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_qubit_utils_tests INTERFACE)
target_link_libraries(lightning_qubit_utils_tests INTERFACE     Catch2::Catch2
                                                                lightning_utils
                                                                lightning_qubit_utils
                                                                )

ProcessTestOptions(lightning_qubit_utils_tests)

target_sources(lightning_qubit_utils_tests INTERFACE runner_lightning_qubit_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_SparseLinAlg.cpp
                    Test_LinearAlgebra.cpp
                    )

add_executable(lightning_qubit_utils_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_qubit_utils_test_runner PRIVATE lightning_qubit_utils_tests)
catch_discover_tests(lightning_qubit_utils_test_runner)

install(TARGETS lightning_qubit_utils_test_runner DESTINATION bin)
