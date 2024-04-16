cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(${PL_BACKEND}_utils_tests INTERFACE)
target_link_libraries(${PL_BACKEND}_utils_tests INTERFACE     Catch2::Catch2
                                                                lightning_utils
                                                                ${PL_BACKEND}_utils
                                                                ${PL_BACKEND}
                                                                )

ProcessTestOptions(${PL_BACKEND}_utils_tests)

target_sources(${PL_BACKEND}_utils_tests INTERFACE runner_${PL_BACKEND}_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES Test_LinearAlgebra.cpp)

add_executable(${PL_BACKEND}_utils_test_runner ${TEST_SOURCES})
target_link_libraries(${PL_BACKEND}_utils_test_runner PRIVATE ${PL_BACKEND}_utils_tests)
catch_discover_tests(${PL_BACKEND}_utils_test_runner)

install(TARGETS ${PL_BACKEND}_utils_test_runner DESTINATION bin)

if(ENABLE_MPI)
add_library(${PL_BACKEND}_utils_tests_mpi INTERFACE)
target_link_libraries(${PL_BACKEND}_utils_tests_mpi INTERFACE     Catch2::Catch2
                                                                lightning_utils
                                                                ${PL_BACKEND}_utils
                                                                ${PL_BACKEND}
                                                                )

ProcessTestOptions(${PL_BACKEND}_utils_tests_mpi)

target_sources(${PL_BACKEND}_utils_tests_mpi INTERFACE ./mpi/runner_${PL_BACKEND}_utils_mpi.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES ./mpi/Test_LinearAlgebraMPI.cpp
                 ./mpi/Test_MPIManager.cpp
                 ./mpi/Test_CSRMatrix.cpp)

add_executable(${PL_BACKEND}_utils_test_runner_mpi ${TEST_SOURCES})
target_link_libraries(${PL_BACKEND}_utils_test_runner_mpi PRIVATE ${PL_BACKEND}_utils_tests_mpi)
catch_discover_tests(${PL_BACKEND}_utils_test_runner_mpi)

endif()