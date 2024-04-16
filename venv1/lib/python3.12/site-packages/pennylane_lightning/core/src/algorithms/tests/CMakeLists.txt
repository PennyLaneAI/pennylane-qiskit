cmake_minimum_required(VERSION 3.20)

project(algorithms_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(algorithms_tests INTERFACE)
target_link_libraries(algorithms_tests INTERFACE    Catch2::Catch2
                                                    lightning_algorithms
                                                    )

# Create dependencies on the dynamically defined simulator/backend targets.
foreach(BACKEND ${PL_BACKEND})
    target_link_libraries(algorithms_tests INTERFACE    ${BACKEND}
                                                        "${BACKEND}_algorithms"
                                                        "${BACKEND}_observables"
                                                        )
endforeach()

ProcessTestOptions(algorithms_tests)

target_sources(algorithms_tests INTERFACE runner_algorithms.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_AdjointJacobian.cpp
                    )

add_executable(algorithms_test_runner ${TEST_SOURCES})
target_link_libraries(algorithms_test_runner PRIVATE algorithms_tests)

catch_discover_tests(algorithms_test_runner)

install(TARGETS algorithms_test_runner DESTINATION bin)

if(ENABLE_MPI)
    add_library(algorithms_tests_mpi INTERFACE)
    target_link_libraries(algorithms_tests_mpi INTERFACE     Catch2::Catch2)
    foreach(BACKEND ${PL_BACKEND})
        target_link_libraries(algorithms_tests_mpi INTERFACE  ${BACKEND}
                                                            "${BACKEND}_gates"
                                                            "${BACKEND}_measurements"
                                                            "${BACKEND}_algorithms"
                                                            "${BACKEND}_observables"
        )
    endforeach()

    ProcessTestOptions(algorithms_tests_mpi)

    target_sources(algorithms_tests_mpi INTERFACE ./mpi/runner_algorithms_mpi.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES    ./mpi/Test_AdjointJacobianMPI.cpp)

    add_executable(algorithms_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(algorithms_test_runner_mpi PRIVATE algorithms_tests_mpi)
    catch_discover_tests(algorithms_test_runner_mpi)
endif()
