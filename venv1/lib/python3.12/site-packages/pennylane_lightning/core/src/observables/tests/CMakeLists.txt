cmake_minimum_required(VERSION 3.20)

project(observables_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(observables_tests INTERFACE)
target_link_libraries(observables_tests INTERFACE   Catch2::Catch2
                                                    lightning_observables
                                                    )

ProcessTestOptions(observables_tests)

target_sources(observables_tests INTERFACE runner_observables.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_Observables.cpp
                    )

add_executable(observables_test_runner ${TEST_SOURCES})
target_link_libraries(observables_test_runner PRIVATE  observables_tests)

# Create dependency on the dynamically defined simulator/backend target.
foreach(BACKEND ${PL_BACKEND})
    target_link_libraries(observables_test_runner PRIVATE ${BACKEND})
endforeach()

catch_discover_tests(observables_test_runner)

install(TARGETS observables_test_runner DESTINATION bin)

if(ENABLE_MPI)
    add_library(observables_tests_mpi INTERFACE)
    target_link_libraries(observables_tests_mpi INTERFACE     Catch2::Catch2)
    foreach(BACKEND ${PL_BACKEND})
        target_link_libraries(observables_tests_mpi INTERFACE  ${BACKEND}
                                                            "${BACKEND}_gates"
                                                            "${BACKEND}_measurements"
                                                            "${BACKEND}_observables"
        )
    endforeach()

    ProcessTestOptions(observables_tests_mpi)

    target_sources(observables_tests_mpi INTERFACE ./mpi/runner_observables_mpi.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES    ./mpi/Test_ObservablesMPI.cpp)

    add_executable(observables_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(observables_test_runner_mpi PRIVATE observables_tests_mpi)
    catch_discover_tests(observables_test_runner_mpi)
endif()
