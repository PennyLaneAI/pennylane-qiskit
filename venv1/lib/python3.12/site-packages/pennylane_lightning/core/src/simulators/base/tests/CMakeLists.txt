cmake_minimum_required(VERSION 3.20)

project(base_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(base_tests INTERFACE)
target_link_libraries(base_tests INTERFACE  Catch2::Catch2)

ProcessTestOptions(base_tests)

# Create dependency on the dynamically defined simulator/backend target.
foreach(BACKEND ${PL_BACKEND})
    target_link_libraries(base_tests INTERFACE ${BACKEND})
endforeach()

target_sources(base_tests INTERFACE runner_base.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_StateVectorBase.cpp
                    )

add_executable(base_test_runner ${TEST_SOURCES})
target_link_libraries(base_test_runner PRIVATE  base_tests)

catch_discover_tests(base_test_runner)

install(TARGETS base_test_runner DESTINATION bin)
