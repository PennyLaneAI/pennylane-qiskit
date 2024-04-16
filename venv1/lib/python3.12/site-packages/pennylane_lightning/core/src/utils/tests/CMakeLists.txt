cmake_minimum_required(VERSION 3.20)

project(utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(utils_tests INTERFACE)
target_link_libraries(utils_tests INTERFACE Catch2::Catch2
                                            lightning_utils
                                            )

ProcessTestOptions(utils_tests)

target_sources(utils_tests INTERFACE runner_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_BitUtil.cpp
                    Test_ConstantUtil.cpp
                    Test_Error.cpp
                    Test_RuntimeInfo.cpp
                    Test_TypeTraits.cpp
                    Test_Util.cpp
                    )

if(ENABLE_LAPACK)
    list(APPEND TEST_SOURCES Test_UtilLinearAlg.cpp)
endif()

add_executable(utils_test_runner ${TEST_SOURCES})
target_link_libraries(utils_test_runner PRIVATE utils_tests)

catch_discover_tests(utils_test_runner)

install(TARGETS utils_test_runner DESTINATION bin)
