cmake_minimum_required(VERSION 3.20)

project(lightning_simulators LANGUAGES CXX)

add_subdirectory(base)

###############################################################################
# Determine simulator and include its directory
###############################################################################
include("${pennylane_lightning_SOURCE_DIR}/cmake/support_simulators.cmake")
FIND_AND_ADD_SIMULATOR()

if (BUILD_TESTS)
    enable_testing()
endif()
