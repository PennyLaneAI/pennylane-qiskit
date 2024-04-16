cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_utils LANGUAGES CXX)

add_library(lightning_qubit_utils INTERFACE)

target_include_directories(lightning_qubit_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_qubit_utils INTERFACE  lightning_utils)

set_property(TARGET lightning_qubit_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()