cmake_minimum_required(VERSION 3.20)

project(lightning_bindings LANGUAGES CXX)

add_library(lightning_bindings INTERFACE)

target_include_directories(lightning_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_bindings INTERFACE lightning_utils lightning_gates lightning_compile_options)

set_property(TARGET lightning_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)
