cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_bindings LANGUAGES CXX)

add_library(${PL_BACKEND}_bindings INTERFACE)

target_include_directories(${PL_BACKEND}_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PL_BACKEND}_bindings INTERFACE  lightning_bindings
                                                        lightning_utils
                                                        ${PL_BACKEND}
                                                        ${PL_BACKEND}_gates
                                                        ${PL_BACKEND}_measurements
                                                        ${PL_BACKEND}_utils
                                                        )

set_property(TARGET ${PL_BACKEND}_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)
