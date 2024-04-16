cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_bindings LANGUAGES CXX)

add_library(lightning_qubit_bindings INTERFACE)

target_include_directories(lightning_qubit_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_qubit_bindings INTERFACE    lightning_bindings
                                                            lightning_qubit
                                                            lightning_gates 
                                                            lightning_utils
                                                            lightning_qubit_gates
                                                            lightning_qubit_utils
                                                            )

set_property(TARGET lightning_qubit_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)
