project(lightning_gates)

add_library(lightning_gates INTERFACE)
target_include_directories(lightning_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_gates INTERFACE  lightning_utils)
