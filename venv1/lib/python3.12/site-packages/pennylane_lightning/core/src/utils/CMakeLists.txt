cmake_minimum_required(VERSION 3.20)

project(lightning_utils LANGUAGES CXX)

set(LQUBIT_UTILS_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)

if(ENABLE_LAPACK)
    list(APPEND LQUBIT_UTILS_FILES UtilLinearAlg.cpp)
endif()

add_library(lightning_utils STATIC ${LQUBIT_UTILS_FILES})

target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_utils INTERFACE lightning_compile_options
                                                lightning_external_libs
                                                )
set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()