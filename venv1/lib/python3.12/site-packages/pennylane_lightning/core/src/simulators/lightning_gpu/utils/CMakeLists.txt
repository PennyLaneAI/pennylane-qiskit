project(${PL_BACKEND}_utils LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

findCUDATK(lightning_compile_options)

add_library(${PL_BACKEND}_utils INTERFACE)
target_include_directories(${PL_BACKEND}_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(${PL_BACKEND}_utils INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse lightning_compile_options)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
