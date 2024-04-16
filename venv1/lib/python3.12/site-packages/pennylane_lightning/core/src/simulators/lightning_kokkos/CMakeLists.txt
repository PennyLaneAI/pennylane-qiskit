cmake_minimum_required(VERSION 3.20)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█░█░█▀█░█░█░█░█░█▀█░█▀▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█▀▄░█░█░█▀▄░█▀▄░█░█░▀▀█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀░▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀▀▀
]=])
message(${LOGO})

project(lightning_kokkos
    DESCRIPTION "PennyLane Lightning Kokkos C++ Backend."
    LANGUAGES CXX C
)

set(LKOKKOS_FILES   StateVectorKokkos.cpp
                    CACHE INTERNAL "" FORCE)

add_library(lightning_kokkos STATIC ${LKOKKOS_FILES})
target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLKOKKOS=1")

##########################
## Enforce C++ Standard ##
##########################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

#########################
## Set Default Options ##
#########################
option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLKOKKOS_ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(PLKOKKOS_ENABLE_WARNINGS "Enable warnings" ON)

# Include macro and functions supporting Kokkos libraries.
include("${pennylane_lightning_SOURCE_DIR}/cmake/support_kokkos.cmake")
FindKokkos(lightning_external_libs)

if(PLKOKKOS_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-fsanitize=address)
    else ()
        add_link_options(-fsanitize=address -static-libasan)
    endif()
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

target_link_libraries(lightning_kokkos PUBLIC  lightning_compile_options
                                            lightning_external_libs
                                            lightning_base
                                            lightning_gates
                                            lightning_utils
                                            lightning_kokkos_utils
                                            lightning_kokkos_gates
                                            )
target_include_directories(lightning_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET lightning_kokkos PROPERTY POSITION_INDEPENDENT_CODE ON)
if(PLKOKKOS_ENABLE_NATIVE)
    target_compile_options(lightning_kokkos PRIVATE -march=native)
endif()

###############################################################################
# Include subdirectories
###############################################################################
set(COMPONENT_SUBDIRS   algorithms
                        bindings
                        gates
                        measurements
                        observables
                        utils
                        )

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
