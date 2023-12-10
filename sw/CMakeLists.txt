# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)
project(occamy LANGUAGES C ASM)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#set(CMAKE_TOOLCHAIN_FILE cmake/toolchain-llvm.cmake CACHE STRING "Toolchain to use")
# use instead: -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-llvm.cmake

option(USE_OCCAMY "Use Occamy Hardware" ON)
if(USE_OCCAMY)
    add_definitions(-DOCCAMY)
endif()

add_subdirectory(external)
add_subdirectory(platform)

set(COMMON_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/blas
        ${CMAKE_CURRENT_SOURCE_DIR}/math
        ${CMAKE_CURRENT_SOURCE_DIR}/prng
)
set(COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/gemm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/axpy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/blas.h
        ${CMAKE_CURRENT_SOURCE_DIR}/prng/lcg.h
)

list(APPEND COMMON_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS} ${PLATFORM_INCLUDE_DIRS})
list(APPEND COMMON_SOURCES ${EXTERNAL_SOURCES} ${PLATFORM_SOURCES})

message("COMMON_SOURCES: ${COMMON_SOURCES}")
add_subdirectory(snrt)
add_subdirectory(ocrt)

message("PLATFORM_SOURCES: ${SNRT_SOURCES}")
message("SNRT_SOURCES: ${SNRT_SOURCES}")
message("OCRT_SOURCES: ${OCRT_SOURCES}")

add_subdirectory(test)
