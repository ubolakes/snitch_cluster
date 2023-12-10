# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)
project(occamy LANGUAGES C ASM)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#set(CMAKE_TOOLCHAIN_FILE cmake/toolchain-llvm.cmake CACHE STRING "Toolchain to use")
# use instead: -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-llvm.cmake

add_subdirectory(external)
message("EXTERNAL_INCLUDE_DIRS: ${EXTERNAL_INCLUDE_DIRS}")
message("EXTERNAL_SOURCES: ${EXTERNAL_SOURCES}")

set(COMMON_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/blas
        ${CMAKE_CURRENT_SOURCE_DIR}/math
        ${CMAKE_CURRENT_SOURCE_DIR}/prng
        ${EXTERNAL_INCLUDE_DIRS}
)
set(COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/gemm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/axpy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/blas.h
        ${CMAKE_CURRENT_SOURCE_DIR}/prng/lcg.h
        ${EXTERNAL_SOURCES}
)
message("COMMON_INCLUDE_DIRS: ${COMMON_INCLUDE_DIRS}")
add_subdirectory(snrt)
add_subdirectory(ocrt)

message("SNRT_SOURCES: ${SNRT_SOURCES}")
message("OCRT_SOURCES: ${OCRT_SOURCES}")

add_subdirectory(test)
