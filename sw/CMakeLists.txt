# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)
project(occamy LANGUAGES C ASM)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#set(CMAKE_TOOLCHAIN_FILE cmake/toolchain-llvm.cmake CACHE STRING "Toolchain to use")
# use instead: -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-llvm.cmake

add_subdirectory(scripts)

add_subdirectory(external)
add_subdirectory(platform)
add_subdirectory(blas)

set(COMMON_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/math
        ${CMAKE_CURRENT_SOURCE_DIR}/prng
)
set(COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/prng/lcg.h
)

list(APPEND COMMON_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS})
list(APPEND COMMON_SOURCES ${EXTERNAL_SOURCES} ${BLAS_SOURCES})

add_subdirectory(snrt)
add_subdirectory(ocrt)

add_subdirectory(test)
