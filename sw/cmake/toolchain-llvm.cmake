# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Based on SiFive example: https://github.com/five-embeddev/riscv-scratchpad/blob/master/cmake/cmake/riscv.cmake

set(CMAKE_SYSTEM_NAME Generic)
#set(CMAKE_SYSTEM_PROCESSOR rv64imafdc) # For Host
#set(CMAKE_SYSTEM_PROCESSOR rv64imafdc) # For Device
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMAKE_C_COMPILER_ID Clang)
set(CMAKE_ASM_COMPILER riscv32-unknown-elf-cc)
set(CMAKE_C_COMPILER riscv32-unknown-elf-cc)
set(CMAKE_CXX_COMPILER riscv32-unknown-elf-c++)
set(CMAKE_AR llvm-ar)
set(CMAKE_STRIP llvm-strip)
set(CMAKE_RANLIB llvm-ranlib)

# Save non-standard variables to cache
set(CMAKE_OBJCOPY llvm-objcopy -O binary CACHE FILEPATH "The toolchain objcopy command" FORCE)
set(CMAKE_OBJDUMP llvm-objdump CACHE FILEPATH "The toolchain objdump command" FORCE)
set(CMAKE_DWARFDUMP llvm-dwarfdump CACHE FILEPATH "The toolchain dwarfdump/objdump-debug command" FORCE)

get_filename_component(RISCV_TOOLCHAIN_BIN_PATH ${CMAKE_C_COMPILER} DIRECTORY)
set(LLVM_LIB_ROOT "${RISCV_TOOLCHAIN_BIN_PATH}/../lib/clang/${LLVM_VER}/lib/" CACHE PATH "Root directory for LLVM libaries" FORCE)

# LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION true)
set(CMAKE_C_COMPILER_AR ${CMAKE_AR})
set(CMAKE_CXX_COMPILER_AR ${CMAKE_AR})
set(CMAKE_C_COMPILER_RANLIB ${CMAKE_RANLIB})
set(CMAKE_CXX_COMPILER_RANLIB ${CMAKE_RANLIB})

##
## Compile options
##

# Generic options for the host and device
#add_compile_options(-mcpu=snitch -mcmodel=medany)
add_compile_options(
        -mcmodel=medany
        -ffast-math -fno-builtin-printf -fno-common -ffunction-sections
        -static
        # For SSR register merge we need to disable the scheduler
        -mllvm -enable-misched=false
        # LLD doesn't support relaxation for RISC-V yet
        -mno-relax
        -fopenmp
        # For smallfloat we need experimental extensions enabled (Zfh)
        -menable-experimental-extensions

        -Wextra
)

# Specific options
set(OCCAMY_HOST_COMPILE_OPTIONS
        -march=rv64imafdc -mabi=lp64d
        CACHE STRING "Compile options for the Occamy CVA6 host" FORCE
)
set(OCCAMY_DEVICE_COMPILE_OPTIONS
        -mcpu=snitch -mabi=ilp32d
        -ftls-model=local-exec
        # -menable-experimental-extensions
#        -mno-fdiv # Not supported by Clang
        -ftls-model=local-exec

        # Required by math library to avoid conflict with stdint definition
        -D__DEFINED_uint64_t
        CACHE STRING "Compile options for the Occamy snitch device" FORCE
)


##
## Link options
##

# Generic options for the host and device
add_link_options(-nostartfiles -fuse-ld=lld -Wl,--image-base=0x80000000)
add_link_options(-static)
# LLD defaults to -z relro which we don't want in a static ELF
add_link_options(-Wl,-z,norelro)
add_link_options(-Wl,--gc-sections)
add_link_options(-Wl,--no-relax)
# add_link_options(-Wl,--verbose)

# Libraries
link_libraries(-lm)
#link_libraries(-lgcc)

# Add preprocessor definition to indicate LLD is used
add_compile_definitions(__LINK_LLD)
add_compile_definitions(__TOOLCHAIN_LLVM__)

# Specific options
set(OCCAMY_HOST_LINK_OPTIONS
        #        -T$(LINKER_SCRIPT)
        CACHE STRING "Linker options for the Occamy CVA6 host" FORCE)
set(OCCAMY_DEVICE_LINK_OPTIONS -mcpu=snitch
        -nostdlib
        -lc
        "-L${LLVM_LIB_ROOT}"
        -lclang_rt.builtins-riscv32
        CACHE STRING "Linker options for the Occamy snitch device" FORCE)


##
## Dump Options
##
set(CMAKE_DUMP_OPTIONS
        --remove-section=.comment
        --remove-section=.riscv.attributes
        --remove-section=.debug_info
        --remove-section=.debug_abbrev
        --remove-section=.debug_line
        --remove-section=.debug_str
        --remove-section=.debug_aranges
        CACHE STRING "Dump options for creating .d files" FORCE
)
