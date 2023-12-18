
# See also: https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/CMakeLists.txt

set(TARGET gemm.elf)
add_executable(${TARGET}
        main.c
        data.h
)

# Generate data header
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/data.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS datagen.py params.hjson
        COMMAND PYTHONPATH=$ENV{PYTHONPATH} python ARGS datagen.py --cfg=params.hjson --section="" > data.h
        COMMENT "Generating ${CMAKE_CURRENT_SOURCE_DIR}/data.h"
)
#configure_file(params.hjson ${CMAKE_CURRENT_BINARY_DIR}/params.hjson COPYONLY)

register_snitch_target(${TARGET} OBJDUMP ON DWARFDUMP ON)

# TODO: fix builtins not appearing __builtin_ssr_enable in inspections
