
# See also: https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/CMakeLists.txt

set(TARGET axpy.elf)
add_executable(${TARGET}
        main.c
        data.h
)

set(AXPY_VECTOR_LENGTH 24)
set(AXPY_DATAGEN_SECTION "")

# Generate data header
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/data.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS datagen.py
        COMMAND PYTHONPATH=$ENV{PYTHONPATH} python ARGS datagen.py ${AXPY_VECTOR_LENGTH} --section=${AXPY_DATAGEN_SECTION} > data.h
        COMMENT "Generating ${CMAKE_CURRENT_SOURCE_DIR}/data.h"
)

register_snitch_target(${TARGET} OBJDUMP ON DWARFDUMP ON)