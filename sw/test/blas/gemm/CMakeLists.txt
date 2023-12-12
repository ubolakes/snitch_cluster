
# See also: https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/CMakeLists.txt

set(TARGET gemm)
add_executable(${TARGET}
    main.c
    data.h
)

target_include_directories(${TARGET} PUBLIC ${COMMON_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS} ${OCRT_INCLUDE_DIRS})
target_sources(${TARGET} PUBLIC ${COMMON_SOURCES} ${SNRT_SOURCES} ${OCRT_SOURCES})
target_compile_options(${TARGET} PUBLIC ${OCCAMY_DEVICE_COMPILE_OPTIONS})

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        COMMAND ${CMAKE_OBJDUMP} ARGS -d ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.elf > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        DEPENDS ${TARGET}
        COMMENT "Generating objdump .d files."
)