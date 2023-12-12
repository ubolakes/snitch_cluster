
# See also: https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/CMakeLists.txt

set(TARGET gemm)
add_executable(${TARGET}
    main.c
    data.h
)

target_include_directories(${TARGET} PUBLIC ${COMMON_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS} ${OCRT_INCLUDE_DIRS})
target_sources(${TARGET} PUBLIC ${COMMON_SOURCES} ${SNRT_SOURCES} ${OCRT_SOURCES})
target_compile_options(${TARGET} PUBLIC ${OCCAMY_DEVICE_COMPILE_OPTIONS})

target_link_directories(${TARGET} PUBLIC ${OCRT_LINK_INCLUDE_DIRS})
target_link_options(${TARGET} PUBLIC -T${SNRT_LINKER_SCRIPT} ${OCCAMY_DEVICE_LINK_OPTIONS} -Wl,-Map=${TARGET}.map)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -T ${SNRT_LINKER_SCRIPT} -Wl,-Map=${TARGET}.map")

set_target_properties(${TARGET} PROPERTIES LINK_DEPENDS "${OCRT_LINK_SOURCES_DEVICE};${SNRT_LINKER_SCRIPT}")

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        COMMAND ${CMAKE_OBJDUMP} ARGS -d ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.elf > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        DEPENDS ${TARGET}
        COMMENT "Generating objdump .d files."
)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/origin.ld
        COMMAND ${CMAKE_OBJDUMP} ARGS -d ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.elf > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        DEPENDS ${TARGET}.part
        COMMENT "Extracting origin.ld"
)