
# See also: https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/CMakeLists.txt

set(TARGET gemm.elf)
add_executable(${TARGET}
    main.c
    data.h
)
target_compile_definitions(${TARGET} PUBLIC SNITCH_CLUSTER) # Set Platform

target_include_directories(${TARGET} PUBLIC ${COMMON_INCLUDE_DIRS} ${PLATFORM_SNC_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS}) # ${OCRT_INCLUDE_DIRS})
target_sources(${TARGET} PUBLIC ${COMMON_SOURCES} ${PLATFORM_SNC_SOURCES} ${SNRT_SOURCES}) # ${OCRT_SOURCES})
target_compile_options(${TARGET} PUBLIC ${OCCAMY_DEVICE_COMPILE_OPTIONS})

target_link_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${TARGET} PUBLIC -T${SNRT_LINKER_SCRIPT} -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.map)
set_target_properties(${TARGET} PROPERTIES LINK_DEPENDS ${SNRT_LINKER_SCRIPT})

add_custom_command(TARGET ${TARGET} POST_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        BYPRODUCTS ${TARGET}.d
        COMMAND ${CMAKE_OBJDUMP} ARGS -d ${TARGET} > ${TARGET}.d
        COMMENT "Generating objdump .d files."
)