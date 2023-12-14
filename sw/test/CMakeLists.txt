function(register_snitch_target TARGET)
    cmake_parse_arguments(
            REGISTER
            "OBJDUMP;DWARFDUMP"
            ""
            ""
            ${ARGN}
    )
    message(STATUS "register_snitch_target(${TARGET}, OBJDUMP=${REGISTER_OBJDUMP}, DWARFDUMP=${REGISTER_DWARFDUMP})")

    target_compile_definitions(${TARGET} PUBLIC SNITCH_CLUSTER) # Set Platform

    target_include_directories(${TARGET} PUBLIC ${COMMON_INCLUDE_DIRS} ${PLATFORM_SNC_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS})
    target_sources(${TARGET} PUBLIC ${COMMON_SOURCES} ${PLATFORM_SNC_SOURCES} ${SNRT_SOURCES})
    target_compile_options(${TARGET} PUBLIC ${OCCAMY_DEVICE_COMPILE_OPTIONS})

    target_link_directories(${TARGET} PUBLIC ${SNRT_LINK_INCLUDE_DIRS})
    target_link_options(${TARGET} PUBLIC ${OCCAMY_DEVICE_LINK_OPTIONS} -T${SNRT_LINKER_SCRIPT} -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.map)
    set_target_properties(${TARGET} PROPERTIES LINK_DEPENDS ${SNRT_LINKER_SCRIPT})

    if ("${REGISTER_OBJDUMP}")
        target_objdump(${TARGET})
    endif ()

    if ("${REGISTER_DWARFDUMP}")
        target_dwarfdump(${TARGET})
    endif ()

endfunction()

add_subdirectory(blas/axpy)
add_subdirectory(blas/gemm)
add_subdirectory(occamy/blas/gemm)

add_custom_target(all.test)
add_custom_target(all.test.snitch_cluster)
add_custom_target(all.test.occamy)
add_dependencies(all.test axpy.elf gemm.elf gemm_oc.elf)
add_dependencies(all.test.snitch_cluster axpy.elf gemm.elf)
add_dependencies(all.test.occamy gemm_oc.elf)