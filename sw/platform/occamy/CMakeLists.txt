cmake_minimum_required(VERSION 3.10)

set(PLATFORM_OCCAMY_INCLUDE_DIRS
        .
        generated
)

set(PLATFORM_OCCAMY_HEADERS
        generated/bitfield.h
        generated/clint.h
        generated/fpu_util.h
        generated/occamy_base_addr.h
        generated/occamy_cfg.h
        generated/occamy_soc_ctrl.h
        generated/snitch_cluster_peripheral.h
        generated/snitch_hbm_xbar_peripheral.h
        generated/snitch_quad_peripheral.h
        generated/sys_dma.h
        generated/tlb.h
        generated/uart.h

        occamy.h
        occamy_memory_map.h
)

set(PLATFORM_OCCAMY_SOURCES
        # generated/sys_dma.c  # TODO: Make sure it includes the header when generated
)
list(APPEND PLATFORM_OCCAMY_SOURCES ${PLATFORM_OCCAMY_HEADERS})

set(PLATFORM_OCCAMY_EXTRACT_ORIGIN_SH "${CMAKE_CURRENT_SOURCE_DIR}/extract-origin-ld.sh" PARENT_SCOPE)

# Make absolute and export to parent scope
transform_lists(
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
        LISTS PLATFORM_OCCAMY_INCLUDE_DIRS PLATFORM_OCCAMY_HEADERS PLATFORM_OCCAMY_SOURCES
        PARENT_SCOPE
)