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
#        generated/sys_dma.c  # TODO: Make sure it includes the header when generated
)

# Make Absolute
list(TRANSFORM PLATFORM_OCCAMY_INCLUDE_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_OCCAMY_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_OCCAMY_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND PLATFORM_OCCAMY_SOURCES ${PLATFORM_OCCAMY_HEADERS})

set(PLATFORM_OCCAMY_INCLUDE_DIRS ${PLATFORM_OCCAMY_INCLUDE_DIRS} PARENT_SCOPE)
set(PLATFORM_OCCAMY_HEADERS ${PLATFORM_OCCAMY_HEADERS} PARENT_SCOPE)
set(PLATFORM_OCCAMY_SOURCES ${PLATFORM_OCCAMY_SOURCES} PARENT_SCOPE)