cmake_minimum_required(VERSION 3.10)

set(PLATFORM_INCLUDE_DIRS
        .
        generated
)

set(PLATFORM_HEADERS
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

        platform.h
        occamy.h
        occamy_memory_map.h
)

set(PLATFORM_SOURCES
#        generated/sys_dma.c
)

# Make Absolute
list(TRANSFORM PLATFORM_INCLUDE_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND PLATFORM_SOURCES ${PLATFORM_HEADERS})

set(PLATFORM_INCLUDE_DIRS ${PLATFORM_INCLUDE_DIRS} PARENT_SCOPE)
set(PLATFORM_HEADERS ${PLATFORM_HEADERS} PARENT_SCOPE)
set(PLATFORM_SOURCES ${PLATFORM_SOURCES} PARENT_SCOPE)


# add_library(ocrt INTERFACE)
# add_library(ocrt::ocrt ALIAS ocrt)
# set_target_properties(ocrt PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${OCRT_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS}")
# target_sources(ocrt INTERFACE "${OCRT_SOURCES} ${OCRT_HEADERS}")

set(OCRT_FOUND true PARENT_SCOPE)