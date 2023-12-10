cmake_minimum_required(VERSION 3.10)

set(OCRT_INCLUDE_DIRS
        include
        host
        device
        platform
        platform/generated
        runtime
)

set(OCRT_HEADERS
        runtime/heterogeneous_runtime.h

        host/host.h

        device/occamy_defs.h
        device/occamy_device.h
        device/occamy_memory.h
        device/snrt.h

        platform/occamy.h
        platform/occamy_memory_map.h

        platform/generated/bitfield.h
        platform/generated/clint.h
        platform/generated/fpu_util.h
        platform/generated/occamy_base_addr.h
        platform/generated/occamy_cfg.h
        platform/generated/occamy_soc_ctrl.h
        platform/generated/snitch_cluster_peripheral.h
        platform/generated/snitch_hbm_xbar_peripheral.h
        platform/generated/snitch_quad_peripheral.h
        platform/generated/sys_dma.h
        platform/generated/tlb.h
        platform/generated/uart.h
)

set(OCRT_SOURCES
        host/host.c
        host/host.ld
        host/start.S

#        device/occamy_device.c
#        device/occamy_memory.c
#        device/occamy_start.c
#        device/occamy_start.S
        device/putchar.c
        device/snrt.c

#        platform/generated/sys_dma.c
)

# Make Absolute
list(TRANSFORM OCRT_INCLUDE_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM OCRT_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM OCRT_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND OCRT_SOURCES ${OCRT_HEADERS})

set(OCRT_INCLUDE_DIRS ${OCRT_INCLUDE_DIRS} PARENT_SCOPE)
set(OCRT_HEADERS ${OCRT_HEADERS} PARENT_SCOPE)
set(OCRT_SOURCES ${OCRT_SOURCES} PARENT_SCOPE)


# add_library(ocrt INTERFACE)
# add_library(ocrt::ocrt ALIAS ocrt)
# set_target_properties(ocrt PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${OCRT_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS}")
# target_sources(ocrt INTERFACE "${OCRT_SOURCES} ${OCRT_HEADERS}")

set(OCRT_FOUND true PARENT_SCOPE)