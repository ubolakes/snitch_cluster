cmake_minimum_required(VERSION 3.10)

set(OCRT_HOST_INCLUDE_DIRS
        global
        host
)

set(OCRT_DEVICE_INCLUDE_DIRS
        global
        device
)

set(OCRT_GLOBAL_HEADERS
        global/heterogeneous_runtime.h
        global/heterogeneous_runtime_decls.h
)

set(OCRT_HOST_HEADERS
        host/host.h
)
list(APPEND OCRT_HOST_HEADERS ${OCRT_GLOBAL_HEADERS})

set(OCRT_DEVICE_HEADERS
        device/ocrt.h
        device/occamy_device.h
        device/occamy_device_decls.h
        device/occamy_memory.h
)
list(APPEND OCRT_DEVICE_HEADERS ${OCRT_GLOBAL_HEADERS})

set(OCRT_HOST_SOURCES
        host/host.c
        host/start.S
)
list(APPEND OCRT_HOST_SOURCES ${OCRT_HOST_HEADERS})

set(OCRT_DEVICE_SOURCES
        device/occamy_device.c
        device/occamy_memory.c
        device/occamy_start.c
        device/occamy_start.S
        device/putchar.c
)
list(APPEND OCRT_DEVICE_SOURCES ${OCRT_DEVICE_HEADERS})

# Linker Scripts
set(OCRT_LINK_INCLUDE_DIRS link)
set(OCRT_HOST_LINKER_SCRIPT link/host.ld)
set(OCRT_DEVICE_LINKER_SCRIPT ${SNRT_LINKER_SCRIPT})
set(OCRT_DEVICE_LINKER_SOURCES ${OCRT_DEVICE_LINKER_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR}/link/memory.ld)

# Make absolute and export to parent scope
transform_lists(
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
        LISTS OCRT_HOST_INCLUDE_DIRS OCRT_HOST_HEADERS OCRT_HOST_SOURCES OCRT_DEVICE_INCLUDE_DIRS
        OCRT_DEVICE_HEADERS OCRT_DEVICE_SOURCES OCRT_LINK_INCLUDE_DIRS OCRT_HOST_LINKER_SCRIPT
        PARENT_SCOPE
)
transform_lists(
        LISTS OCRT_DEVICE_LINKER_SCRIPT OCRT_DEVICE_LINKER_SOURCES
        PARENT_SCOPE
)

# add_library(ocrt INTERFACE)
# add_library(ocrt::ocrt ALIAS ocrt)
# set_target_properties(ocrt PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${OCRT_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS}")
# target_sources(ocrt INTERFACE "${OCRT_SOURCES} ${OCRT_HEADERS}")

set(OCRT_FOUND true PARENT_SCOPE)