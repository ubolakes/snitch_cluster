cmake_minimum_required(VERSION 3.10)

set(OCRT_INCLUDE_DIRS
        include
        host
        device
        runtime
)

set(OCRT_HEADERS
        runtime/heterogeneous_runtime.h
        runtime/heterogeneous_runtime_decls.h

        host/host.h

        device/occamy_defs.h
        device/occamy_device.h
        device/occamy_device_decls.h
        device/occamy_memory.h
        device/snrt.h
)

set(OCRT_SOURCES
        host/host.c
        host/host.ld
        host/start.S

        device/occamy_device.c
        device/occamy_memory.c
        device/occamy_start.c
#        device/occamy_start.S
        device/putchar.c
        device/snrt.c
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