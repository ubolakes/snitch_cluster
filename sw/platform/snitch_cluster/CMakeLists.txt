cmake_minimum_required(VERSION 3.10)

set(PLATFORM_SNC_INCLUDE_DIRS
        .
        generated
)

set(PLATFORM_SNC_HEADERS
        snitch_cluster_defs.h
        snitch_cluster_global_interrupts.h
        snitch_cluster_memory.h

        generated/snitch_cluster_addrmap.h
        generated/snitch_cluster_addrmap.h.tpl
        generated/snitch_cluster_cfg.h
        generated/snitch_cluster_cfg.h.tpl
        generated/snitch_cluster_peripheral.h
)

set(PLATFORM_SNC_SOURCES
        snitch_cluster_memory.c

        putchar.c
#        snitch_cluster_start.c  # TODO: is this needed?
        snitch_cluster_start.S
)

# Make Absolute
list(TRANSFORM PLATFORM_SNC_INCLUDE_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_SNC_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PLATFORM_SNC_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND PLATFORM_SNC_SOURCES ${PLATFORM_SNC_HEADERS})

set(PLATFORM_SNC_INCLUDE_DIRS ${PLATFORM_SNC_INCLUDE_DIRS} PARENT_SCOPE)
set(PLATFORM_SNC_HEADERS ${PLATFORM_SNC_HEADERS} PARENT_SCOPE)
set(PLATFORM_SNC_SOURCES ${PLATFORM_SNC_SOURCES} PARENT_SCOPE)