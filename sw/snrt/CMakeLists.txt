cmake_minimum_required(VERSION 3.10)

set(SNRT_INCLUDE_DIRS
        api
        api/omp
        src
        src/omp
)

set(SNRT_HEADERS
        src/snrt.h

        api/alloc_decls.h
        api/cls_decls.h
        api/cluster_interrupt_decls.h
        api/global_interrupt_decls.h
        api/memory_decls.h
        api/riscv_decls.h
        api/start_decls.h
        api/sync_decls.h
        api/team_decls.h

        api/omp/eu_decls.h

        src/alloc.h
        src/cls.h
        src/cluster_interrupts.h
        src/dm.h
        src/dma.h
        src/dump.h
        src/global_interrupts.h
        src/perf_cnt.h
        src/printf.h
        src/riscv.h
        src/ssr.h
        src/sync.h
        src/team.h

        src/omp/eu.h
        src/omp/interface.h
        src/omp/kmp.h
        src/omp/omp.h
)

set(SNRT_SOURCES
        src/alloc.c
        src/cls.c
        src/cluster_interrupts.c
        src/dm.c
        src/dma.c
        src/global_interrupts.c
        src/printf.c
        src/ssr.c
        src/start.c
        src/sync.c
        src/team.c

        src/omp/eu.c
        src/omp/kmp.c
        src/omp/omp.c
)

set(SNRT_LINKER_SCRIPT base.ld)


#set_source_files_properties(${HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make Absolute
list(TRANSFORM SNRT_INCLUDE_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM SNRT_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM SNRT_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM SNRT_LINKER_SCRIPT PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND SNRT_SOURCES ${SNRT_HEADERS})

# Export variables
set(SNRT_INCLUDE_DIRS ${SNRT_INCLUDE_DIRS} PARENT_SCOPE)
set(SNRT_HEADERS ${SNRT_HEADERS} PARENT_SCOPE)
set(SNRT_SOURCES ${SNRT_SOURCES} PARENT_SCOPE)
set(SNRT_LINKER_SCRIPT ${SNRT_LINKER_SCRIPT} PARENT_SCOPE)

mark_as_advanced(SNRT_INCLUDE_DIRS SNRT_HEADERS SNRT_SOURCES)
set(SNRT_FOUND true)