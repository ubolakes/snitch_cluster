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
list(APPEND SNRT_SOURCES ${SNRT_HEADERS})

set(SNRT_LINK_INCLUDE_DIRS link)
set(SNRT_LINKER_SCRIPT link/base.ld)

# Make absolute and export to parent scope
transform_lists(
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
        LISTS SNRT_INCLUDE_DIRS SNRT_HEADERS SNRT_SOURCES SNRT_LINK_INCLUDE_DIRS SNRT_LINKER_SCRIPT
        PARENT_SCOPE
)

set(SNRT_FOUND true)