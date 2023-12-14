set(EXTERNAL_INCLUDE_DIRS
        .
)

set(EXTERNAL_HEADERS
        printf/printf.h
        riscv-opcodes/encoding.h
)

set(EXTERNAL_SOURCES
        # printf/printf.c
)
list(APPEND EXTERNAL_SOURCES ${EXTERNAL_HEADERS})

# Make absolute and export to parent scope
transform_lists(
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
        LISTS EXTERNAL_INCLUDE_DIRS EXTERNAL_HEADERS EXTERNAL_SOURCES
        PARENT_SCOPE
)