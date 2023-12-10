
add_executable(gemm 
    main.c
    data.h
)

target_include_directories(gemm PUBLIC ${COMMON_INCLUDE_DIRS} ${SNRT_INCLUDE_DIRS} ${OCRT_INCLUDE_DIRS})
target_sources(gemm PUBLIC ${COMMON_SOURCES} ${SNRT_SOURCES} ${OCRT_SOURCES})