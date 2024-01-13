cmake_minimum_required(VERSION 3.10)

set(BLAS_INCLUDE_DIRS
        .
)

set(BLAS_HEADERS
        blas.h
        axpy.h
        gemm.h
        gemm_1c.h
        gemm_oc.h
        gemm_oc_opt1d.h
        gemm_oc_opt2d.h
        gemm_tiled.h
)

set(BLAS_SOURCES

)
list(APPEND BLAS_SOURCES ${BLAS_HEADERS})

transform_lists(
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
        LISTS BLAS_INCLUDE_DIRS BLAS_HEADERS BLAS_SOURCES
        PARENT_SCOPE
)
