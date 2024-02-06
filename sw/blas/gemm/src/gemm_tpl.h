#include "gemm_decls.h"

#ifndef METHOD 
#error "Define METHOD to use this template."
#endif

#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

// Instantiate template code
#ifndef SNBLAS_GEMM_TILING
#define SNBLAS_GEMM_TILING(method, float_t, is_dm_core) CONCAT4(snblas_gemm_, method, float_t, is_dm_core)
#endif

#define GEMM_TILING_TPL_H STR(CONCAT3(gemm_tiling_, METHOD, _tpl.h))
#define IS_DM_CORE true
#include GEMM_TILING_TPL_H
#undef IS_DM_CORE

#define IS_DM_CORE false
#include GEMM_TILING_TPL_H
#undef IS_DM_CORE


#ifndef SNBLAS_GEMM
#define SNBLAS_GEMM(method, float_t) CONCAT3(snblas_gemm_, method, float_t)
#endif

extern void SNBLAS_GEMM(METHOD, FLOAT_T) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, bool bench);
inline void SNBLAS_GEMM(METHOD, FLOAT_T) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, bool bench) {
    if (snrt_is_dm_core()) {
        SNBLAS_GEMM_TILING(METHOD, FLOAT_T, true)(info, args, bench);
    } else {
        SNBLAS_GEMM_TILING(METHOD, FLOAT_T, false)(info, args, bench);
    }
}
