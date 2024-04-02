#include "gemm_decls.h"

#ifndef METHOD 
#error "Define METHOD to use this template."
#endif

#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

// Instantiate template code
#ifndef SNBLAS_GEMM_TILING
#define SNBLAS_GEMM_TILING(method, float_t, is_dm_core, beta_nz) CONCAT5(snblas_gemm_, method, float_t, is_dm_core, beta_nz)
#endif

#define GEMM_TILING_TPL_H STR(CONCAT3(gemm_tiling_, METHOD, _tpl.h))
#define IS_DM_CORE true
#define BETA_NZ true
#include GEMM_TILING_TPL_H
#undef BETA_NZ
#define BETA_NZ false
#include GEMM_TILING_TPL_H
#undef BETA_NZ
#undef IS_DM_CORE

#define IS_DM_CORE false
#define BETA_NZ true
#include GEMM_TILING_TPL_H
#undef BETA_NZ
#define BETA_NZ false
#include GEMM_TILING_TPL_H
#undef BETA_NZ
#undef IS_DM_CORE


#ifndef SNBLAS_GEMM
#define SNBLAS_GEMM(method, float_t) CONCAT3(snblas_gemm_, method, float_t)
#endif

extern void SNBLAS_GEMM(METHOD, FLOAT_T) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl);
inline __attribute__((always_inline)) void SNBLAS_GEMM(METHOD, FLOAT_T) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl) {
    if (args.beta == (FLOAT_T)0.0) {
        if (snrt_is_dm_core()) {
            SNBLAS_GEMM_TILING(METHOD, FLOAT_T, true, false)(info, args, impl);
        } else {
            SNBLAS_GEMM_TILING(METHOD, FLOAT_T, false, false)(info, args, impl);
        }
    } else {
        if (snrt_is_dm_core()) {
            SNBLAS_GEMM_TILING(METHOD, FLOAT_T, true, true)(info, args, impl);
        } else {
            SNBLAS_GEMM_TILING(METHOD, FLOAT_T, false, true)(info, args, impl);
        }
    }
}
