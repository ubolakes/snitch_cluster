#include "gemm_decls.h"

#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

// Instantiate template code
#define USE_C2C_TILES true

#define IS_DM_CORE true
#include "gemm_tiling_2dpipe_tpl.h"
#undef IS_DM_CORE

#define IS_DM_CORE false
#include "gemm_tiling_2dpipe_tpl.h"
#undef IS_DM_CORE

#undef USE_C2C_TILES

#ifndef SNBLAS_GEMM
#define SNBLAS_GEMM(float_t) CONCAT(snblas_gemm_, float_t)
#endif

void SNBLAS_GEMM(FLOAT_T) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, bool bench) {
    if (snrt_is_dm_core()) {
        SNBLAS_GEMM_TILING(true, FLOAT_T)(info, args, bench);
    } else {
        SNBLAS_GEMM_TILING(false, FLOAT_T)(info, args, bench);
    }
}
