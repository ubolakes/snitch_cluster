#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef SNBLAS_GEMM_ARGS
#define SNBLAS_GEMM_ARGS(float_t) CONCAT(SnblasGemmArgs_, float_t)
#define SNBLAS_GEMM_TCDM(float_t) CONCAT(SnblasGemmTcdm_, float_t)
#endif

/// Arguments to execute a GEMM computation, given a corresponding GemmInfo instance
typedef struct {
    const FLOAT_T* A;
    const FLOAT_T* B;
    FLOAT_T*       C;
    FLOAT_T        alpha;
    FLOAT_T        beta;
} SNBLAS_GEMM_ARGS(FLOAT_T);

/**
 * \brief Maps the layout of the TCDM. May be double buffered.
 */
typedef struct {
    FLOAT_T A[L1_M * L1_K] __attribute__ ((aligned (32*8)));
    FLOAT_T B[L1_K * L1_N] __attribute__ ((aligned (32*8)));
    FLOAT_T C[L1_M * L1_N] __attribute__ ((aligned (32*8)));
} SNBLAS_GEMM_TCDM(FLOAT_T);

// NAMED_DUMP(SNBLAS_GEMM_TCDM(FLOAT_T)*, CONCAT(l1_, FLOAT_T), 0x8)
