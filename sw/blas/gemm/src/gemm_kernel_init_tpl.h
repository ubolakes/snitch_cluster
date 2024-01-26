#include "gemm_decls.h"

#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef SNBLAS_GEMM_CLUSTER_KERNEL_INIT
#define SNBLAS_GEMM_CLUSTER_KERNEL_INIT(float_t)    CONCAT(snblas_gemm_cluster_kernel_init_, float_t)
#define SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(float_t) CONCAT(snblas_gemm_cluster_kernel_compute_, float_t)
#define SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(float_t)  CONCAT(snblas_gemm_cluster_kernel_deinit_, float_t)
#define SNBLAS_GEMM_CLUSTER_KERNEL(float_t)         CONCAT(snblas_gemm_cluster_kernel_, float_t)
#endif

extern void SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(const SnblasGemmInfo info);
inline void SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(const SnblasGemmInfo info) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    const uint32_t M   = info.M / P[0];
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.lda * P[0];
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc * P[0];
    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;

    // Unrolling factor of most inner loop.
    // Should be at least as high as the FMA delay
    // for maximum utilization
    const uint32_t unroll = 8;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    uint32_t ssr0_b[4] = {unroll, K / VECTOR_SIZE(FLOAT_T), N / unroll, M};
    uint32_t ssr0_i[4] = {0, sizeof(FLOAT_T) * VECTOR_SIZE(FLOAT_T), 0, sizeof(FLOAT_T) * lda};

    uint32_t ssr1_b[4] = {unroll, K / VECTOR_SIZE(FLOAT_T), N / unroll, M};
    uint32_t ssr1_i[4] = {sizeof(FLOAT_T) * ldb, sizeof(FLOAT_T) * VECTOR_SIZE(FLOAT_T),
                            sizeof(FLOAT_T) * unroll * ldb, 0};

    snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                        ssr0_i[1], ssr0_i[2], ssr0_i[3]);
    snrt_ssr_repeat(SNRT_SSR_DM0, unroll);

    snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                        ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2], ssr1_i[3]);

    snrt_ssr_enable();
}

extern void SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(const SnblasGemmInfo info);
inline void SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(const SnblasGemmInfo info) {
    snrt_ssr_disable();
}

void SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args);

/**
 * \brief Perform a one-time gemm computation for data in TCDM. 
 * Use the `init`, `compute` and `deinit` directly to get maximum performance when running multiple times.
*/
extern void SNBLAS_GEMM_CLUSTER_KERNEL(FLOAT_T)(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args);
inline void SNBLAS_GEMM_CLUSTER_KERNEL(FLOAT_T)(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args) {
    SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(info);
    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(info, args);
    SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(info);
}