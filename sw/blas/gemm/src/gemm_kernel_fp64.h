#include "gemm_decls.h"

extern void snblas_gemm_cluster_kernel_init_fp64(const SnblasGemmInfo info, const SnblasGemmImpl impl);
inline __attribute__((always_inline)) void snblas_gemm_cluster_kernel_init_fp64(const SnblasGemmInfo info, const SnblasGemmImpl impl) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;
    const uint32_t tc  = info.tc;
    const uint32_t M   = info.M;
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.lda;
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc;

    // Unrolling factor of most inner loop.
    // Should be at least as high as the FMA delay
    // for maximum utilization
    const uint32_t unroll = FMADD_D_UNROLL;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    // First matrix is stored in transposed format
    //            loop order = {j0,     k, j1,         i}
    const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M / P[0]};
    if (ta) {
        const uint32_t ssr0_i[4] = {0, lda, 0, P[0]};

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                                       ssr0_i[1] * sizeof(fp64), ssr0_i[2] * sizeof(fp64), ssr0_i[3] * sizeof(fp64));
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll); // because ssr0_i[0] == 0
    } else {

        const uint32_t ssr0_i[4] = {0, 1, 0, lda * P[0]};

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                                       ssr0_i[1] * sizeof(fp64), ssr0_i[2] * sizeof(fp64), ssr0_i[3] * sizeof(fp64));
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll); // because ssr0_i[0] == 0
    }

    // Second matrix is stored in transposed format
    const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M / P[0]};
    if (tb) {
        const uint32_t ssr1_i[4] = {ldb, 1, unroll * ldb, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2], ssr1_b[3], 
                                       ssr1_i[0] * sizeof(fp64), ssr1_i[1] * sizeof(fp64), ssr1_i[2] * sizeof(fp64), ssr1_i[3] * sizeof(fp64));
        snrt_ssr_repeat(SNRT_SSR_DM1, 1);
    } else {

        const uint32_t ssr1_i[4] = {1, ldb, unroll, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2], ssr1_b[3], 
                                       ssr1_i[0] * sizeof(fp64), ssr1_i[1] * sizeof(fp64), ssr1_i[2] * sizeof(fp64), ssr1_i[3] * sizeof(fp64));
        snrt_ssr_repeat(SNRT_SSR_DM1, 1);
    }

    snrt_ssr_enable();
}

extern void snblas_gemm_cluster_kernel_deinit_fp64(const SnblasGemmInfo info, const SnblasGemmImpl impl);
inline __attribute__((always_inline)) void snblas_gemm_cluster_kernel_deinit_fp64(const SnblasGemmInfo info, const SnblasGemmImpl impl) {
    snrt_ssr_disable();
}

#if USE_C2C_TILES
#define SNRT_BARRIER_KERNEL(with_mcycle) \
snrt_global_barrier(); \
if (with_mcycle) snrt_mcycle(); 
#else
#define SNRT_BARRIER_KERNEL(with_mcycle) \
snrt_cluster_hw_barrier(); \
if (with_mcycle) snrt_mcycle(); 
#endif

extern void snblas_gemm_cluster_kernel_compute_fp64(const SnblasGemmInfo info, const SnblasGemmArgs_fp64 args, const SnblasGemmImpl impl);
inline __attribute__((always_inline)) __attribute__((flatten)) void snblas_gemm_cluster_kernel_compute_fp64(const SnblasGemmInfo info, const SnblasGemmArgs_fp64 args, const SnblasGemmImpl impl) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;
    const uint32_t tc  = info.tc;
    const uint32_t M   = info.M; // Compute fraction of C rows every core computes
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.lda;
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc;

    const double* const A = args.A + p[0] * (ta ? 1 : lda);
    const double* const B = args.B;
          double* const C = args.C;
    const double alpha    = args.alpha;
    // const double beta     = args.beta; // beta=1
    
    // Unroll by at least fmadd.d latency to fill pipeline
    // Additional unrolling reduces indexing overhead but needs available registers
    const uint32_t unroll = FMADD_D_UNROLL;
    
    snrt_fpu_fence(); // wait for previous fpu instructions to finish
    // snrt_cluster_hw_barrier(); 
    // snrt_global_barrier();
    // if (impl.bench) snrt_mcycle();
    SNRT_BARRIER_KERNEL(impl.bench)

    // SSR start address need to be configured each time
    // note: will start buffering elements into the fifo immediately, must be valid already
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void*) A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void*) B);

    // if (impl.bench) snrt_mcycle();
    for (uint32_t m = p[0]; m < M; m += P[0]) {
        uint32_t n = 0;
        for (; n < N; n += unroll) {
            double c[unroll];

            // Load intermediate result
            const uint32_t cIdx = m * ldc + n;
            // const uint32_t cIdx = m + n * ldc;
            c[0] = C[cIdx + 0];
            c[1] = C[cIdx + 1];
            c[2] = C[cIdx + 2];
            c[3] = C[cIdx + 3];
            c[4] = C[cIdx + 4];
            c[5] = C[cIdx + 5];
            c[6] = C[cIdx + 6];
            c[7] = C[cIdx + 7];

            asm volatile(
                "frep.o %[n_frep], %[unroll], 0, 0 \n"
                "fmadd.d %[c0], ft0, ft1, %[c0] \n"
                "fmadd.d %[c1], ft0, ft1, %[c1] \n"
                "fmadd.d %[c2], ft0, ft1, %[c2] \n"
                "fmadd.d %[c3], ft0, ft1, %[c3] \n"
                "fmadd.d %[c4], ft0, ft1, %[c4] \n"
                "fmadd.d %[c5], ft0, ft1, %[c5] \n"
                "fmadd.d %[c6], ft0, ft1, %[c6] \n"
                "fmadd.d %[c7], ft0, ft1, %[c7] \n"
                : [ c0 ] "+f"(c[0]), [ c1 ] "+f"(c[1]), [ c2 ] "+f"(c[2]),
                  [ c3 ] "+f"(c[3]), [ c4 ] "+f"(c[4]), [ c5 ] "+f"(c[5]),
                  [ c6 ] "+f"(c[6]), [ c7 ] "+f"(c[7])
                : [ n_frep ] "r"(K - 1), [ unroll ] "i"(unroll)
                : "ft0", "ft1", "ft2");

                // BUG: "r"(unroll) causes clang compiler to crash, use "i" instead

            // Store results back
            C[cIdx + 0] = c[0];
            C[cIdx + 1] = c[1];
            C[cIdx + 2] = c[2];
            C[cIdx + 3] = c[3];
            C[cIdx + 4] = c[4];
            C[cIdx + 5] = c[5];
            C[cIdx + 6] = c[6];
            C[cIdx + 7] = c[7];
        }
    }

}

/**
 * \brief Perform a one-time gemm computation for data in TCDM. 
 * Use the `init`, `compute` and `deinit` directly to get maximum performance when running multiple times.
*/
extern void snblas_gemm_cluster_kernel_fp64(const SnblasGemmInfo info, const SnblasGemmArgs_fp64 args, const SnblasGemmImpl impl);
inline __attribute__((always_inline)) void snblas_gemm_cluster_kernel_fp64(const SnblasGemmInfo info, const SnblasGemmArgs_fp64 args, const SnblasGemmImpl impl) {
    snblas_gemm_cluster_kernel_init_fp64(info, impl);
    snblas_gemm_cluster_kernel_compute_fp64(info, args, impl);
    snblas_gemm_cluster_kernel_deinit_fp64(info, impl);
}