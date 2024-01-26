#include "gemm_decls.h"
#define FLOAT_T fp8
#include "gemm_kernel_init_tpl.h"

extern void SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args);
inline void SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    const uint32_t M   = info.M / P[0]; // Compute fraction of C rows every core computes
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.lda * P[0];
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc * P[0];
    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;

    const FLOAT_T* const A = args.A + p[0] * info.lda;
    const FLOAT_T* const B = args.B;
          FLOAT_T* const C = args.C + p[0] * info.ldc;
    const FLOAT_T alpha    = args.alpha;
    const FLOAT_T beta     = args.beta;
    
    const uint32_t unroll = 8;

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void*)A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void*)B);

    // Kernel progresses by 8 values each step
    const uint32_t n_frep = K / 8 - 1;

    for (uint32_t m = 0; m < M; m++) {
        uint32_t n = 0;
        for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
            FLOAT_T* _C = &C[m * ldc + n];
            const register float zero = 0.0;
            v8f8 c[unroll];
            v4f16 reduce_reg[unroll];
            uint32_t alpha;

            asm volatile(
                "lw      %[alpha], 0(%[beta]) \n"
                "beqz    %[alpha], 1f \n"
                "flb %[reduce_reg0], 0(%[C]) \n"
                "flb %[reduce_reg1], 1(%[C]) \n"
                "flb %[reduce_reg2], 2(%[C]) \n"
                "flb %[reduce_reg3], 3(%[C]) \n"
                "flb %[reduce_reg4], 4(%[C]) \n"
                "flb %[reduce_reg5], 5(%[C]) \n"
                "flb %[reduce_reg6], 6(%[C]) \n"
                "flb %[reduce_reg7], 7(%[C]) \n"
                // Convert intermediate results before packing
                "vfcvt.s.b %[reduce_reg0], %[reduce_reg0]\n"
                "vfcvt.s.b %[reduce_reg1], %[reduce_reg1]\n"
                "vfcvt.s.b %[reduce_reg2], %[reduce_reg2]\n"
                "vfcvt.s.b %[reduce_reg3], %[reduce_reg3]\n"
                "vfcvt.s.b %[reduce_reg4], %[reduce_reg4]\n"
                "vfcvt.s.b %[reduce_reg5], %[reduce_reg5]\n"
                "vfcvt.s.b %[reduce_reg6], %[reduce_reg6]\n"
                "vfcvt.s.b %[reduce_reg7], %[reduce_reg7]\n"
                // Initialize reduce register to zero
                "vfcpka.s.s %[c0], %[zero], %[zero]\n"
                "vfcpka.s.s %[c1], %[zero], %[zero]\n"
                "vfcpka.s.s %[c2], %[zero], %[zero]\n"
                "vfcpka.s.s %[c3], %[zero], %[zero]\n"
                "vfcpka.s.s %[c4], %[zero], %[zero]\n"
                "vfcpka.s.s %[c5], %[zero], %[zero]\n"
                "vfcpka.s.s %[c6], %[zero], %[zero]\n"
                "vfcpka.s.s %[c7], %[zero], %[zero]\n"
                // Pack intermediate results into SIMD vector
                "vfcpka.h.s %[c0], %[reduce_reg0], %[zero]\n"
                "vfcpka.h.s %[c1], %[reduce_reg1], %[zero]\n"
                "vfcpka.h.s %[c2], %[reduce_reg2], %[zero]\n"
                "vfcpka.h.s %[c3], %[reduce_reg3], %[zero]\n"
                "vfcpka.h.s %[c4], %[reduce_reg4], %[zero]\n"
                "vfcpka.h.s %[c5], %[reduce_reg5], %[zero]\n"
                "vfcpka.h.s %[c6], %[reduce_reg6], %[zero]\n"
                "vfcpka.h.s %[c7], %[reduce_reg7], %[zero]\n"
                "j 2f \n"
                "1: \n"
                // Initialize SIMD vector with zeros
                "vfcpka.s.s %[c0], %[zero], %[zero]\n"
                "vfcpka.s.s %[c1], %[zero], %[zero]\n"
                "vfcpka.s.s %[c2], %[zero], %[zero]\n"
                "vfcpka.s.s %[c3], %[zero], %[zero]\n"
                "vfcpka.s.s %[c4], %[zero], %[zero]\n"
                "vfcpka.s.s %[c5], %[zero], %[zero]\n"
                "vfcpka.s.s %[c6], %[zero], %[zero]\n"
                "vfcpka.s.s %[c7], %[zero], %[zero]\n"
                "2: \n"
                // Perform expanding sum-dotproducts
                "frep.o  %[n_frep], 8, 0, 0 \n"
                "vfdotpex.h.b %[c0], ft1, ft0 \n"
                "vfdotpex.h.b %[c1], ft1, ft0 \n"
                "vfdotpex.h.b %[c2], ft1, ft0 \n"
                "vfdotpex.h.b %[c3], ft1, ft0 \n"
                "vfdotpex.h.b %[c4], ft1, ft0 \n"
                "vfdotpex.h.b %[c5], ft1, ft0 \n"
                "vfdotpex.h.b %[c6], ft1, ft0 \n"
                "vfdotpex.h.b %[c7], ft1, ft0 \n"
                // Initialize reduce register to zero
                "vfcpka.s.s %[reduce_reg0], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg1], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg2], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg3], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg4], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg5], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg6], %[zero], %[zero]\n"
                "vfcpka.s.s %[reduce_reg7], %[zero], %[zero]\n"
                // Sum-reduce vector
                "vfsumex.s.h %[reduce_reg0], %[c0] \n"
                "vfsumex.s.h %[reduce_reg1], %[c1] \n"
                "vfsumex.s.h %[reduce_reg2], %[c2] \n"
                "vfsumex.s.h %[reduce_reg3], %[c3] \n"
                "vfsumex.s.h %[reduce_reg4], %[c4] \n"
                "vfsumex.s.h %[reduce_reg5], %[c5] \n"
                "vfsumex.s.h %[reduce_reg6], %[c6] \n"
                "vfsumex.s.h %[reduce_reg7], %[c7] \n"
                //
                // Initialize reduce register to zero
                "vfcpka.s.s %[c0], %[zero], %[zero] \n"
                "vfcpka.s.s %[c1], %[zero], %[zero] \n"
                "vfcpka.s.s %[c2], %[zero], %[zero] \n"
                "vfcpka.s.s %[c3], %[zero], %[zero] \n"
                "vfcpka.s.s %[c4], %[zero], %[zero] \n"
                "vfcpka.s.s %[c5], %[zero], %[zero] \n"
                "vfcpka.s.s %[c6], %[zero], %[zero] \n"
                "vfcpka.s.s %[c7], %[zero], %[zero] \n"
                // Sum-reduce vector
                "vfsum.s %[c0], %[reduce_reg0] \n"
                "vfsum.s %[c1], %[reduce_reg1] \n"
                "vfsum.s %[c2], %[reduce_reg2] \n"
                "vfsum.s %[c3], %[reduce_reg3] \n"
                "vfsum.s %[c4], %[reduce_reg4] \n"
                "vfsum.s %[c5], %[reduce_reg5] \n"
                "vfsum.s %[c6], %[reduce_reg6] \n"
                "vfsum.s %[c7], %[reduce_reg7] \n"
                // Pack and convert results to FP8 vectors
                "vfcpka.b.s %[c0], %[c0], %[c1] \n"
                "vfcpkb.b.s %[c0], %[c2], %[c3] \n"
                "vfcpkc.b.s %[c0], %[c4], %[c5] \n"
                "vfcpkd.b.s %[c0], %[c6], %[c7] \n"
                // // // Pack and convert results to FP8 vectors
                // "vfcpka.b.s %[c0], %[reduce_reg0], %[reduce_reg1] \n"
                // "vfcpkb.b.s %[c0], %[reduce_reg2], %[reduce_reg3] \n"
                // "vfcpkc.b.s %[c0], %[reduce_reg4], %[reduce_reg5] \n"
                // "vfcpkd.b.s %[c0], %[reduce_reg6], %[reduce_reg7] \n"
                : [ c0 ] "+f"(c[0]), [ c1 ] "+f"(c[1]), [ c2 ] "+f"(c[2]),
                  [ c3 ] "+f"(c[3]), [ c4 ] "+f"(c[4]), [ c5 ] "+f"(c[5]),
                  [ c6 ] "+f"(c[6]), [ c7 ] "+f"(c[7]), [ alpha ] "=r"(alpha),
                  [ reduce_reg0 ] "+f"(reduce_reg[0]),
                  [ reduce_reg1 ] "+f"(reduce_reg[1]),
                  [ reduce_reg2 ] "+f"(reduce_reg[2]),
                  [ reduce_reg3 ] "+f"(reduce_reg[3]),
                  [ reduce_reg4 ] "+f"(reduce_reg[4]),
                  [ reduce_reg5 ] "+f"(reduce_reg[5]),
                  [ reduce_reg6 ] "+f"(reduce_reg[6]),
                  [ reduce_reg7 ] "+f"(reduce_reg[7])
                : [ C ] "r"(_C), [ n_frep ] "r"(n_frep), [ beta ] "r"(&beta),
                  [ zero ] "f"(zero)
                : "ft0", "ft1", "ft2");

            // Store results back
            ((v8f8*)_C)[0] = c[0];
            n += unroll;
        }

        // Clean up left over column
        // snrt_ssr_disable();

        // for (; n < N; n++) {
        //     char c = beta ? C[m * ldc + n] : 0.0;
        //     for (uint32_t k = 0; k < K; k++) {
        //         c += A[k + m * lda] * B[k + n * ldb];
        //     }
        //     C[m * ldc + n] = c;
        // }

        // snrt_ssr_enable();
    }

    snrt_fpu_fence();
}

#undef FLOAT_T
