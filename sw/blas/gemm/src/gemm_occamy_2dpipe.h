// GEMM implementation for OCCAMY with a 2D tile pipeline

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "gemm_kernel.h"
#include "snrt.h"

#include "dump.h"
NAMED_DUMP(uint32_t, aIdx, 0x1a)
NAMED_DUMP(uint32_t, bIdx, 0x1b)
NAMED_DUMP(uint32_t, cIdx, 0x1c)
NAMED_DUMP(uint32_t, ib, 0x10)
NAMED_DUMP(uint32_t, jb, 0x11)
NAMED_DUMP(uint32_t, kb, 0x12)
NAMED_DUMP(uint32_t, pk, 0x13)
NAMED_DUMP(uint32_t, p_src, 0x14)
NAMED_DUMP(double, a, 0xa)
NAMED_DUMP(double, b, 0xb)
NAMED_DUMP(double, c, 0xc)

/**
 * \brief Implements a reversing loop for an index range
 * \param begin Beginning of the range
 * \param end End of the range
 * \param dir Sets the direction of traversal. True: loop starts at begin.
 * \param i_prev Set the previous index to the first index, must update this
 * manually at the end of the loop. \details i_end_floor will contain the exact
 * end with the stride, s.t. the reversed loop starts at the correct index.
 */
#define FOR_EACH(i, begin, end, stride, dir, i_prev)                     \
    dir = !dir;                                                          \
    const int i##_end_floor =                                            \
        ((end - begin + stride - 1) / stride) * stride - stride + begin; \
    const int i##_first = dir ? begin : i##_end_floor;                   \
    const int i##_last = dir ? i##_end_floor : begin;                    \
    i = i##_first;                                                       \
    for (; dir ? i <= i##_last : i >= i##_last;                          \
         i = dir ? i + stride : i - stride)

#define L1_M 8
#define L1_N 8
#define L1_K 8
#define L1_LDA L1_K
#define L1_LDB L1_N
#define L1_LDC L1_N

/**
 * \brief Maps the layout of the TCDM. May be double buffered.
 */
typedef struct {
    double A[L1_M * L1_K];
    double B[L1_K * L1_N];
    double C[L1_M * L1_N];
} TcdmLayout;

NAMED_DUMP(TcdmLayout*, l1, 0x8)

/**
 * \brief Each cluster performs a GEMM for A, B, C inside each TCDM
 */
void gemm_cluster_kernel_baseline(double alpha, double beta, uint32_t M, uint32_t N,
                         uint32_t K, double* const A, double* const B,
                         double* const C, int lda, int ldb, int ldc) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    for (uint32_t i = p[0]; i < M; i += P[0]) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t cIdx = i * ldc + j;  // C[i][j]
            register double c0 = beta * C[cIdx];

            for (uint32_t k = 0; k < K; k++) {
                uint32_t aIdx = i * lda + k;  // A[i][k]
                uint32_t bIdx = k * ldb + j;  // B[k][j]

                c0 += A[aIdx] * B[bIdx];
            }
            C[cIdx] = c0;
        }
    }
    snrt_fpu_fence();
}

/// Constants related to a GEMM computation to precompute and initialize
typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t ta;
    uint32_t tb;
} GemmInfo;

/// Arguments to execute a GEMM computation, given a corresponding GemmInfo instance
typedef struct {
    const double* A;
    const double* B;
    double* C;
    double alpha;
    double beta;
} GemmArgs;

extern void gemm_cluster_kernel_init(const GemmInfo info);
inline void gemm_cluster_kernel_init(const GemmInfo info) {
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
    // First matrix is stored in transposed format
    if (ta) {
        const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr0_i[4] = {0, 8 * lda, 0, 8 * 8};

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                         ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
    } else {
        const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr0_i[4] = {0, 8, 0, 8 * lda};

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                         ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
    }

    // Second matrix is stored in transposed format
    if (tb) {
        const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr1_i[4] = {8 * ldb, 8, 8 * ldb * unroll, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                         ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                         ssr1_i[3]);
    } else {
        const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr1_i[4] = {8, 8 * ldb, 8 * unroll, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                         ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                         ssr1_i[3]);
    }

    snrt_ssr_enable();
}

extern void gemm_cluster_kernel_deinit(const GemmInfo info);
inline void gemm_cluster_kernel_deinit(const GemmInfo info) {
    snrt_ssr_disable();
}

extern void gemm_cluster_kernel(const GemmInfo info, const GemmArgs args);
inline void gemm_cluster_kernel(const GemmInfo info, const GemmArgs args) {
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

    const double* const A = args.A + p[0] * info.lda;
    const double* const B = args.B;
          double* const C = args.C + p[0] * info.ldc;
    const double alpha    = args.alpha;
    const double beta     = args.beta;
    
    const uint32_t unroll = 8;

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);

    for (uint32_t m = 0; m < M; m++) {
        uint32_t n = 0;
        for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
            double c[unroll];

            // Load intermediate result
            const uint32_t cIdx = m * ldc + n;
            if (beta != 0.0) {
                snrt_ssr_disable(); // Need to disable ssr to use ft0, ft1 for fmul
                c[0] = beta * C[cIdx + 0];
                c[1] = beta * C[cIdx + 1];
                c[2] = beta * C[cIdx + 2];
                c[3] = beta * C[cIdx + 3];
                c[4] = beta * C[cIdx + 4];
                c[5] = beta * C[cIdx + 5];
                c[6] = beta * C[cIdx + 6];
                c[7] = beta * C[cIdx + 7];
                snrt_ssr_enable();
            }
            else {
                c[0] = 0.0;
                c[1] = 0.0;
                c[2] = 0.0;
                c[3] = 0.0;
                c[4] = 0.0;
                c[5] = 0.0;
                c[6] = 0.0;
                c[7] = 0.0;
            }

            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
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
                : [ n_frep ] "r"(K - 1)
                : "ft0", "ft1", "ft2");

            // Store results back
            C[cIdx + 0] = c[0];
            C[cIdx + 1] = c[1];
            C[cIdx + 2] = c[2];
            C[cIdx + 3] = c[3];
            C[cIdx + 4] = c[4];
            C[cIdx + 5] = c[5];
            C[cIdx + 6] = c[6];
            C[cIdx + 7] = c[7];
            n += unroll;
        }
    }

    snrt_fpu_fence();
}

#define IS_DM_CORE true
#include "gemm_occamy_2dpipe_tpl.h"
#define IS_DM_CORE false
#include "gemm_occamy_2dpipe_tpl.h"
#undef IS_DM_CORE

void gemm_oc (const GemmInfo info, const GemmArgs args) {
    if (snrt_is_dm_core()) {
        gemm_oc_dm(info, args);
    } else {
        gemm_oc_compute(info, args);
    }
}
