// GEMM implementation for OCCAMY

#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "ocrt.h"
#include "gemm_1c.h"

#include "dump.h"
NAMED_DUMP(uint32_t, aIdx, 0x1a)
NAMED_DUMP(uint32_t, bIdx, 0x1b)
NAMED_DUMP(uint32_t, cIdx, 0x1c)
NAMED_DUMP(uint32_t, ib, 0x10)
NAMED_DUMP(uint32_t, jb, 0x11)
NAMED_DUMP(uint32_t, kb, 0x12)
NAMED_DUMP(double, a, 0xa)
NAMED_DUMP(double, b, 0xb)
NAMED_DUMP(double, c, 0xc)

/**
 * \brief Implements a reversing loop for an index range
 * \param begin Beginning of the range
 * \param end End of the range
 * \param dir Sets the direction of traversal. True: loop starts at begin.
 * \param i_prev Set the previous index to the first index, must update this manually at the end of the loop.
 * \details i_end_floor will contain the exact end with the stride, s.t. the reversed loop starts at the correct index.
 */
#define FOR_EACH(i, begin, end, stride, dir, i_prev)                                                                   \
    dir = !dir;                                                                                                        \
    const int i##_end_floor = ((end - begin + stride - 1) / stride) * stride - stride + begin;                         \
    const int i##_first     = dir ? begin : i##_end_floor;                                                             \
    const int i##_last      = dir ? i##_end_floor : begin;                                                             \
    i                       = i##_first;                                                                               \
    i_prev                  = i;                                                                                       \
    for (; dir ? i <= i##_last : i >= i##_last; i = dir ? i + stride : i - stride)

#define L1_M 8 //128;
#define L1_N 8 //128;
#define L1_K 8 //128;
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
void gemm_cluster_kernel(double alpha, double beta,
                         uint32_t M, uint32_t N, uint32_t K,
                         double* const A, double* const B, double* const C,
                         int lda, int ldb, int ldc) {
    uint32_t p[3], P[3];
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    for (uint32_t i = p[0]; i < M; i += P[0]) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t cIdx = i * ldc + j; // C[i][j]
            // dump_cIdx(cIdx);
            // dump_c(C[cIdx]);
            register double c0 = beta * C[cIdx];
            for (uint32_t k = 0; k < K; k++) {
                uint32_t aIdx = i * lda + k; // A[i][k]
                uint32_t bIdx = k * ldb + j; // B[k][j]
                // dump_aIdx(aIdx);
                // dump_bIdx(bIdx);
                // dump_a(A[aIdx]);
                // dump_b(B[bIdx]);

                c0 += A[aIdx] * B[bIdx];
            }
            C[cIdx] = c0;
        }
    }
    snrt_fpu_fence();
}

void gemm_oc_baseline(double alpha, double beta,
                      uint32_t m, uint32_t n, uint32_t k,
                      double* A, double* B, double* C,
                      uint32_t lda, uint32_t ldb, uint32_t ldc) {
    /**
    * Problem is double buffered in L1. The buffer that is used is toggled at each iteration.
    * The DMA cores are one index step ahead so they load the data in advance into the buffer that will be used.
    *
    */

    // Setup layout for TCDM L1
    // For double buffering l1 is a size 2 array
    TcdmLayout* l1 = (TcdmLayout*) snrt_l1_next();
    // if (snrt_is_dm_core()) {
    //     l1 = (TcdmLayout*) snrt_l1alloc(2 * sizeof(TcdmLayout));
    // }
    // snrt_cluster_hw_barrier(); // DMA core is one index ahead
    // dump_l1(l1);


    bool l1Id_AB = false;
    bool l1Id_C  = false;

    // Initialize indices
    const uint32_t I = m, J = n, K = k;

    volatile uint32_t p[3] = {0, 0, 0};
    volatile uint32_t P[3] = {0, 0, 0};
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    const uint32_t PI = P[1], PJ = 1;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev, jb_prev, kb_prev;
    bool ib_dir = false, jb_dir = false, kb_dir = false;

    bool storeC = false;

    // Debug
    volatile int ib_cnt = 0, jb_cnt = 0, kb_cnt = 0;

    if (snrt_is_compute_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }

    // FOR_EACH(ib, pi, I / L1_M, PI, ib_dir, ib_prev) {
    ib_dir                 = !ib_dir;
    const int ib_end_floor = ((I / 8 - pi + PI - 1) / PI) * PI - PI + pi;
    const int ib_first     = ib_dir ? pi : ib_end_floor;
    const int ib_last      = ib_dir ? ib_end_floor : pi;
    ib                     = ib_first;
    ib_prev                = ib;
    for (; ib_dir ? ib <= ib_last : ib >= ib_last; ib = ib_dir ? ib + PI : ib - PI) {
        ib_cnt += ib;
        // FOR_EACH(jb, pj, J / L1_N, PJ, jb_dir, jb_prev) {
        jb_dir                 = !jb_dir;
        const int jb_end_floor = ((J / 8 - pj + PJ - 1) / PJ) * PJ - PJ + pj;
        const int jb_first     = jb_dir ? pj : jb_end_floor;
        const int jb_last      = jb_dir ? jb_end_floor : pj;
        jb                     = jb_first;
        jb_prev                = jb;
        for (; jb_dir ? jb <= jb_last : jb >= jb_last; jb = jb_dir ? jb + PJ : jb - PJ) {
            jb_cnt += jb;

            double* const l1_C = l1[l1Id_C].C;

            if (snrt_is_dm_core()) {
                dump_ib(ib);
                dump_jb(jb);
                snrt_dma_load_2d_tile(l1_C, C, ib, jb, L1_M, L1_N, ldc, FP64);
                if (ib != ib_first || jb != jb_first)
                    storeC = true;
            }

            // FOR_EACH(kb, 0, K / L1_K, 1, kb_dir, kb_prev) {
            kb_dir                 = !kb_dir;
            const int kb_end_floor = ((K / L1_K - 0 + 1 - 1) / 1) * 1 - 1 + 0;
            const int kb_first     = kb_dir ? 0 : kb_end_floor;
            const int kb_last      = kb_dir ? kb_end_floor : 0;
            kb                     = kb_first;
            kb_prev                = kb;
            for (; kb_dir ? kb <= kb_last : kb >= kb_last; kb = kb_dir ? kb + 1 : kb - 1) {
                kb_cnt += kb;
                double* const l1_A = l1[l1Id_AB].A;
                double* const l1_B = l1[l1Id_AB].B;

                // load next A, B
                if (snrt_is_dm_core()) {
                    snrt_dma_load_2d_tile(l1_A, A, ib, kb, L1_M, L1_K, lda, FP64);
                    snrt_dma_load_2d_tile(l1_B, B, kb, jb, L1_K, L1_N, ldb, FP64);

                    snrt_dma_wait_all();
                } else {
                    // solve block already in l1, parallelize inside each cluster
                    gemm_cluster_kernel(alpha, beta, L1_M, L1_N, L1_K, l1_A, l1_B, l1_C, L1_LDA, L1_LDB, L1_LDC);

                    // gemm(FP64, 0, true, false, false,
                    //      m, n, k, alpha,
                    //      l1_A, l1_lda, l1_B, l1_ldb, beta, l1_C, l1_ldc);
                }

                l1Id_AB = !l1Id_AB; // switch buffers
                snrt_cluster_hw_barrier();

                if (snrt_is_dm_core()) {
                    if (storeC) {
                        storeC = false;
                        snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N, ldc, FP64);
                    }
                }
                kb_prev = kb;
            }

            l1Id_C  = !l1Id_C; // switch buffers
            jb_prev = jb;
            ib_prev = ib;
        }
    }

    if (snrt_is_dm_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead

        // store final tile
        snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N, ldc, FP64);
        snrt_dma_wait_all();
    }

    // Free memory once implemented by snrt
    // snrt_l1free(l1);
}

inline void gemm_oc(precision_t prec, uint32_t expand, uint32_t setup_ssr,
                    uint32_t transa, uint32_t transb, uint32_t m, uint32_t n,
                    uint32_t k, double alpha, void* a, uint32_t lda, void* b,
                    uint32_t ldb, uint32_t beta, void* c, uint32_t ldc) {
    // gemm_cluster_kernel(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
    // snrt_fpu_fence();
    // snrt_cluster_hw_barrier();

    gemm_oc_baseline(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
}
