// GEMM implementation for OCCAMY

#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "snrt.h"
#include "gemm_1c.h"

/**
 * \brief Implements a reversing loop for an index range
 * \param begin Beginning of the range
 * \param end End of the range
 * \param dir Sets the direction of traversal. True: loop starts at begin.
 * \details i_end_floor will contain the exact end with the stride, s.t. the reversed loop starts at the correct index.
 */
#define FOR_EACH(i, begin, end, stride, dir)                                                                           \
  dir = !dir;                                                                                                          \
  const int i##_end_floor = ((end - begin + stride - 1) / stride) * stride - stride + begin;                           \
  const int i##_first = dir ? begin : i##_end_floor;                                                                   \
  const int i##_last = dir ? i##_end_floor : begin;                                                                    \
  for (i = i##_first; dir ? i <= i##_last : i >= i##_last; i = dir ? i + stride : i - stride)

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

/**
 * \brief Each cluster performs a GEMM for A, B, C inside each TCDM
 */
void gemm_cluster_kernel(double alpha, double beta,
                         uint32_t M, uint32_t N, uint32_t K,
                         double* const A, double* const B, double* const C,
                         int lda, int ldb, int ldc) {
    const int P = snrt_cluster_core_num();
    const int p = snrt_cluster_core_idx();

    for (uint32_t i = p; i < M; i += P) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t cIdx      = i * ldc + j; // C[i][j]
            register double c0 = beta * C[cIdx];
            for (uint32_t k = 0; k < K; k++) {
                uint32_t aIdx = i * lda + k; // A[i][k]
                uint32_t bIdx = k * ldb + j; // B[k][j]
                c0 += A[aIdx] * B[bIdx];
            }
            C[cIdx] = c0;
        }
    }
}

void gemm_oc_baseline(double alpha, double beta,
                      uint32_t m, uint32_t n, uint32_t k,
                      void* A, void* B, void* C,
                      uint32_t lda, uint32_t ldb, uint32_t ldc) {
    /**
    * Problem is double buffered in L1. The buffer that is used is toggled at each iteration.
    * The DMA cores are one index step ahead so they load the data in advance into the buffer that will be used.
    *
    */

    // Setup layout for TCDM L1
    // For double buffering l1 is a size 2 array
    TcdmLayout* const l1 = (TcdmLayout*) snrt_l1alloc(2 * sizeof(TcdmLayout));
    bool l1Id_AB         = false;
    bool l1Id_C          = false;

    // Initialize indices
    const int I = m, J = n, K = k;

    const int PI = snrt_cluster_num(), PJ = 1;
    const int P  = PI * PJ;

    const int p  = snrt_cluster_idx();
    const int pi = 0; // p / PJ;
    const int pj = p; // p % PJ;

    int ib, jb, kb;
    bool i_dir = false, j_dir = false, k_dir = false;

    if (snrt_is_compute_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }


    // TODO: check that no out of bound accesses are made
    // TODO: check that dma transfers are finished before the data is used, use a barrier
    FOR_EACH(ib, pi, I / L1_M, PI, i_dir) {
        FOR_EACH(jb, pj, J / L1_N, PJ, j_dir) {
            const int i = ib * L1_LDA;
            const int j = jb * L1_LDB;

            double* const l1_C = l1[l1Id_C].C;

            snrt_dma_txid_t dma_txid_load_C  = -1;
            snrt_dma_txid_t dma_txid_store_C = -1;

            // load next C
            if (snrt_is_dm_core()) {
                // TODO: implement piecewise transfer of C in inner loop
                snrt_dma_load_2d_tile(l1_C, c, ib, jb, L1_M, L1_N, ldc, FP64);
            }

            FOR_EACH(kb, 0, K / L1_K, 1, k_dir) {
                double* const l1_A = l1[l1Id_AB].A;
                double* const l1_B = l1[l1Id_AB].B;

                // load next A, B
                if (snrt_is_dm_core()) {
                    snrt_dma_load_2d_tile(l1_A, a, ib, kb, L1_M, L1_K, lda, FP64);
                    snrt_dma_load_2d_tile(l1_B, b, kb, jb, L1_K, L1_N, ldb, FP64);

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
            }

            l1Id_C = !l1Id_C; // switch buffers

            if (snrt_is_dm_core()) {
                // store C
                snrt_dma_store_2d_tile(c, l1_C, ib, jb, L1_M, L1_N, ldc, FP64);
            }
        }
    }

    if (snrt_is_dm_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }

    // Free memory once implemented by snrt
    // snrt_l1free(l1);
}

inline void gemm_oc(precision_t prec, uint32_t expand, uint32_t setup_ssr,
                    uint32_t transa, uint32_t transb, uint32_t m, uint32_t n,
                    uint32_t k, double alpha, void* a, uint32_t lda, void* b,
                    uint32_t ldb, uint32_t beta, void* c, uint32_t ldc) {
    gemm_cluster_kernel(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
    snrt_cluster_hw_barrier();

    // gemm_oc_baseline(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
}
