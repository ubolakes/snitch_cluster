// GEMM implementation for OCCAMY

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "snrt.h"
#include "gemm_1c.h"

/**
 * \brief Implements a reversing loop for an index range
 * \param end Must be exact, for correct reversing behavior
 * \param dir Sets the direction of traversal. True: loop starts at begin.
 */
#define FOR_EACH(i, begin, end, stride, dir)                                   \
  dir = !dir;                                                                  \
  const int first = dir ? begin : end - 1;                                     \
  const int last = dir ? end : begin;                                          \
  for (i = first; dir ? i < last : i >= last; dir ? i += stride : i -= stride)


const int l1_M   = 128;
const int l1_N   = 128;
const int l1_K   = 128;
const int l1_lda = l1_M;
const int l1_ldb = l1_K;
const int l1_ldc = l1_M;

/**
 * \brief Maps the layout of the TCDM. May be double buffered.
 */
typedef struct {
    double A[l1_M * l1_K];
    double B[l1_K * l1_N];
    double C[l1_M * l1_N];
} TcdmLayout;

static_assert(sizeof(TcdmLayout) < snrt_l1_allocator()->size, "TCDM size is exceeded for single buffering.");


inline void gemm_oc_baseline(double alpha, double beta,
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
    TcdmLayout* const l1 = (TcdmLayout*) snrt_l1_allocator()->next;
    bool l1Id_AB         = false;
    bool l1Id_C          = false;

    // Initialize indices
    const int I  = m, J  = n, K = k;
    const int PI = 2, PJ = 2;
    const int P  = PI * PJ;

    const int p  = snrt_cluster_idx();
    const int pi = p / PI;
    const int pj = p % PI;

    int ib     = 0, jb        = 0, kb        = 0;
    bool i_dir = false, j_dir = false, k_dir = false;

    if (snrt_is_compute_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }

    FOR_EACH(ib, pi, I - PI + pi +1, PI, i_dir) {
        FOR_EACH(jb, pj, J - PJ + pj +1, PJ, j_dir) {
            const auto i = ib * l1_lda;
            const auto j = jb * l1_ldb;

            auto* const l1_C = l1[l1Id_C].C;

            // load next C
            snrt_dma_txid_t dma_tx_C = -1;
            if (snrt_is_dm_core()) {
                dma_tx_C = snrt_dma_load_2d_tile(l1_C, c, ib, jb, l1_M, l1_N, ldc, FP64);
                // TODO: implement piecewise transfer of C in inner loop
            }

            FOR_EACH(kb, 0, K, 1, k_dir) {
                auto* const l1_A = l1[l1Id_AB].A;
                auto* const l1_B = l1[l1Id_AB].B;

                // load next A, B
                if (snrt_is_dm_core()) {
                    snrt_dma_load_2d_tile(l1_A, a, ib, kb, l1_M, l1_K, lda, FP64);
                    snrt_dma_load_2d_tile(l1_B, b, kb, jb, l1_K, l1_N, ldb, FP64);

                    snrt_dma_wait_all();
                } else {
                    // solve block already in l1, parallelize inside each cluster
                    gemm(FP64, 0, true, false, false,
                         m, n, k, alpha,
                         l1_A, l1_lda, l1_B, l1_ldb, beta, l1_C, l1_ldc);
                }

                l1Id_AB = !l1Id_AB; // switch buffers
                snrt_cluster_hw_barrier();
            }

            if (snrt_is_dm_core()) {
                snrt_dma_wait(dma_tx_C);
            }

            l1Id_C = !l1Id_C; // switch buffers
            snrt_cluster_hw_barrier();

            if(snrt_is_dm_core()) { // store C
                dma_tx_C = snrt_dma_store_2d_tile(c, l1_C,
                                       ib, jb,
                                       l1_M, l1_N, ldc, FP64);

                snrt_dma_wait(dma_tx_C);
            }
        }
    }

    if (snrt_is_dm_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }
}

inline void gemm_oc(precision_t prec, uint32_t expand, uint32_t setup_ssr,
                    uint32_t transa, uint32_t transb, uint32_t m, uint32_t n,
                    uint32_t k, double alpha, void* a, uint32_t lda, void* b,
                    uint32_t ldb, uint32_t beta, void* c, uint32_t ldc) {
    gemm_oc_baseline(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
}
