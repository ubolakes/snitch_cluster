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
    // Setup layout for TCDM L1
    // For double buffering l1 is a size 2 array
    TcdmLayout* const l1    = (TcdmLayout*) snrt_l1_allocator()->base;
    bool l1ComputeBuffer_AB = false;
    bool l1ComputeBuffer_C  = false;

    // Initialize indices
    const int I  = 8, J  = 8, K = 4;
    const int PI = 2, PJ = 2;
    const int P  = PI * PJ;

    const int p  = snrt_cluster_idx();
    const int pi = p / PI;
    const int pj = p % PI;

    int ib     = 0, jb       = 0, kb        = 0;
    bool i_dir = true, j_dir = false, k_dir = false;

    if (snrt_is_compute_core()) {
        snrt_cluster_hw_barrier(); // DMA core is one index ahead
    }

    FOR_EACH(ib, pi, I - PI + pi +1, PI, i_dir) {
        j_dir = !j_dir;
        FOR_EACH(jb, pj, J - PJ + pj +1, PJ, j_dir) {
            // TODO: l1 buffer is decided by loop iter, dma is a step ahead so no need to do different calculations
            //       just toggle which one is used at each iter
            auto* const l1_C = l1[l1ComputeBuffer_C].C;

            // load next C
            snrt_dma_txid_t dma_tx_C = -1;
            if (snrt_is_dm_core()) {
                const auto* const l3_C_block = C + ib * ldc + jb; // C[i][j]
                auto* const l1_C_dma = l1[!l1ComputeBuffer_C].C;
                dma_tx_C = snrt_dma_start_2d(l1_C_dma, l3_C_block, l1_ldc, l1_ldc, ldc, sizeof(l1->C) / l1_ldc);
                // TODO: implement piecewise transfer of C in inner loop
            }
            k_dir = !k_dir;
            FOR_EACH(kb, 0, K, 1, k_dir) {
                // load next A, B
                if (snrt_is_dm_core()) {
                    const auto* const l3_A_block = A + ib * lda + kb; // A[i][k]
                    const auto* const l3_B_block = B + kb * ldb + jb; // B[k][j]

                    // TODO: transfer strided l3 block into a contiguous l1 block, use snrt_dma_start_2d
                    auto* const l1_A_dma = l1[!l1ComputeBuffer_AB].A;
                    auto* const l1_B_dma = l1[!l1ComputeBuffer_AB].B;

                    // TODO: use tile method from git:dnn/verification, just define above
                    const auto dma_tx_A = snrt_dma_start_2d(l1_A_dma, l3_A_block, l1_lda, l1_lda, lda,
                                                            sizeof(l1->A) / l1_lda);
                    const auto dma_tx_B = snrt_dma_start_2d(l1_B_dma, l3_B_block, l1_ldb, l1_ldb, ldb,
                                                            sizeof(l1->B) / l1_ldb);
                    snrt_dma_wait(dma_tx_A);
                    snrt_dma_wait(dma_tx_B);
                } else {
                    auto* const l1_A = l1[l1ComputeBuffer_AB].A;
                    auto* const l1_B = l1[l1ComputeBuffer_AB].B;
                    // solve block already in l1
                    gemm(FP64, 0, true, false, false,
                         m, n, k, alpha,
                         l1_A, l1_lda, l1_B, l1_ldb, beta, l1_C, l1_ldc);
                }

                l1ComputeBuffer_AB = !l1ComputeBuffer_AB; // switch buffers
                snrt_cluster_hw_barrier();
            }
            if (snrt_is_dm_core()) {
                snrt_dma_wait(dma_tx_C);
            }
            l1ComputeBuffer_C = !l1ComputeBuffer_C; // switch buffers
            snrt_cluster_hw_barrier();
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
