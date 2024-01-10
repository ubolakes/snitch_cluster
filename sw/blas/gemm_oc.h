// GEMM implementation for OCCAMY

#pragma once

#include <stdint.h>
#include <stdbool.h>

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
  for (i = first; dir ? i < last : i >= last; i = dir ? i + stride : i - stride)


const int l1_M   = 8; //128;
const int l1_N   = 8; //128;
const int l1_K   = 8; //128;
const int l1_lda = l1_K;
const int l1_ldb = l1_N;
const int l1_ldc = l1_N;

/**
 * \brief Maps the layout of the TCDM. May be double buffered.
 */
typedef struct {
    double A[l1_M * l1_K];
    double B[l1_K * l1_N];
    double C[l1_M * l1_N];
} TcdmLayout;

void gemm_kernel(double alpha, double beta,
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
    // FOR_EACH(ib, pi, I - PI + pi +1, PI, i_dir) {
    i_dir             = !i_dir;
    const int i_first = i_dir ? pi : I - PI + pi + 1 - 1;
    const int i_last  = i_dir ? I - PI + pi + 1 : pi;
    for (ib = i_first; i_dir ? ib < i_last : ib >= i_last; ib = i_dir ? ib + PI : ib - PI) {
        // FOR_EACH(jb, pj, J - PJ + pj +1, PJ, j_dir) {
        j_dir             = !j_dir;
        const int j_first = j_dir ? pj : J - PJ + pj + 1 - 1;
        const int j_last  = j_dir ? J - PJ + pj + 1 : pj;
        for (jb = j_first; j_dir ? jb < j_last : jb >= j_last; jb = j_dir ? jb + PJ : jb - PJ) {
            const int i = ib * l1_lda;
            const int j = jb * l1_ldb;

            double* const l1_C = l1[l1Id_C].C;

            // load next C
            snrt_dma_txid_t dma_tx_C = -1;
            if (snrt_is_dm_core()) {
                dma_tx_C = snrt_dma_load_2d_tile(l1_C, c, ib, jb, l1_M, l1_N, ldc, FP64);
                // TODO: implement piecewise transfer of C in inner loop
                snrt_dma_wait_all(); // TODO: only wait here the first time
            }

            // TODO: dma wait needs a barrier before compute cores can use the data
            snrt_cluster_hw_barrier();

            // FOR_EACH(kb, 0, K, 1, k_dir) {
            k_dir             = !k_dir;
            const int k_first = k_dir ? 0 : K - 1;
            const int k_last  = k_dir ? K : 0;
            for (kb = k_first; k_dir ? kb < k_last : kb >= k_last; kb = k_dir ? kb + 1 : kb - 1) {
                double* const l1_A = l1[l1Id_AB].A;
                double* const l1_B = l1[l1Id_AB].B;

                // load next A, B
                if (snrt_is_dm_core()) {
                    snrt_dma_load_2d_tile(l1_A, a, ib, kb, l1_M, l1_K, lda, FP64);
                    snrt_dma_load_2d_tile(l1_B, b, kb, jb, l1_K, l1_N, ldb, FP64);

                    snrt_dma_wait_all();
                } else {
                    // solve block already in l1, parallelize inside each cluster
                    gemm_kernel(alpha, beta, l1_M, l1_N, l1_K, l1_A, l1_B, l1_C, l1_lda, l1_ldb, l1_ldc);

                    // gemm(FP64, 0, true, false, false,
                    //      m, n, k, alpha,
                    //      l1_A, l1_lda, l1_B, l1_ldb, beta, l1_C, l1_ldc);
                }

                l1Id_AB = !l1Id_AB; // switch buffers
                snrt_cluster_hw_barrier();
            }

            // if (snrt_is_dm_core()) {
            //     snrt_dma_wait(dma_tx_C); // TODO: don't wait the first time
            // }

            l1Id_C = !l1Id_C; // switch buffers
            snrt_cluster_hw_barrier();

            if (snrt_is_dm_core()) {
                // store C
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

    // Free memory once implemented by snrt
    // snrt_l1free(l1);
}

inline void gemm_oc(precision_t prec, uint32_t expand, uint32_t setup_ssr,
                    uint32_t transa, uint32_t transb, uint32_t m, uint32_t n,
                    uint32_t k, double alpha, void* a, uint32_t lda, void* b,
                    uint32_t ldb, uint32_t beta, void* c, uint32_t ldc) {
    gemm_kernel(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
}
