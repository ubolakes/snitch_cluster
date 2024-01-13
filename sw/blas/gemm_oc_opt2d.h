// GEMM implementation for OCCAMY

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "ocrt.h"
#include "gemm_1c.h"

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

TcdmLayout* l1AddrGlobal[SNRT_CLUSTER_NUM] = {0};

void gemm_oc_opt2d(double alpha, double beta,
                   uint32_t m, uint32_t n, uint32_t k,
                   double* A, double* B, double* C,
                   uint32_t lda, uint32_t ldb, uint32_t ldc) {
    /**
    * Problem is double buffered in L1. The buffer that is used is toggled at each iteration.
    * The DMA cores are one index step ahead so they load the data in advance into the buffer that will be used.
    */

    volatile uint32_t p[3] = {0, 0, 0};
    volatile uint32_t P[3] = {0, 0, 0};
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    // Setup layout for TCDM L1
    // For double buffering l1 is a size 2 array
    TcdmLayout* l1 = (TcdmLayout*) snrt_l1_next();
    TcdmLayout* l1Addr[SNRT_CLUSTER_NUM] = {0};

    // Sync l1 pointers between clusters
    if (snrt_is_dm_core())
        l1AddrGlobal[p[1]] = l1;
    snrt_global_barrier();
    if (snrt_is_dm_core()) {
        memcpy(l1Addr, l1AddrGlobal, SNRT_CLUSTER_NUM * sizeof(*l1Addr));
    }

    bool l1Id_AB = false;
    bool l1Id_C  = false;

    // Initialize indices
    const uint32_t I = m, J = n, K = k;

    const uint32_t PI = 2, PJ = 2;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev, jb_prev, kb_prev;
    bool ib_dir = false, jb_dir = false, kb_dir = false;

    bool storeC = false;

    // Debug
    volatile int ib_cnt = 0, jb_cnt = 0, kb_cnt = 0;

    if (snrt_is_compute_core()) {
        snrt_global_barrier(); // DMA core is one index ahead
    }

    // Compute C2C sources for 2D pipeline
    volatile const uint32_t pk = (PI + 2 * PJ - pi - pj - 1) % PJ; // pipeline step
    int PK                     = PJ; // pipeline depth

    // Determine C2C source cluster index for each matrix, < 0 is from DRAM
    TcdmLayout* c2cL1_A = NULL;
    TcdmLayout* c2cL1_B = NULL;
    if (snrt_is_dm_core()) {
        dump_pk(pk);

        const bool fetch_dram = pk == 0;

        volatile const uint32_t p_srcA = pi * PJ + ((2 * PJ - pi - pk) % PJ);
        volatile const uint32_t p_srcB = pj + PJ * ((2 * PJ - pj - pk) % PJ);
        dump_p_src(fetch_dram ? -1 : p_srcA);
        dump_p_src(fetch_dram ? -1 : p_srcB);

        c2cL1_A = fetch_dram ? NULL : l1Addr[p_srcA];
        c2cL1_B = fetch_dram ? NULL : l1Addr[p_srcB];
    }

    // Wait for pipeline to be filled
    for (int pipeline = pk; pipeline > 0; --pipeline) {
        snrt_global_barrier();
    }

    FOR_EACH(ib, pi, I / L1_M, PI, ib_dir, ib_prev) {
        ib_cnt += ib;
        FOR_EACH(jb, pj, J / L1_N, PJ, jb_dir, jb_prev) {
            jb_cnt += jb;

            double* const l1_C = l1[l1Id_C].C;

            if (snrt_is_dm_core()) {
                dump_ib(ib);
                dump_jb(jb);
                snrt_dma_load_2d_tile(l1_C, C, ib, jb, L1_M, L1_N, ldc, FP64);
                if (ib != ib_first || jb != jb_first)
                    storeC = true;
            }

            FOR_EACH(kb, 0, K / L1_K, 1, kb_dir, kb_prev) {
                kb_cnt += kb;
                double* const l1_A = l1[l1Id_AB].A;
                double* const l1_B = l1[l1Id_AB].B;

                // load next A, B
                if (snrt_is_dm_core()) {
                    if (c2cL1_A == NULL)
                        snrt_dma_load_2d_tile(l1_A, A, ib, kb, L1_M, L1_K, lda, FP64);
                    else {
                        double* const c2c_A = c2cL1_A[l1Id_AB].A;
                        snrt_dma_start_1d(l1_A, c2c_A, L1_M * L1_K * FP64);
                    }
                    if (c2cL1_B == NULL)
                        snrt_dma_load_2d_tile(l1_B, B, kb, jb, L1_K, L1_N, ldb, FP64);
                    else {
                        double* const c2c_B = c2cL1_B[l1Id_AB].B;
                        snrt_dma_start_1d(l1_B, c2c_B, L1_K * L1_N * FP64);
                    }

                    snrt_dma_wait_all();
                } else {
                    // solve block already in l1, parallelize inside each cluster
                    gemm(FP64, 0, true, false, false,
                         L1_M, L1_N, L1_K, alpha,
                         l1_A, L1_LDA, l1_B, L1_LDB, beta, l1_C, L1_LDC);
                }

                l1Id_AB = !l1Id_AB; // switch buffers
                snrt_global_barrier();

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
        snrt_global_barrier(); // DMA core is one index ahead

        // store final tile
        snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N, ldc, FP64);
        snrt_dma_wait_all();
    }

    // Wait for pipeline to be emptied
    for (int pipeline = pk; pipeline < PK; ++pipeline) {
        snrt_global_barrier();
    }
}

inline void gemm_oc(precision_t prec, uint32_t expand, uint32_t setup_ssr,
                    uint32_t transa, uint32_t transb, uint32_t m, uint32_t n,
                    uint32_t k, double alpha, void* a, uint32_t lda, void* b,
                    uint32_t ldb, uint32_t beta, void* c, uint32_t ldc) {
    // gemm_cluster_kernel(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
    // snrt_fpu_fence();
    // snrt_cluster_hw_barrier();

    gemm_oc_opt2d(alpha, beta, m, n, k, a, b, c, lda, ldb, ldc);
}
