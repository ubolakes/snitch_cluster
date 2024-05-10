// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Jose Pedro Castro Fonseca <jcastro@ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>
#include "args.h"
#include "snrt.h"

static inline void gemv(uint32_t M, uint32_t N, uint32_t K,
                        double *A, double *B, double *C) {
    // Configure SSR 0 to stream A
    const uint32_t ssr0_b[2] = {K, M};
    const uint32_t ssr0_i[2] = {8, K*8};
    snrt_ssr_loop_2d(SNRT_SSR_DM0, ssr0_b[0], ssr0_b[1], ssr0_i[0], ssr0_i[1]);

    // Configure SSR 1 to stream B
    const uint32_t ssr1_b[2] = {K, M};
    const uint32_t ssr1_i[2] = {8, 0};
    snrt_ssr_loop_2d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_i[0], ssr1_i[1]);

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, B);
    snrt_ssr_enable();

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            double c = 0.0;

            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmadd.d %[c], ft0, ft1, %[c] \n"
                : [ c ] "+f"(c)
                : [ n_frep ] "r"(K - 1)
                : "ft0", "ft1", "ft2");

            C[m] = c;
        }
    }
    snrt_ssr_disable();
    snrt_fpu_fence();
}

static inline void atax(uint32_t M, uint32_t N, double *A, double *x,
                        double *y, double *tmp) {
    double tmp_fs;
    int core_range, core_offset, cluster_core_offset;

    // tmp = A * x
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        // Distribute rows to cores in cluster
        uint32_t frac_m = M / snrt_cluster_compute_core_num();
        uint32_t rem_m = M % snrt_cluster_compute_core_num();
        uint32_t start_m = snrt_cluster_core_idx() * frac_m;
        uint32_t core_m =
            snrt_cluster_core_idx() == (snrt_cluster_compute_core_num() - 1) ?
            frac_m + rem_m : frac_m;
        gemv(core_m, 1, N, &A[start_m * N], x, &tmp[start_m]);
        snrt_mcycle();
    }

    snrt_cluster_hw_barrier();

    // y = At * tmp
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        core_range = N / snrt_global_compute_core_num();
        core_offset = snrt_global_compute_core_idx() * core_range;
        cluster_core_offset = snrt_cluster_core_idx() * core_range;
        for (int j1 = 0; j1 < core_range; j1++) {
            int j = core_offset + j1;
            int cluster_j = cluster_core_offset + j1;
            tmp_fs = 0.0;
            for (int i = 0; i < M; i++) {
                // The order of the for loops was exchanged, so that each loop
                // reduces in y at position j, iterating through the i
                // positions.
                tmp_fs += A[i * N + j] * tmp[i];
            }
            y[cluster_j] = tmp_fs;
        }
        snrt_fpu_fence();
        snrt_mcycle();
    }
}

void atax_job(void *args) {
    double *local_A;
    double *local_x;
    double *local_y;
    double *local_tmp;
    atax_args_t *local_args;

#ifndef JOB_ARGS_PRELOADED
    // Allocate space for job arguments in TCDM
    local_args = (atax_args_t *)snrt_l1_alloc_cluster_local(sizeof(atax_args_t), sizeof(double));

    // Copy job arguments to TCDM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_args, args, sizeof(atax_args_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
#else
    local_args = (atax_args_t *)args;
#endif

    // Aliases
    uint32_t M = local_args->M;
    uint32_t N = local_args->N;
    double *A = (double *)(local_args->A_addr);
    double *x = (double *)(local_args->x_addr);
    double *y = (double *)(local_args->y_addr);

    // Allocate local variables
    size_t size_A = M * N * sizeof(double);
    size_t size_x = N * sizeof(double);
    size_t size_y = N * sizeof(double);
    size_t size_tmp = M * sizeof(double);
    size_t size_y_tile = size_y / snrt_cluster_num();
    local_A = snrt_l1_alloc_cluster_local(size_A, sizeof(double));
    local_x = snrt_l1_alloc_cluster_local(size_x, sizeof(double));
    local_y = snrt_l1_alloc_cluster_local(size_y_tile, sizeof(double));
    local_tmp = snrt_l1_alloc_cluster_local(size_tmp, sizeof(double));

    // Initialize input matrices
    if (snrt_is_dm_core()) {
        void *zero_mem = (void *)snrt_zero_memory_ptr();
        snrt_dma_start_1d(local_A, A, size_A);
        snrt_dma_start_1d(local_x, x, size_x);
        snrt_dma_wait_all();
    }
    snrt_mcycle();
    snrt_cluster_hw_barrier();

    // Compute
    atax(M, N, local_A, local_x, local_y, local_tmp);
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Writeback results
    if (snrt_is_dm_core()) {
        snrt_dma_store_1d_tile(y, local_y, snrt_cluster_idx(), N / snrt_cluster_num(), sizeof(double));
        snrt_dma_wait_all();
        snrt_mcycle();
    }
    snrt_cluster_hw_barrier();

    // Free memory
#ifndef JOB_ARGS_PRELOADED
    snrt_l1_update_next_v2(local_args);
#else
    snrt_l1_update_next_v2(local_A);
#endif
}
