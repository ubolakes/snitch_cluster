// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Jose Pedro Castro Fonseca <jcastro@ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>
#include "args.h"
#include "snrt.h"

// Single-cluster computation of the first step in the covariance kernel
static inline void covariance_step1(uint32_t N, uint32_t M, double *data) {
    int i1, i, j, k;
    int core_range, core_offset;

    // Compute deviations
    if (snrt_is_compute_core()) {

        snrt_mcycle();

        // Distribute different attributes to the different cores
        core_range = M / snrt_cluster_compute_core_num();
        core_offset = snrt_cluster_core_idx() * core_range;

        for (i1 = 0; i1 < core_range; i1++) {
            i = core_offset + i1;

            // Calculate mean vector
            double mean = 0.0;
            for (k = 0; k < N; k++) {
                mean += data[k * M + i];
            }
            mean = mean / N;

            // Standardize data to zero mean
            for (k = 0; k < N; k++) {
                data[k * M + i] -= mean;
            }
        }
        snrt_fpu_fence();

        snrt_mcycle();
    }
}

// Single-cluster computation of the second step in the covariance kernel
static inline void covariance_step2(uint32_t N, uint32_t M, double *data,
                                     double *cov) {
    int i1, i, j, k;
    int core_range, core_offset;

    // Compute covariance
    if (snrt_is_compute_core()) {

        snrt_mcycle();

        // Distribute different attributes to the different cores
        core_range = M / snrt_cluster_compute_core_num();
        core_offset = snrt_cluster_core_idx() * core_range;

        for (i1 = 0; i1 < core_range; i1++) {
            i = core_offset + i1;
            for (j = 0; j <= i; j++) {
                double tmp = 0.0;
                for (k = 0; k < N; k++) {
                    tmp += data[k * M + i] * data[k * M + j];
                }
                cov[i * M + j] = tmp / (N - 1);
                cov[j * M + i] = cov[i * M + j];
            }
        }
        snrt_fpu_fence();

        snrt_mcycle();
    }
}

void covariance_job(void *args) {
    double *local_data;
    double *local_cov;
    covariance_args_t *local_args;

#ifndef JOB_ARGS_PRELOADED
    // Allocate space for job arguments in TCDM
    local_args = (covariance_args_t *)snrt_l1_alloc_cluster_local(
        sizeof(covariance_args_t), sizeof(double));

    // Copy job arguments to TCDM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_args, args, sizeof(covariance_args_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
#else
    local_args = (covariance_args_t *)args;
#endif

    // Aliases
    uint32_t M = local_args->M;
    uint32_t N = local_args->N;
    double *data = (double *)(local_args->data_addr);
    double *cov = (double *)(local_args->cov_addr);

    // Allocate local variables
    size_t size_data = N * M * sizeof(double);
    size_t size_cov = M * M * sizeof(double);
    local_data = snrt_l1_alloc_cluster_local(size_data, sizeof(double));
    local_cov = snrt_l1_alloc_cluster_local(size_cov, sizeof(double));

    // Parallelize step 1 across clusters, distributing the M columns
    size_t tile_M = M / snrt_cluster_num();

    // Load input matrix tile
    if (snrt_is_dm_core()) {
        snrt_dma_load_2d_tile(
            local_data,          // dst
            data,                // src
            0,                   // tile_x1_idx
            snrt_cluster_idx(),  // tile_x0_idx
            N,                   // tile_x1_size
            tile_M,              // tile_x0_size
            M,                   // full_x0_size
            sizeof(double)       // prec
        );
        snrt_dma_wait_all();
    }
    snrt_mcycle();
    snrt_cluster_hw_barrier();

    // Perform step 1 of the covariance
    covariance_step1(N, tile_M, local_data);
    snrt_global_barrier();

    // The rest of the computation is done only on cluster 0
    if (snrt_cluster_idx() == 0) {

        // Aggregate data in cluster 0
        if (snrt_is_dm_core() ) {

            snrt_mcycle();

            // Theoretically speaking, moving the data in cluster 0's TCDM
            // is not required. However we need to reshape it because
            // `covariance_step1` is currently implemented in a way such
            // that it stores the output tile as contiguous data, not with
            // the proper stride it would have in the full matrix.
            for (unsigned int i = 0; i < snrt_cluster_num(); i++) {
                double *remote_data = snrt_remote_l1_ptr(local_data, snrt_cluster_idx(), i);
                snrt_dma_store_2d_tile(
                    local_data,     // dst
                    remote_data,    // src
                    0,              // tile_x1_idx
                    i,              // tile_x0_idx
                    N,              // tile_x1_size
                    tile_M,         // tile_x0_size
                    M,              // full_x0_size
                    sizeof(double)  // prec
                );
            }
            snrt_dma_wait_all();

            snrt_mcycle();
        }
        snrt_cluster_hw_barrier();

        // Perform step 2 of the covariance
        covariance_step2(N, M, local_data, local_cov);
        snrt_cluster_hw_barrier();
        snrt_mcycle();

        // Cluster 0 writes back output matrix
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(cov, local_cov, size_cov);
            snrt_dma_wait_all();
            snrt_mcycle();
        }
        snrt_cluster_hw_barrier();
    } else {
        snrt_mcycle();
    }

    // Free memory
#ifndef JOB_ARGS_PRELOADED
    snrt_l1_update_next_v2(local_args);
#else
    snrt_l1_update_next_v2(local_data);
#endif
}
