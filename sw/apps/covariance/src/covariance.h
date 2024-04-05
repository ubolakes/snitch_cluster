// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Jose Pedro Castro Fonseca <jcastro@ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>
#include "args.h"
#include "snrt.h"

void covariance(uint32_t N, uint32_t M, double *data, double *cov) {
    int i1, i, j, k;
    int core_range, core_offset;

    // Compute deviations
    if (snrt_is_compute_core()) {
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
    }

    snrt_cluster_hw_barrier();

    // Compute covariance
    if (snrt_is_compute_core()) {
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

    // Initialize input matrix
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_data, data, size_data);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Perform Computations
    covariance(N, M, local_data, local_cov);
    snrt_cluster_hw_barrier();

    // Writeback outputs
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(cov, local_cov, size_cov);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
}
