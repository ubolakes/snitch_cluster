// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Jose Pedro Castro Fonseca <jcastro@ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>
#include "args.h"
#include "snrt.h"

#include "gemv/src/gemv.h"

// Assumes the A and B buffers are at the same offset in the TCDM of every
// cluster. Also assumes the last element in the two buffers can be used as a
// status flag for synchronization. The status flag encodes whether the data
// in the buffer is valid (0/1) and what iteration (level) it belongs to as:
// status = level * 2 + valid.
// `len` is the number of elements in the buffer, without counting the
// notification flag. At the beginning, the source data is expected to be
// in the A buffer, which is also where the output will be.
static inline void global_reduction(volatile double *a_buffer, volatile double *b_buffer,
                             size_t len) {
    // If we have a single cluster there is no reduction to perform
    if (snrt_cluster_num() > 1) {
        // Iterate levels in the binary reduction tree
        int num_levels = ceil(log2(snrt_cluster_num()));
        for (unsigned int level = 0; level < num_levels; level++) {
            // Determine whether the current cluster is an active cluster.
            // An active cluster is a cluster that participates in the current
            // level of the reduction tree. Every second cluster among the
            // active ones is a sender.
            uint32_t is_active = (snrt_cluster_idx() % (1 << level)) == 0;
            uint32_t is_sender = (snrt_cluster_idx() % (1 << (level + 1))) != 0;

            // If the cluster is a sender, it must wait for the receiver to be
            // done processing the data in the B buffer from the
            // previous level. It then updates the status flag in the A buffer
            // to mark the data arriving in the destination's B buffer as
            // valid.
            // If the cluster is a receiver, it polls the status flag
            // to check if the data is valid, i.e. if it fully arrived.
            if (snrt_is_dm_core()) snrt_mcycle();
            if (is_active && snrt_is_dm_core()) {
                if (is_sender) {
                    volatile double *b_buffer_dst = (volatile double *)
                        ((void *)b_buffer - (1 << level) * SNRT_CLUSTER_OFFSET);
                    while (b_buffer_dst[len] != (level * 2)) ;
                    a_buffer[len] = level * 2 + 1;
                    snrt_dma_start_1d(
                        (void *)b_buffer_dst,
                        (void *)a_buffer,
                        (len + 1) * sizeof(double)
                    );
                    snrt_dma_wait_all();
                } else {
                    while (b_buffer[len] == (level * 2)) ;
                }
            }
            if (snrt_is_dm_core()) snrt_mcycle();

            // Synchronize DM and compute cores
            snrt_cluster_hw_barrier();

            // Every cluster which is not a sender performs the reduction
            if (snrt_is_compute_core()) snrt_mcycle();
            if (is_active && !is_sender) {
                // Computation is parallelized over the compute cores (strided)
                if (snrt_is_compute_core()) {
                    uint32_t items_per_core =
                        len / snrt_cluster_compute_core_num();
                    uint32_t remainder_items =
                        len % snrt_cluster_compute_core_num();
                    uint32_t core_offset = snrt_cluster_core_idx();
                    if (snrt_cluster_core_idx() < remainder_items)
                        items_per_core++;
                    for (uint32_t i = 0; i < items_per_core; i++) {
                        uint32_t abs_i = core_offset + i * snrt_cluster_compute_core_num();
                        a_buffer[abs_i] += b_buffer[abs_i];
                    }
                    // Core 0 updates the status flag, to indicate that the
                    // buffer's contents can be overriden with the data for
                    // the next iteration.
                    if (snrt_cluster_core_idx() == 0) b_buffer[len] += 1;
                    snrt_fpu_fence();
                }
            }
            if (snrt_is_compute_core()) snrt_mcycle();

            // Synchronize compute and DM cores for next tree level
            snrt_cluster_hw_barrier();
        }
    }
}

// A^t*A product given a matrix A of size MxN
static inline void ata_fp64_opt(uint32_t m, uint32_t n, double* A, double *C) {
    // Derive GEMM parameters
    uint32_t M = n;
    uint32_t N = n;
    uint32_t K = m;
    uint32_t ldA = N;
    uint32_t ldC = N;

    // Configure SSRs to stream a
    const uint32_t ssr0_b[3] = {K, N, M};
    const uint32_t ssr0_i[3] = {8 * ldA, 0, 8};
    snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[0], ssr0_b[1], ssr0_b[2],
                     ssr0_i[0], ssr0_i[1], ssr0_i[2]);
    const uint32_t ssr1_b[3] = {K, N, M};
    const uint32_t ssr1_i[3] = {8 * ldA, 8, 0};
    snrt_ssr_loop_3d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                     ssr1_i[0], ssr1_i[1], ssr1_i[2]);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_3D, A);
    snrt_ssr_enable();

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            double c = 0.0;
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmadd.d %[c], ft0, ft1, %[c] \n"
                : [ c ] "+f"(c)
                : [ n_frep ] "r"(K - 1)
                : "ft0", "ft1", "ft2", "memory");

            // Store results back
            C[m * ldC + n] = c;
        }
    }
    snrt_ssr_disable();
    snrt_fpu_fence();
}

// Subtract a vector x [1, n] from a matrix A [m, n].
// The x vector is broadcasted across all rows of A.
static inline void apx(uint32_t m, uint32_t n, double* a, double *x) {

    // Configure SSR 0 to read a
    snrt_ssr_loop_2d(SNRT_SSR_DM0, m, n, n * 8, 8);

    // Configure SSR 1 to read x
    snrt_ssr_loop_2d(SNRT_SSR_DM1, m, n, 0, 8);

    // Configure SSR 2 to write a
    snrt_ssr_loop_2d(SNRT_SSR_DM2, m, n, n * 8, 8);

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, a);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, x);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_2D, a);
    snrt_ssr_enable();

    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.d ft2, ft0, ft1 \n"
        :
        : [ n_frep ] "r"(n * m - 1)
        : "ft0", "ft1", "ft2", "memory");

    snrt_ssr_disable();
    snrt_fpu_fence();
}

// Single-cluster computation of the first step in the covariance kernel
static inline __attribute__((always_inline)) void covariance_step1(uint32_t N, uint32_t M, double *data, double *mean) {

    // Compute mean vector
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        double one = 1;
        double alpha = -1.0 / N;
        gemv(1, M, N, alpha, data, &one, 0, mean);
        snrt_mcycle();
    }

    snrt_cluster_hw_barrier();

    // Normalize the data
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        if (snrt_cluster_core_idx() == 0) {
            apx(N, M, data, mean);
        }
        snrt_mcycle();
    }
}

// Single-cluster computation of the second step in the covariance kernel
static inline __attribute__((always_inline)) void covariance_step2(uint32_t N, uint32_t M, double *data,
                                    double *cov_a, double *cov_b) {
    // Every cluster computes the matrix product between a
    // [M, N_frac] tile of data^T and data.
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        if (snrt_cluster_core_idx() == 0) {
            uint32_t N_frac = N / snrt_cluster_num();
            uint32_t N_offset = N_frac * snrt_cluster_idx();
            ata_fp64_opt(N_frac, M, &data[N_offset * M], cov_a);
        }
        snrt_mcycle();
    }
    snrt_cluster_hw_barrier();

    // Sum the partial results from the various clusters
    global_reduction((double*)cov_a, (double*)cov_b, M * M);

    // Normalize results
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        if (snrt_cluster_core_idx() == 0 && snrt_cluster_idx() == 0) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++)
                    cov_a[i * M + j] /= N - 1;
            snrt_fpu_fence();
        }
        snrt_mcycle();
    }
}

void covariance_job(void *args) {
    double *local_data, *local_mean, *local_cov_a, *local_cov_b;
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
    size_t size_mean = M * sizeof(double);
    size_t size_cov = M * M * sizeof(double);
    local_data = snrt_l1_alloc_cluster_local(size_data, sizeof(double));
    local_mean = snrt_l1_alloc_cluster_local(size_mean, sizeof(double));
    // Add one more element to store the notification flag in the buffers
    // which will be used in the global reduction, and initialize it.
    local_cov_a = snrt_l1_alloc_cluster_local(size_cov + sizeof(double), sizeof(double));
    local_cov_b = snrt_l1_alloc_cluster_local(size_cov + sizeof(double), sizeof(double));
    local_cov_b[M * M] = 0;

    // Load input matrix tile
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_data, data, size_data);
        snrt_dma_wait_all();
    }
    snrt_mcycle();
    snrt_cluster_hw_barrier();

    // Perform step 1 of the covariance
    covariance_step1(N, M, local_data, local_mean);
    snrt_cluster_hw_barrier();

    // Perform step 2 of the covariance
    covariance_step2(N, M, local_data, local_cov_a, local_cov_b);
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Copy data out of TCDM
    if (snrt_is_dm_core()) {
        if (snrt_cluster_idx() == 0) {
            snrt_dma_start_1d(cov, local_cov_a, size_cov);
            snrt_dma_wait_all();
            snrt_mcycle();
        }
    }

    // Free memory
#ifndef JOB_ARGS_PRELOADED
    snrt_l1_update_next_v2(local_args);
#else
    snrt_l1_update_next_v2(local_data);
#endif
}
