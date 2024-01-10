// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "data.h"
#include "gemm.h"
#include "snrt.h"

int main() {
    const bool setup_ssr = true;
    uint32_t start_cycle = snrt_mcycle();

    volatile uint32_t lda = K;
    volatile uint32_t ldb = N;
    volatile uint32_t ldc = N;

    gemm_oc(dtype_size, expand, setup_ssr, TA, TB, M, N, K, ALPHA,
            a, lda, b, ldb, BETA, c,  ldc);

    uint32_t end_cycle = snrt_mcycle();

    snrt_cluster_hw_barrier();

    return 0;
}
