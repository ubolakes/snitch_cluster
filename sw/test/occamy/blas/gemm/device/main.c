// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "snrt.h"

#include "gemm.h"
#include "data.h"

#include "dump.h"
NAMED_DUMP(uint32_t, err, 0x7)

#define BIST
#include "data.h"

int main() {
    const bool setup_ssr = true;
    uint32_t start_cycle = snrt_mcycle();

    uint32_t lda = K;
    uint32_t ldb = N;
    uint32_t ldc = N;

    gemm_oc(dtype_size, expand, setup_ssr, TA, TB, M, N, K, ALPHA,
            a, lda, b, ldb, 1, c, ldc);

    uint32_t end_cycle = snrt_mcycle();

    snrt_fpu_fence();
    snrt_global_barrier();

#ifdef BIST_COMPUTE
    uint32_t errors = M * N;

    if (snrt_global_core_idx() == 0) {
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                uint32_t idx = m * N + n;
                if (fabs(result[idx] - c[idx]) < 0.001)
                    errors--;
            }
        }
        // printf("%d/%d Errors\n", errors, M * N);
        dump_err(errors);
    }

    return errors;
#endif

    return 0;
}
