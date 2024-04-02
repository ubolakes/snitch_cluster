// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "snrt.h"

#define BIST
#include "data.h"

#include "gemm.h"
#include "dma_xfer_test.h"
#include "dma_tile2tile_test.h"

#include "dump.h"
NAMED_DUMP(uint32_t, err, 0x7)
NAMED_DUMP(uint32_t, bench_iter, 0x7)


int main() {
    const bool setup_ssr = true;

    // load into TCDM
    uint32_t iters = bench_iters;

    SnblasGemmInfo gemmInfo = {0};
    gemmInfo.M   = M;
    gemmInfo.N   = N;
    gemmInfo.K   = K;
    gemmInfo.ta  = TA;
    gemmInfo.tb  = TB;
    gemmInfo.tc  = TC;
    gemmInfo.lda = gemmInfo.ta ? gemmInfo.M : gemmInfo.K;
    gemmInfo.ldb = gemmInfo.tb ? gemmInfo.K : gemmInfo.N;
    gemmInfo.ldc = gemmInfo.tc ? gemmInfo.M : gemmInfo.N;
    
    SNBLAS_GEMM_ARGS(DTYPE) gemmArgs = {0};
    gemmArgs.A     = a;
    gemmArgs.B     = b;
    gemmArgs.C     = c;
    gemmArgs.alpha = 1;
    gemmArgs.beta  = BETA;

    SnblasGemmImpl gemmImpl = {0};
    gemmImpl.ta_tile = TA_TILE;
    gemmImpl.tb_tile = TB_TILE;
    gemmImpl.tc_tile = TC_TILE;

    for (volatile int i = iters; i > 0; --i) {
        // if (i == 1) snrt_mcycle(); // start
        gemmImpl.bench = i == 1;
        SNBLAS_GEMM(USE_METHOD, DTYPE)(gemmInfo, gemmArgs, gemmImpl);
        // dma_tile2tile_test(gemmInfo, gemmArgs, gemmImpl);
        // dma_xfer_test(c, M*N, i == 1);

        if (snrt_global_core_idx() == 0)
            dump_bench_iter(-i);
        snrt_fpu_fence();
        snrt_global_barrier();
        if (i == 1) snrt_mcycle(); // end
    }

#ifdef BIST_COMPUTE
    uint32_t errors = M * N;

    if (snrt_global_core_idx() == 0) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                uint32_t idx = i * N + j;
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