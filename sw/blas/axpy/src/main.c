// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

#define XSSR
#include "axpy.h"
#include "data.h"


int main() {

    axpy_args_t args = {n, a, (uint64_t)x, (uint64_t)y, (uint64_t)z};
    axpy_job(&args);

// TODO: currently only works for single cluster otherwise need to
//       synchronize all cores here
#ifdef BIST
    uint32_t n = args.n;
    double* z = args.z_addr;
    uint32_t nerr = n;

    // Check computation is correct
    if (snrt_global_core_idx() == 0) {
        for (int i = 0; i < n; i++) {
            if (z[i] == g[i]) nerr--;
            printf("%d %d\n", z[i], g[i]);
        }
    }

    return nerr;
#endif

    return 0;
}
