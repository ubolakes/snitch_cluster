// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>

typedef struct {
    uint32_t n;
    double a;
    uint64_t x_addr;
    uint64_t y_addr;
    uint64_t z_addr;
} axpy_args_t;
