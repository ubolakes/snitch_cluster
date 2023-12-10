// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "sync_decls.h"
#include "heterogeneous_runtime_decls.h"

typedef enum { SYNC_ALL, SYNC_CLUSTERS, SYNC_NONE } sync_t;

inline uint32_t __attribute__((const)) snrt_quadrant_idx();

inline void post_wakeup_cl();

inline comm_buffer_t* __attribute__((const)) get_communication_buffer();

inline uint32_t elect_director(uint32_t num_participants);

inline void return_to_cva6(sync_t sync);
