// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <heterogeneous_runtime_decls.h>

#include <stdint.h>

#include "occamy.h"
#include "occamy_memory_map.h"

/**************/
/* Interrupts */
/**************/

static inline void set_host_sw_interrupt() { *clint_msip_ptr(0) = 1; }

static inline void clear_host_sw_interrupt_unsafe() { *clint_msip_ptr(0) = 0; }

static inline void wait_host_sw_interrupt_clear() {
    while (*clint_msip_ptr(0))
        ;
}

static inline void clear_host_sw_interrupt() {
    clear_host_sw_interrupt_unsafe();
    wait_host_sw_interrupt_clear();
}

/**************************/
/* Quadrant configuration */
/**************************/

// Configure RO cache address range
static inline void configure_read_only_cache_addr_rule(uint32_t quad_idx,
                                                uint32_t rule_idx,
                                                uint64_t start_addr,
                                                uint64_t end_addr) {
    volatile uint64_t* rule_ptr =
        quad_cfg_ro_cache_addr_rule_ptr(quad_idx, rule_idx);
    *(rule_ptr) = start_addr;
    *(rule_ptr + 1) = end_addr;
}

// Enable RO cache
static inline void enable_read_only_cache(uint32_t quad_idx) {
    *(quad_cfg_ro_cache_enable_ptr(quad_idx)) = 1;
}
