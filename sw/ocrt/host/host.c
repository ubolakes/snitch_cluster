// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "host.h"

void initialize_bss() {
    extern volatile uint64_t __bss_start, __bss_end;

    size_t bss_size = (size_t)(&__bss_end) - (size_t)(&__bss_start);
    if (bss_size)
        sys_dma_blk_memcpy((uint64_t)(&__bss_start), WIDE_ZERO_MEM_BASE_ADDR,
                           bss_size);
}

void enable_fpu() {
    uint64_t mstatus;

    asm volatile("csrr %[mstatus], mstatus" : [ mstatus ] "=r"(mstatus));
    mstatus |= (1 << MSTATUS_FS_OFFSET);
    asm volatile("csrw mstatus, %[mstatus]" : : [ mstatus ] "r"(mstatus));
}


