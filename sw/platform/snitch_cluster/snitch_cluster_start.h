#pragma once

#define SNRT_INIT_TLS
#define SNRT_INIT_BSS
#define SNRT_INIT_CLS
#define SNRT_INIT_LIBS
#define SNRT_CRT0_PRE_BARRIER
#define SNRT_INVOKE_MAIN
#define SNRT_CRT0_POST_BARRIER
#define SNRT_CRT0_EXIT

#include <stdint.h>

// #include "snitch_cluster_memory.h"

extern uintptr_t volatile tohost, fromhost;

static inline volatile uint32_t* snrt_exit_code_destination() {
    return (volatile uint32_t*)&tohost;
}