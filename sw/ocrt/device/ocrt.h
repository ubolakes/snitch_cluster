#pragma once

#include <snrt.h>

#include <stddef.h>
#include <stdint.h>

// Occamy specific definitions
#include "occamy_defs.h"
#include "occamy_memory_map.h"
#include "sys_dma.h"

// Forward declarations
#include "alloc_decls.h"
#include "cls_decls.h"
#include "cluster_interrupt_decls.h"
#include "global_interrupt_decls.h"
#include "memory_decls.h"
#include "sync_decls.h"
#include "team_decls.h"

// Implementation
#include "alloc.h"
#include "cls.h"
#include "cluster_interrupts.h"
#include "dma.h"
#include "global_interrupts.h"
#include "occamy_device.h"
#include "occamy_memory.h"
#include "printf.h"
#include "riscv.h"
#include "ssr.h"
#include "sync.h"
#include "team.h"
