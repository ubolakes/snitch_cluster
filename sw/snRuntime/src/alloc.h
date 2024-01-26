// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define ALIGN_UP(addr, size) (((addr) + (size)-1) & ~((size)-1))
#define ALIGN_DOWN(addr, size) ((addr) & ~((size)-1))

#define MIN_CHUNK_SIZE 8

extern snrt_allocator_t l3_allocator;
extern __thread snrt_allocator_t l1_allocator;

inline snrt_allocator_t *snrt_l1_allocator() { return &l1_allocator; }
inline snrt_allocator_t *snrt_l3_allocator() { return &l3_allocator; }

inline void *snrt_l1_next() { return (void *)snrt_l1_allocator()->next; }
inline void *snrt_l3_next() { return (void *)snrt_l3_allocator()->next; }

/**
 * @brief Allocate a chunk of memory in the L3 memory
 * @details This currently does not support free-ing of memory
 *
 * @param size number of bytes to allocate
 * @return pointer to the allocated memory
 */
inline void *snrt_l3_alloc(size_t size) {
    snrt_allocator_t *alloc = snrt_l3_allocator();

    // TODO: L3 alloc size check

    void *ret = (void *)alloc->next;
    alloc->next += size;
    return ret;
}

inline void snrt_alloc_init() {
    // Note: at the moment the allocator assumes all of the TCDM is
    // available for allocation. However, the CLS, TLS and stack already
    // occupy a possibly significant portion.
    snrt_l1_allocator()->base = snrt_l1_start_addr();
    snrt_l1_allocator()->size = snrt_l1_end_addr() - snrt_l1_start_addr();
    snrt_l1_allocator()->next = snrt_l1_allocator()->base;

    // Only one core in the system has to initialize the L3 allocator
    if (snrt_is_dm_core()) {
        extern uint32_t _edram;
        snrt_l3_allocator()->base = ALIGN_UP((uint32_t)&_edram, MIN_CHUNK_SIZE);
        snrt_l3_allocator()->size = 0;
        snrt_l3_allocator()->next = snrt_l3_allocator()->base;
    }
}

// Dynamically allocate space for a variable of size `size` in the cluster's L1
// memory. This function should be invoked by every core in a cluster. Every
// core receives a pointer to the allocated variable.
inline void *snrt_l1_alloc_cluster_local(size_t size, const size_t alignment) {
    snrt_l1_allocator()->next = ALIGN_UP(snrt_l1_allocator()->next, alignment);
    void *retval = snrt_l1_next();
    snrt_l1_allocator()->next += size;
    return retval;
}

// Dynamically allocate space for N variables of size `size` in the cluster's
// L1 memory, N being the number of compute cores in the cluster. This function
// should be invoked by every core in a cluster. Every compute core receives a
// pointer to a unique variable among the N which have been allocated. The
// return value for the DM core is undefined.
inline void *snrt_l1_alloc_compute_core_local(size_t size,
                                              const size_t alignment) {
    snrt_l1_allocator()->next = ALIGN_UP(snrt_l1_allocator()->next, alignment);
    void *retval = snrt_l1_next() + size * snrt_cluster_core_idx();
    snrt_l1_allocator()->next += size * snrt_cluster_compute_core_num();
    return retval;
}

// Takes a pointer to a variable in the source cluster's L1 memory and returns
// a pointer to the same offset in the destination cluster's L1 memory.
inline void *snrt_remote_l1_ptr(void *ptr, uint32_t src_cluster_idx,
                                uint32_t dst_cluster_idx) {
    return (void *)((uintptr_t)ptr +
                    (dst_cluster_idx - src_cluster_idx) * SNRT_CLUSTER_OFFSET);
}

// TODO colluca: optimize by using DMA
inline void *snrt_memset(void *ptr, int value, size_t num) {
    for (uint32_t i = 0; i < num; ++i)
        *((uint8_t *)ptr + i) = (unsigned char)value;
    return ptr;
}
