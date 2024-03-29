// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//================================================================================
// Mutex functions
//================================================================================

inline volatile uint32_t *snrt_mutex() { return &_snrt_mutex; }

/**
 * @brief lock a mutex, blocking
 * @details declare mutex with `static volatile uint32_t mtx = 0;`
 */
inline void snrt_mutex_acquire(volatile uint32_t *pmtx) {
    asm volatile(
        "li            t0,1          # t0 = 1\n"
        "1:\n"
        "  amoswap.w.aq  t0,t0,(%0)   # t0 = oldlock & lock = 1\n"
        "  bnez          t0,1b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "t0");
}

/**
 * @brief lock a mutex, blocking
 * @details test and test-and-set (ttas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
inline void snrt_mutex_ttas_acquire(volatile uint32_t *pmtx) {
    asm volatile(
        "1:\n"
        "  lw t0, 0(%0)\n"
        "  bnez t0, 1b\n"
        "  li t0,1          # t0 = 1\n"
        "2:\n"
        "  amoswap.w.aq  t0,t0,(%0)   # t0 = oldlock & lock = 1\n"
        "  bnez          t0,2b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "t0");
}

/**
 * @brief Release the mutex
 */
inline void snrt_mutex_release(volatile uint32_t *pmtx) {
    asm volatile("amoswap.w.rl  x0,x0,(%0)   # Release lock by storing 0\n"
                 : "+r"(pmtx));
}

//================================================================================
// Barrier functions
//================================================================================

/// Synchronize cores in a cluster with a hardware barrier
inline __attribute__((always_inline)) void snrt_cluster_hw_barrier() {
    asm volatile("csrr x0, 0x7C2" ::: "memory");
}

/// Synchronize all DM cores with a software barrier
inline void snrt_global_dm_core_barrier() {
    // Synchronize all DM cores in software
    if (snrt_is_dm_core()) {
        // Get barrier in cluster 0 tcdm
        snrt_barrier_t* tcdm_barrier = (snrt_barrier_t*)
                ((uint32_t) (&_snrt_barrier) - cluster_offset * snrt_cluster_idx());

        // Remember previous iteration
        uint32_t prev_barrier_iteration = tcdm_barrier->iteration;
        __atomic_add_fetch(&(tcdm_barrier->cnt), 1, __ATOMIC_RELAXED);

        // Increment the barrier counter
        if (snrt_cluster_idx() == 0) {
            while(tcdm_barrier->cnt != snrt_cluster_num());
            tcdm_barrier->cnt = 0;
            __atomic_add_fetch(&(tcdm_barrier->iteration), 1, __ATOMIC_RELAXED);
        } else {
            while (prev_barrier_iteration == tcdm_barrier->iteration);
        }
    }
}

/// Synchronize clusters globally with a global software barrier
inline __attribute__((always_inline)) void snrt_global_barrier() {
    snrt_cluster_hw_barrier();       // DM core waits for cluster compute cores to enter
    snrt_global_dm_core_barrier();   // DM core signals it's cluster is ready and waits for other clusters
    snrt_cluster_hw_barrier();       // Compute cores wait for DM to exit the barrier
}

inline uint32_t snrt_global_all_to_all_reduction(uint32_t value) {
    __atomic_add_fetch(&_reduction_result, value, __ATOMIC_RELAXED);
    snrt_global_barrier();
    return _reduction_result;
}

/**
 * @brief Generic barrier
 *
 * @param barr pointer to a barrier
 * @param n number of harts that have to enter before released
 */
inline void snrt_partial_barrier(snrt_barrier_t *barr, uint32_t n) {
    // Remember previous iteration
    uint32_t prev_it = barr->iteration;
    uint32_t cnt = __atomic_add_fetch(&barr->cnt, 1, __ATOMIC_RELAXED);

    // Increment the barrier counter
    if (cnt == n) {
        barr->cnt = 0;
        __atomic_add_fetch(&barr->iteration, 1, __ATOMIC_RELAXED);
    } else {
        // Some threads have not reached the barrier --> Let's wait
        while (prev_it == barr->iteration)
            ;
    }
}