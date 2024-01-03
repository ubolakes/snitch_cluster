// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <platform.h>
#include "team_decls.h"
#include "sync_decls.h"

/// A DMA transfer identifier.
typedef uint32_t snrt_dma_txid_t;

/// Initiate an asynchronous 1D DMA transfer with wide 64-bit pointers.
inline snrt_dma_txid_t snrt_dma_start_1d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size) {
    // Current DMA does not allow transfers with size == 0 (blocks)
    // TODO(colluca) remove this check once new DMA is integrated
    if (size > 0) {
        register uint32_t reg_dst_low asm("a0") = dst >> 0;    // 10
        register uint32_t reg_dst_high asm("a1") = dst >> 32;  // 11
        register uint32_t reg_src_low asm("a2") = src >> 0;    // 12
        register uint32_t reg_src_high asm("a3") = src >> 32;  // 13
        register uint32_t reg_size asm("a4") = size;           // 14

        // dmsrc a2, a3
        asm volatile(
            ".word (0b0000000 << 25) | \
                (     (13) << 20) | \
                (     (12) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
            "r"(reg_src_low));

        // dmdst a0, a1
        asm volatile(
            ".word (0b0000001 << 25) | \
                (     (11) << 20) | \
                (     (10) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
            "r"(reg_dst_low));

        // dmcpyi a0, a4, 0b00
        register uint32_t reg_txid asm("a0");  // 10
        asm volatile(
            ".word (0b0000010 << 25) | \
                (  0b00000 << 20) | \
                (     (14) << 15) | \
                (    0b000 << 12) | \
                (     (10) <<  7) | \
                (0b0101011 <<  0)   \n"
            : "=r"(reg_txid)
            : "r"(reg_size));

        return reg_txid;
    } else {
        return -1;
    }
}

/// Initiate an asynchronous 1D DMA transfer.
inline snrt_dma_txid_t snrt_dma_start_1d(void *dst, const void *src,
                                         size_t size) {
    return snrt_dma_start_1d_wideptr((size_t)dst, (size_t)src, size);
}

/// Initiate an asynchronous 2D DMA transfer with wide 64-bit pointers.
inline snrt_dma_txid_t snrt_dma_start_2d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size, size_t dst_stride,
                                                 size_t src_stride,
                                                 size_t repeat) {
    // Current DMA does not allow transfers with size == 0 (blocks)
    // TODO(colluca) remove this check once new DMA is integrated
    if (size > 0) {
        register uint32_t reg_dst_low asm("a0") = dst >> 0;       // 10
        register uint32_t reg_dst_high asm("a1") = dst >> 32;     // 11
        register uint32_t reg_src_low asm("a2") = src >> 0;       // 12
        register uint32_t reg_src_high asm("a3") = src >> 32;     // 13
        register uint32_t reg_size asm("a4") = size;              // 14
        register uint32_t reg_dst_stride asm("a5") = dst_stride;  // 15
        register uint32_t reg_src_stride asm("a6") = src_stride;  // 16
        register uint32_t reg_repeat asm("a7") = repeat;          // 17

        // dmsrc a0, a1
        asm volatile(
            ".word (0b0000000 << 25) | \
                (     (13) << 20) | \
                (     (12) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
            "r"(reg_src_low));

        // dmdst a0, a1
        asm volatile(
            ".word (0b0000001 << 25) | \
                (     (11) << 20) | \
                (     (10) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
            "r"(reg_dst_low));

        // dmstr a5, a6
        asm volatile(
            ".word (0b0000110 << 25) | \
                (     (15) << 20) | \
                (     (16) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n"
            :
            : "r"(reg_dst_stride), "r"(reg_src_stride));

        // dmrep a7
        asm volatile(
            ".word (0b0000111 << 25) | \
                (     (17) << 15) | \
                (    0b000 << 12) | \
                (0b0101011 <<  0)   \n"
            :
            : "r"(reg_repeat));

        // dmcpyi a0, a4, 0b10
        register uint32_t reg_txid asm("a0");  // 10
        asm volatile(
            ".word (0b0000010 << 25) | \
                (  0b00010 << 20) | \
                (     (14) << 15) | \
                (    0b000 << 12) | \
                (     (10) <<  7) | \
                (0b0101011 <<  0)   \n"
            : "=r"(reg_txid)
            : "r"(reg_size));

        return reg_txid;
    } else {
        return -1;
    }
}

/// Initiate an asynchronous 2D DMA transfer.
inline snrt_dma_txid_t snrt_dma_start_2d(void *dst, const void *src,
                                         size_t size, size_t dst_stride,
                                         size_t src_stride, size_t repeat) {
    return snrt_dma_start_2d_wideptr((size_t)dst, (size_t)src, size, dst_stride,
                                     src_stride, repeat);
}

/// Block until a transfer finishes.
inline void snrt_dma_wait(snrt_dma_txid_t tid) {
    // dmstati t0, 0  # 2=status.completed_id
    asm volatile(
        "1: \n"
        ".word (0b0000100 << 25) | \
               (  0b00000 << 20) | \
               (    0b000 << 12) | \
               (      (5) <<  7) | \
               (0b0101011 <<  0)   \n"
        "sub t0, t0, %0 \n"
        "blez t0, 1b \n" ::"r"(tid)
        : "t0");
}

/// Block until all operation on the DMA ceases.
inline void snrt_dma_wait_all() {
    // dmstati t0, 2  # 2=status.busy
    asm volatile(
        "1: \n"
        ".word (0b0000100 << 25) | \
               (  0b00010 << 20) | \
               (    0b000 << 12) | \
               (      (5) <<  7) | \
               (0b0101011 <<  0)   \n"
        "bne t0, zero, 1b \n" ::
            : "t0");
}

/**
 * @brief start tracking of dma performance region. Does not have any
 * implications on the HW. Only injects a marker in the DMA traces that can be
 * analyzed
 *
 */
inline void snrt_dma_start_tracking() { asm volatile("dmstati zero, 1"); }

/**
 * @brief stop tracking of dma performance region. Does not have any
 * implications on the HW. Only injects a marker in the DMA traces that can be
 * analyzed
 *
 */
inline void snrt_dma_stop_tracking() { asm volatile("dmstati zero, 3"); }

/**
 * @brief fast memset function performed by DMA
 *
 * @param ptr pointer to the start of the region
 * @param value value to set
 * @param len number of bytes, must be multiple of DMA bus-width
 */
inline void snrt_dma_memset(void *ptr, uint8_t value, uint32_t len) {
    // set first 64bytes to value
    // memset(ptr, value, 64);
    uint8_t *p = ptr;
    uint32_t nbytes = 64;
    while (nbytes--) {
        *p++ = value;
    }

    // DMA copy the the rest
    snrt_dma_txid_t memset_txid =
        snrt_dma_start_2d(ptr, ptr, 64, 64, 0, len / 64);
    snrt_dma_wait_all();
}

//================================================================================
// Matrix tile functions
//================================================================================


/// Load a 2D-tile of shape (tile_x1_size, tile_x0_size) from the 2D array
/// of shape (full_x1_size, full_x0_size). The specific tile is selected
/// by the (tile_x1_idx, tile_x0_idx) tuple. Every element in the src and
/// destination arrays has prec bytes.
inline snrt_dma_txid_t snrt_dma_load_2d_tile(void *dst, void *src,
                                             size_t tile_x1_idx, size_t tile_x0_idx,
                                             size_t tile_x1_size, size_t tile_x0_size,
                                             size_t full_x0_size, uint32_t prec) {
    size_t src_offset = 0;
    // Advance src array in x0 and x1 dimensions, and convert to byte offset
    src_offset += tile_x0_idx * tile_x0_size;
    src_offset += tile_x1_idx * tile_x1_size * full_x0_size;
    src_offset *= prec;
    // Initiate transfer
    return snrt_dma_start_2d(
        dst,                  // dst
        src + src_offset,     // src
        tile_x0_size * prec,  // size
        tile_x0_size * prec,  // dst_stride
        full_x0_size * prec,  // src_stride
        tile_x1_size          // repeat
    );
}

/// Store a 2D-tile of shape (tile_x1_size, tile_x0_size) to the 2D array
/// of shape (full_x1_size, full_x0_size). The specific tile is selected
/// by the (tile_x1_idx, tile_x0_idx) tuple. Every element in the src and
/// destination arrays has prec bytes.
inline snrt_dma_txid_t snrt_dma_store_2d_tile(void *dst, void *src,
                                             size_t tile_x1_idx, size_t tile_x0_idx,
                                             size_t tile_x1_size, size_t tile_x0_size,
                                             size_t full_x0_size, uint32_t prec) {
    size_t dst_offset = 0;
    // Advance dst array in x0 and x1 dimensions, and convert to byte offset
    dst_offset += tile_x0_idx * tile_x0_size;
    dst_offset += tile_x1_idx * tile_x1_size * full_x0_size;
    dst_offset *= prec;
    // Initiate transfer
    return snrt_dma_start_2d(
        dst + dst_offset,     // dst
        src,                  // src
        tile_x0_size * prec,  // size
        full_x0_size * prec,  // dst_stride
        tile_x0_size * prec,  // src_stride
        tile_x1_size          // repeat
    );
}

//================================================================================
// Reduction functions
//================================================================================

// Assumes the dst and src buffers are at the same offset in the TCDM of every
// cluster
inline void snrt_global_reduction_dma(double* dst_buffer, double* src_buffer,
                                      size_t len) {
    // If we have a single cluster the reduction degenerates to a memcpy
    if (snrt_cluster_num() == 1) {
        if (!snrt_is_compute_core()) {
            snrt_dma_start_1d(dst_buffer, src_buffer, len * sizeof(double));
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    } else {
        // Iterate levels in the binary reduction tree
        int num_levels = ceil(log2(snrt_cluster_num()));
        for (unsigned int level = 0; level < num_levels; level++) {

            // Determine whether the current cluster is an active cluster.
            // An active cluster is a cluster that participates in the current
            // level of the reduction tree. Every second cluster among the active
            // ones is a sender.
            uint32_t is_active = (snrt_cluster_idx() % (1 << level)) == 0;
            uint32_t is_sender = (snrt_cluster_idx() % (1 << (level + 1))) != 0;

            // If the cluster is a sender, it sends the data in its source
            // buffer to the respective receiver's destination buffer
            if (is_active && is_sender) {
                if (!snrt_is_compute_core()) {
                    void *dst = (void *)dst_buffer -
                        (1 << level) * SNRT_CLUSTER_OFFSET;
                    snrt_dma_start_1d(dst, src_buffer, len * sizeof(double));
                    snrt_dma_wait_all();
                }
            }

            // Synchronize senders and receivers
            snrt_global_barrier();

            // Every cluster which is not a sender performs the reduction
            if (is_active && !is_sender) {
                // Computation is parallelized over the compute cores
                if (snrt_is_compute_core()) {
                    uint32_t items_per_core =
                        len / snrt_cluster_compute_core_num();
                    uint32_t core_offset =
                        snrt_cluster_core_idx() * items_per_core;
                    for (uint32_t i = 0; i < items_per_core; i++) {
                        uint32_t abs_i = core_offset + i;
                        dst_buffer[abs_i] += src_buffer[abs_i];
                    }
                }
            }

            // Synchronize compute and DM cores for next tree level
            snrt_cluster_hw_barrier();
        }
    }
}
