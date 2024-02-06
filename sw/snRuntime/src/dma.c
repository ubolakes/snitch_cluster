// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern snrt_dma_txid_t snrt_dma_start_1d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size);

extern snrt_dma_txid_t snrt_dma_start_1d(void *dst, const void *src,
                                         size_t size);

extern snrt_dma_txid_t snrt_dma_start_2d_wideptr(uint64_t dst, uint64_t src,
                                                 size_t size, size_t dst_stride,
                                                 size_t src_stride,
                                                 size_t repeat);

extern snrt_dma_txid_t snrt_dma_start_2d(void *dst, const void *src,
                                         size_t size, size_t dst_stride,
                                         size_t src_stride, size_t repeat);

extern void snrt_dma_wait(snrt_dma_txid_t tid);

extern void snrt_dma_wait_all();

extern snrt_dma_txid_t snrt_dma_load_2d_tile(
    void *dst, void *src, size_t tile_x1_idx, size_t tile_x0_idx,
    size_t tile_x1_size, size_t tile_x0_size, size_t full_x0_size,
    uint32_t prec);

extern snrt_dma_txid_t snrt_dma_load_2d_tile_transpose(
    void *dst, void *src, size_t tile_x1_idx, size_t tile_x0_idx,
    size_t tile_x1_size, size_t tile_x0_size, size_t full_x0_size,
    uint32_t prec);

extern snrt_dma_txid_t snrt_dma_store_2d_tile(
    void *dst, void *src, size_t tile_x1_idx, size_t tile_x0_idx,
    size_t tile_x1_size, size_t tile_x0_size, size_t full_x0_size,
    uint32_t prec);

extern snrt_dma_txid_t snrt_dma_store_2d_tile_transpose(
    void *dst, void *src, size_t tile_x1_idx, size_t tile_x0_idx,
    size_t tile_x1_size, size_t tile_x0_size, size_t full_x0_size,
    uint32_t prec);