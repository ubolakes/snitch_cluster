#pragma once

#include <snrt.h>

#include <dump.h>
NAMED_DUMP(uint32_t, aIdx, 0x1a)
NAMED_DUMP(uint32_t, bIdx, 0x1b)
NAMED_DUMP(uint32_t, cIdx, 0x1c)
NAMED_DUMP(uint32_t, ib, 0x10)
NAMED_DUMP(uint32_t, jb, 0x11)
NAMED_DUMP(uint32_t, kb, 0x12)
NAMED_DUMP(double, a, 0xa)
NAMED_DUMP(double, b, 0xb)
NAMED_DUMP(double, c, 0xc)

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)
#define CONCAT_IMPL(a,b) a ## b
#define CONCAT(a,b) CONCAT_IMPL(a,b)
#define CONCAT3(a,b,c) CONCAT(a,CONCAT(b,c))
#define CONCAT4(a,b,c,d) CONCAT3(a,b,CONCAT(c,d))

#ifndef PRECISION_T
#define PRECISION_T
typedef enum { FP64 = 8, FP32 = 4, FP16 = 2, FP8 = 1 } precision_t;

typedef double fp64;
typedef float  fp32;
typedef __fp16 fp16;
typedef char   fp8;

typedef fp32 v2f32 __attribute__((vector_size(8)));
typedef fp16 v4f16 __attribute__((vector_size(8)));
typedef fp8  v8f8  __attribute__((vector_size(8)));

#define VECTOR_SIZE(type) 8 / sizeof(type)
#endif

/// Constants related to a GEMM computation to precompute and initialize
typedef struct {
    precision_t prec;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t ta; // false = row-major, true = col-major
    uint32_t tb;
    uint32_t tc;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
} SnblasGemmInfo;

/**
 * Constants related to which implementation should be used. 
 * Only for non-template parameters
*/
typedef struct {
    bool bench;   // Enable benchmarking code
    bool ta_tile; // Transpose the A tile when loading into TCDM
    bool tb_tile;
    bool tc_tile;
} SnblasGemmImpl;

#define L1_LDA L1_K
#define L1_LDB L1_N
#define L1_LDC L1_N

#define FLOAT_T fp64
#include "gemm_decls_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_decls_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_decls_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_decls_tpl.h"
#undef FLOAT_T

/**
 * \brief Implements a reversing loop for an index range
 * \param begin Beginning of the range
 * \param end End of the range
 * \param dir Sets the direction of traversal. True: loop starts at begin.
 * \param i_prev Set the previous index to the first index, must update this
 * manually at the end of the loop. \details i_end_floor will contain the exact
 * end with the stride, s.t. the reversed loop starts at the correct index.
 */
#define FOR_EACH(i, begin, end, stride, dir, i_prev)                     \
    dir = !dir;                                                          \
    const int i##_end_floor =                                            \
        ((end - begin + stride - 1) / stride) * stride - stride + begin; \
    const int i##_first = dir ? begin : i##_end_floor;                   \
    const int i##_last = dir ? i##_end_floor : begin;                    \
    i = i##_first;                                                       \
    for (; dir ? i <= i##_last : i >= i##_last;                          \
    i = dir ? i + stride : i - stride)


// -- Function pointer typedefs
typedef snrt_dma_txid_t (*snrt_dma_load_2d_tile_t)(void *, void *, size_t, size_t, size_t, size_t, size_t, uint32_t);
