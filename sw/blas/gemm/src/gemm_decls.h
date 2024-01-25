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

#ifndef PRECISION_T
#define PRECISION_T
typedef enum { FP64 = 8, FP32 = 4, FP16 = 2, FP8 = 1 } precision_t;

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));
#endif

/// Constants related to a GEMM computation to precompute and initialize
typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t ta;
    uint32_t tb;
    precision_t prec;
} GemmInfo;

/// Arguments to execute a GEMM computation, given a corresponding GemmInfo instance
typedef struct {
    const double* A;
    const double* B;
    double* C;
    double alpha;
    double beta;
} GemmArgs;

#define L1_M 8
#define L1_N 8
#define L1_K 8
#define L1_LDA L1_K
#define L1_LDB L1_N
#define L1_LDC L1_N

/**
 * \brief Maps the layout of the TCDM. May be double buffered.
 */
typedef struct {
    double A[L1_M * L1_K];
    double B[L1_K * L1_N];
    double C[L1_M * L1_N];
} TcdmLayout;

NAMED_DUMP(TcdmLayout*, l1, 0x8)


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