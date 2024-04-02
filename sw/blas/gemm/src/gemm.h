#pragma once

#include "gemm_kernel.h"
#ifdef OCCAMY

// -- 2D Pipeline
#define METHOD 2dpipe
#define FLOAT_T fp64
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_tpl.h"
#undef FLOAT_T
#undef METHOD

// -- 1dpipe
#define METHOD 1dpipe
#define FLOAT_T fp64
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_tpl.h"
#undef FLOAT_T
#undef METHOD

// -- Baseline
#define METHOD baseline
#define FLOAT_T fp64
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_tpl.h"
#undef FLOAT_T
#undef METHOD

// -- Single Buffer
#define METHOD singlebuffer
#define FLOAT_T fp64
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_tpl.h"
#undef FLOAT_T
#undef METHOD

// -- Streaming Buffer
#define METHOD streambuffer
#define FLOAT_T fp64
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_tpl.h"
#undef FLOAT_T
#undef METHOD

#endif