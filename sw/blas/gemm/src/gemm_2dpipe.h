#pragma once

#define FLOAT_T fp64
#include "gemm_2dpipe_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp32
#include "gemm_2dpipe_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp16
#include "gemm_2dpipe_tpl.h"
#undef FLOAT_T

#define FLOAT_T fp8
#include "gemm_2dpipe_tpl.h"
#undef FLOAT_T