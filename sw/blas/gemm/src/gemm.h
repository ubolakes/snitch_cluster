#pragma once

#include "gemm_kernel.h"
#ifdef OCCAMY
// #include "gemm_baseline.h"
// #include "gemm_1dpipe.h"
#include "gemm_2dpipe.h"
#endif