#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Authors: Tim Fischer     <fischeti@iis.ee.ethz.ch>
#          Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import emit_license, format_scalar_definition, \
                       format_vector_definition, format_ifdef_wrapper  # noqa: E402


np.random.seed(42)

C_TYPES = {
  '64': 'double',
  '32': 'float',
  '16': '__fp16',
  '8': 'char'
}

NUMPY_TYPES = {
  '64': np.double,
  '32': np.single,
  '16': np.half,
  '8': np.ubyte
}

FP8_FORMATS = {
    'fp8': {'exp': 5, 'mant': 2},
    'fp8alt': {'exp': 4, 'mant': 3}
}

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096


def golden_model(alpha, a, b, beta, c):
    return alpha * np.matmul(a, b) + beta * c


def emit_header(**kwargs):
    gemmInfo = kwargs['gemmInfo']
    gemmArgs = kwargs['gemmArgs']
    gemmImpl = kwargs['gemmImpl']

    # Generate random input matrices
    dtype = NUMPY_TYPES[str(gemmInfo['prec'])]
    if (gemmInfo['prec']) == 8:
        # sign -1 or 1
        sign_a = np.random.randint(0, 2, (gemmInfo['M'], gemmInfo['K'])).astype(dtype)
        # esponent < 0b01111
        exponent_a = np.random.randint(0, 16, (gemmInfo['M'], gemmInfo['K'])).astype(dtype)
        # mantissa can be arbitrary
        mantissa_a = np.random.randint(0, 4, (gemmInfo['M'], gemmInfo['K'])).astype(dtype)
        # sign -1 or 1
        sign_b = np.random.randint(0, 2, (gemmInfo['K'], gemmInfo['N'])).astype(dtype)
        # esponent < 0b01111
        exponent_b = np.random.randint(0, 16, (gemmInfo['K'], gemmInfo['N'])).astype(dtype)
        # mantissa can be arbitrary
        mantissa_b = np.random.randint(0, 4, (gemmInfo['K'], gemmInfo['N'])).astype(dtype)
        # sign -1 or 1
        sign_c = np.random.randint(0, 2, (gemmInfo['M'], gemmInfo['N'])).astype(dtype)
        # esponent < 0b01111
        exponent_c = np.random.randint(0, 16, (gemmInfo['M'], gemmInfo['N'])).astype(dtype)
        # mantissa can be arbitrary
        mantissa_c = np.random.randint(0, 4, (gemmInfo['M'], gemmInfo['N'])).astype(dtype)
        _a = ((-1.0)**sign_a.astype(np.double))*(2.0**(exponent_a.astype(np.double)-15.0)) \
            * (1.0 + mantissa_a.astype(np.double) / (2**2))
        _b = ((-1.0)**sign_b.astype(np.double))*(2.0**(exponent_b.astype(np.double)-15.0)) \
            * (1.0 + mantissa_b.astype(np.double) / (2**2))
        _c = ((-1.0)**sign_c.astype(np.double))*(2.0**(exponent_c.astype(np.double)-15.0)) \
            * (1.0 + mantissa_c.astype(np.double) / (2**2))
        result = golden_model(1, _a, _b, gemmArgs['beta'], _c)
        a = sign_a << 7 | exponent_a << FP8_FORMATS['fp8']['mant'] | mantissa_a
        b = sign_b << 7 | exponent_b << FP8_FORMATS['fp8']['mant'] | mantissa_b
        c = sign_c << 7 | exponent_c << FP8_FORMATS['fp8']['mant'] | mantissa_c
    else:
        if kwargs['datagen']['linspace']:
            a = np.linspace(0.1, gemmInfo['M'] * gemmInfo['K'] + 0.1 -1, num=gemmInfo['M'] * gemmInfo['K']).reshape((gemmInfo['M'], gemmInfo['K'])).astype(dtype)
            b = np.linspace(0.2, gemmInfo['K'] * gemmInfo['N'] + 0.2 -1, num=gemmInfo['K'] * gemmInfo['N']).reshape((gemmInfo['K'], gemmInfo['N'])).astype(dtype)
            c = np.linspace(0.3, gemmInfo['M'] * gemmInfo['N'] + 0.3 -1, num=gemmInfo['M'] * gemmInfo['N']).reshape((gemmInfo['M'], gemmInfo['N'])).astype(dtype)
        else:
            a = np.random.rand(gemmInfo['M'], gemmInfo['K']).astype(dtype)
            b = np.random.rand(gemmInfo['K'], gemmInfo['N']).astype(dtype)
            c = np.random.rand(gemmInfo['M'], gemmInfo['N']).astype(dtype)
        result = golden_model(gemmArgs['alpha'], a, b, gemmArgs['beta'], c)

    # Store matrices in transposed form if requested
    a = a.T if gemmInfo['ta'] else a
    b = b.T if gemmInfo['tb'] else b
    c = c.T if gemmInfo['tc'] else c
    result = result.T if gemmInfo['tc'] else result

    data_str = [emit_license()]
    data_str = ["#pragma once"]

    data_str += ["// -- gemmInfo"]
    data_str += [f"#define DTYPE fp{gemmInfo['prec']}"]
    # data_str += [format_scalar_definition('uint32_t', 'dtype_size', gemmInfo['prec']//8)]
    data_str += [format_scalar_definition('uint32_t', 'M', gemmInfo['M'])]
    data_str += [format_scalar_definition('uint32_t', 'N', gemmInfo['N'])]
    data_str += [format_scalar_definition('uint32_t', 'K', gemmInfo['K'])]
    data_str += [format_scalar_definition('uint32_t', 'TA', int(gemmInfo['ta']))]
    data_str += [format_scalar_definition('uint32_t', 'TB', int(gemmInfo['tb']))]
    data_str += [format_scalar_definition('uint32_t', 'TC', int(gemmInfo['tc']))]
    
    # gemmArgs
    data_str += ["// -- gemmArgs"]
    data_str += [format_scalar_definition('double', 'ALPHA', gemmArgs['alpha'])]
    data_str += [format_scalar_definition('double', 'BETA', gemmArgs['beta'])]

    # gemmImpl
    data_str += ["// -- gemmImpl"]
    data_str += [f"#define USE_METHOD {gemmImpl['method']}"]
    data_str += [f"#define L1_M {gemmImpl['L1_M']}"]
    data_str += [f"#define L1_N {gemmImpl['L1_N']}"]
    data_str += [f"#define L1_K {gemmImpl['L1_K']}"]
    data_str += [format_scalar_definition('uint32_t', 'TA_TILE', int(gemmImpl['ta_tile']))]
    data_str += [format_scalar_definition('uint32_t', 'TB_TILE', int(gemmImpl['tb_tile']))]
    data_str += [format_scalar_definition('uint32_t', 'TC_TILE', int(gemmImpl['tc_tile']))]
    data_str += [format_scalar_definition('uint32_t', 'expand', gemmImpl['expand'])]
    data_str += [f"#define FMADD_D_UNROLL {gemmImpl['fmadd_d_unroll']}"]

    # bench
    data_str += ["// -- bench"]
    data_str += [format_scalar_definition('uint32_t', 'bench_iters', kwargs['bench']['iters'])]

    # datagen
    data_str += ["// -- datagen"]
    data_str += [format_vector_definition(C_TYPES[str(gemmInfo['prec'])], 'a', a.flatten(),
                 alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    data_str += [format_vector_definition(C_TYPES[str(gemmInfo['prec'])], 'b', b.flatten(),
                 alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    data_str += [format_vector_definition(C_TYPES[str(gemmInfo['prec'])], 'c', c.flatten(),
                 alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    if gemmInfo['prec'] == 8:
        result_def = format_vector_definition(C_TYPES['64'], 'result', result.flatten())
    else:
        result_def = format_vector_definition(C_TYPES[str(gemmInfo['prec'])],
                                              'result',
                                              result.flatten())
    data_str += [format_ifdef_wrapper('BIST', result_def)]
    data_str = '\n\n'.join(data_str)

    return data_str


def main():

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel'
    )
    parser.add_argument(
        '--section',
        type=str,
        help='Section to store matrices in')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param['section'] = args.section

    # Emit header file
    print(emit_header(**param))


if __name__ == '__main__':
    main()
