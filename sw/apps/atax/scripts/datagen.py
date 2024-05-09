#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Jose Pedro Castro Fonseca <jose.pc.fonseca@gmail, jcastro@ethz.ch>
#         Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
import data_utils  # noqa: E402
from data_utils import format_scalar_definition, format_array_definition, \
                       format_array_declaration, format_ifdef_wrapper, DataGen  # noqa: E402


# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096


class AtaxDataGen(DataGen):

    def golden_model(self, A, x):
        return np.matmul(A.transpose(), np.matmul(A, x))

    def validate_config(self, M, N, **kwargs):
        assert (N % 8) == 0, "N must be an integer multiple of the number of cores"

        # Calculate total TCDM occupation
        a_size = M * N * 8
        x_size = N * 8
        y_size = N * 8
        tmp_size = M * 8
        total_size = a_size
        total_size += x_size
        total_size += y_size
        total_size += tmp_size
        data_utils.validate_tcdm_footprint(total_size)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # Validate parameters
        self.validate_config(**kwargs)

        M, N = kwargs['M'], kwargs['N']
        A = np.random.random_integers(-200, 100, size=(M, N))/100
        x = np.random.random_integers(-200, 100, size=(N, 1))/100
        y = self.golden_model(A, x)

        A = A.flatten()
        x = x.flatten()
        y = y.flatten()

        header += [format_scalar_definition('uint32_t', 'M', M)]
        header += [format_scalar_definition('uint32_t', 'N', N)]
        header += [format_array_definition('double', 'A', A, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        header += [format_array_definition('double', 'x', x, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        header += [format_array_declaration('double', 'y', y.shape, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        result_def = format_array_definition('double', 'golden', y, alignment=BURST_ALIGNMENT)
        header += [format_ifdef_wrapper('BIST', result_def)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    AtaxDataGen().main()
