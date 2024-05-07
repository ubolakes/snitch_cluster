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


class CorrelationDataGen(DataGen):

    def golden_model(self, data):
        return np.corrcoef(data, rowvar=False)

    def validate_config(self, M, N, **kwargs):
        assert (M % 8) == 0, "M must be an integer multiple of the number of cores"

        # Calculate total TCDM occupation
        data_size = N * M * 8
        corr_size = M * M * 8
        stddev_size = M * 8
        total_size = data_size
        total_size += corr_size
        total_size += stddev_size
        data_utils.validate_tcdm_footprint(total_size)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # Validate parameters
        self.validate_config(**kwargs)

        M, N = kwargs['M'], kwargs['N']
        data = np.random.random_integers(-200, 100, size=(N, M))/100
        corr = self.golden_model(data)

        data = data.flatten()
        corr = corr.flatten()

        header += [format_scalar_definition('uint32_t', 'M', M)]
        header += [format_scalar_definition('uint32_t', 'N', N)]
        header += [format_array_definition('double', 'data', data, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        header += [format_array_declaration('double', 'corr', corr.shape,
                   alignment=BURST_ALIGNMENT, section=kwargs['section'])]
        result_def = format_array_definition('double', 'golden', corr, alignment=BURST_ALIGNMENT)
        header += [format_ifdef_wrapper('BIST', result_def)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    CorrelationDataGen().main()
