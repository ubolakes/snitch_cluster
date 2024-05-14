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


class CovarianceDataGen(DataGen):

    def golden_model(self, data):
        return np.cov(data, rowvar=False)

    def validate_config(self, M, N, **kwargs):
        # Calculate total TCDM occupation
        data_size = N * M * 8
        cov_size = M * M * 8
        total_size = data_size
        total_size += 2 * cov_size
        data_utils.validate_tcdm_footprint(total_size)

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # Validate parameters
        self.validate_config(**kwargs)

        M, N = kwargs['M'], kwargs['N']
        data = np.random.random_integers(-200, 100, size=(N, M))
        cov = self.golden_model(data)

        data = data.flatten()
        cov = cov.flatten()

        header += [format_scalar_definition('uint32_t', 'M', M)]
        header += [format_scalar_definition('uint32_t', 'N', N)]
        header += [format_array_definition('double', 'data', data, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        header += [format_array_declaration('double', 'cov', cov.shape, alignment=BURST_ALIGNMENT,
                   section=kwargs['section'])]
        result_def = format_array_definition('double', 'golden', cov, alignment=BURST_ALIGNMENT)
        header += [format_ifdef_wrapper('BIST', result_def)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    CovarianceDataGen().main()
