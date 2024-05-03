#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../util/sim/"))
from data_utils import format_scalar_definition, DataGen  # noqa: E402


class MonteCarloDataGen(DataGen):

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        n_samples = kwargs['n_samples']

        assert (n_samples % 8) == 0, "Number of samples must be an integer multiple of the" \
                                     " number of cores"

        header += [format_scalar_definition('uint32_t', 'n_samples', n_samples)]
        header = '\n\n'.join(header)

        return header


if __name__ == '__main__':
    MonteCarloDataGen().main()
