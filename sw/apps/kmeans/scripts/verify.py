#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import sys
from pathlib import Path
from datagen import KmeansDataGen

sys.path.append(str(Path(__file__).parent / '../../../util/sim/'))
from verif_utils import Verifier  # noqa: E402


class KmeansVerifier(Verifier):

    OUTPUT_UIDS = ['centroids']

    def parser(self):
        p = super().parser()
        p.add_argument(
            '--no-gui',
            action='store_true',
            help='Run without GUI')
        return p

    def get_actual_results(self):
        return self.get_output_from_symbol(self.OUTPUT_UIDS[0], 'double')

    def get_expected_results(self):
        # Get inputs
        max_iter = self.get_input_from_symbol('n_iter', 'uint32_t')[0]
        n_clusters = self.get_input_from_symbol('n_clusters', 'uint32_t')[0]
        self.n_features = self.get_input_from_symbol('n_features', 'uint32_t')[0]
        n_samples = self.get_input_from_symbol('n_samples', 'uint32_t')[0]
        self.initial_centroids = self.get_input_from_symbol('centroids', 'double')
        self.samples = self.get_input_from_symbol('samples', 'double')
        # Reshape arrays
        self.initial_centroids = self.initial_centroids.reshape((n_clusters, self.n_features))
        self.samples = self.samples.reshape((n_samples, self.n_features))
        # Calculate expected results
        final_centroids, _ = KmeansDataGen().golden_model(self.samples, n_clusters, self.initial_centroids, max_iter)
        return final_centroids.flatten()

    def check_results(self, *args):
        return super().check_results(*args, rtol=1e-10)

    def main(self):
        retcode = super().main()

        # Visualize results
        expected_centroids = self.get_expected_results()
        actual_centroids = self.get_actual_results()
        if self.n_features == 2 and not self.args.no_gui:
            visualize_clusters(self.samples, self.initial_centroids, "Initial centroids")
            visualize_clusters(self.samples, expected_centroids, "Expected centroids")
            visualize_clusters(self.samples, actual_centroids, "Actual centroids")

        return retcode


if __name__ == "__main__":
    sys.exit(KmeansVerifier().main())
