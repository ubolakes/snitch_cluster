#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys
from pathlib import Path
import numpy as np
from data.datagen import golden_model, visualize_clusters

sys.path.append(str(Path(__file__).parent / '../../../util/sim/'))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import bytes_to_doubles, bytes_to_uint32s  # noqa: E402


ERR_THRESHOLD = 1E-10


def main():
    # Run simulation and get outputs
    args = verification.parse_args()
    raw_results = verification.simulate(sim_bin=args.sim_bin,
                                        snitch_bin=args.snitch_bin,
                                        symbols_bin=args.symbols_bin,
                                        log=args.log,
                                        output_uids=['centroids'])
    centroids_actual = np.array(bytes_to_doubles(raw_results['centroids']))

    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)
    max_iter = bytes_to_uint32s(elf.get_symbol_contents('n_iter'))[0]
    n_clusters = bytes_to_uint32s(elf.get_symbol_contents('n_clusters'))[0]
    n_features = bytes_to_uint32s(elf.get_symbol_contents('n_features'))[0]
    n_samples = bytes_to_uint32s(elf.get_symbol_contents('n_samples'))[0]
    initial_centroids = np.array(bytes_to_doubles(elf.get_symbol_contents('centroids')))
    samples = np.array(bytes_to_doubles(elf.get_symbol_contents('samples')))

    # Reshape
    samples = samples.reshape((n_samples, n_features))
    initial_centroids = initial_centroids.reshape((n_clusters, n_features))
    centroids_actual = centroids_actual.reshape((n_clusters, n_features))

    # Visualize centroids computed in simulation
    visualize_clusters(samples, initial_centroids, "Initial centroids")
    visualize_clusters(samples, centroids_actual, "Actual centroids")

    # Verify results
    centroids_golden, _ = golden_model(samples, n_clusters, initial_centroids, max_iter)
    visualize_clusters(samples, centroids_golden, "Golden centroids")
    relative_err = np.absolute((centroids_golden - centroids_actual) / centroids_golden)
    fail = np.any(relative_err > ERR_THRESHOLD)
    if (fail):
        verification.dump_results_to_csv([centroids_golden, centroids_actual, relative_err],
                                         Path.cwd() / 'kmeans_results.csv')
    return int(fail)


if __name__ == "__main__":
    sys.exit(main())
