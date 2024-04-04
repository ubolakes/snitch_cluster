#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Luca Colagrande <colluca@iis.ee.ethz.ch>

import argparse
import json5
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import emit_license, format_scalar_definition, \
                       format_vector_definition, format_ifdef_wrapper  # noqa: E402


def test(**kwargs):

    n_samples = kwargs['n_samples']
    n_features = kwargs['n_features']
    n_clusters = kwargs['n_clusters']
    max_iter = kwargs['max_iter']
    seed = kwargs['seed']

    # Generate random samples    
    samples, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=seed
    )

    # Generate initial centroids randomly
    rng = np.random.default_rng(seed=seed)
    centroids = rng.uniform(low=samples.min(axis=0), high=samples.max(axis=0), size=(n_clusters, n_features))

    for k in range(max_iter):
        # Assignment step
        membership = []
        membership_cnt = [0 for i in range(n_clusters)]
        for i in range(n_samples):
            min_dist = float('inf')
            membership.append(0)
            for j in range(n_clusters):
                dist = 0
                for k in range(n_features):
                    # print(centroids[j][k])
                    dist += (samples[i][k] - centroids[j][k]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    membership[-1] = j
            membership_cnt[membership[-1]] += 1

        print(membership_cnt)

        # Update step
        centroids = np.zeros((n_clusters, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                centroids[membership[i]] += samples[i][j]
        for i in range(n_clusters):
            for j in range(n_features):
                centroids[i][j] /= membership_cnt[i]
                # print(centroids[i][j])


def main():

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = json5.loads(f.read())

    # Test
    test(**param)


if __name__ == '__main__':
    main()
