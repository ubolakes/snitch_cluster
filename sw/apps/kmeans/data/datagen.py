#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Luca Colagrande <colluca@iis.ee.ethz.ch>

import argparse
import hjson
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import emit_license, format_scalar_definition, \
                       format_vector_definition  # noqa: E402


# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096


def golden_model(samples, n_clusters, initial_centroids, max_iter):
    # Apply k-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=initial_centroids,
        max_iter=max_iter
    )
    kmeans.fit(samples)
    return kmeans.cluster_centers_, kmeans.n_iter_


def visualize_clusters(samples, centroids, title=None):
    plt.scatter(samples[:, 0], samples[:, 1], s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red')
    if not title:
        title = "K-means clusters"
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def emit_header(**kwargs):

    # Aliases
    n_samples = kwargs['n_samples']
    n_features = kwargs['n_features']
    n_clusters = kwargs['n_clusters']
    max_iter = kwargs['max_iter']
    seed = kwargs['seed']
    gui = not kwargs['no_gui']

    # Generate random samples
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=seed
    )

    # Generate initial centroids randomly
    rng = np.random.default_rng(seed=seed)
    initial_centroids = rng.uniform(low=X.min(axis=0), high=X.max(axis=0),
                                    size=(n_clusters, n_features))

    # Visualize the generated samples
    if gui:
        visualize_clusters(X, initial_centroids)

    # Apply k-means clustering
    centers, n_iter = golden_model(X, n_clusters, initial_centroids, max_iter)

    # Visualize the clusters
    if gui:
        visualize_clusters(X, centers)

    # Generate header
    data_str = [emit_license()]
    data_str += [format_scalar_definition('uint32_t', 'n_samples', n_samples)]
    data_str += [format_scalar_definition('uint32_t', 'n_features', n_features)]
    data_str += [format_scalar_definition('uint32_t', 'n_clusters', n_clusters)]
    data_str += [format_scalar_definition('uint32_t', 'n_iter', n_iter)]
    data_str += [format_vector_definition('double', 'centroids', initial_centroids.flatten(),
                 alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    data_str += [format_vector_definition('double', 'samples', X.flatten(),
                 alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    data_str = '\n\n'.join(data_str)
    return data_str


def main():

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel')
    parser.add_argument(
        '--section',
        type=str,
        help='Section to store matrices in')
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without visualization')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param['section'] = args.section
    param['no_gui'] = args.no_gui

    # Emit header file
    print(emit_header(**param))


if __name__ == '__main__':
    main()
