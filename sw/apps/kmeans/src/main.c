// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>

#include "data.h"
#include "kmeans.h"

int main() {
    kmeans(n_samples, n_features, n_clusters, n_iter, samples, centroids);
    return 0;
}
