# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(npoints, niters, seed, ndims, ncentroids):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.float64
    XL = 1.0
    XH = 5.0

    default_rng.seed(seed)

    arrayP = default_rng.uniform(XL, XH, (npoints, ndims)).astype(dtype)
    arrayPclusters = np.ones(npoints, dtype=np.int64)
    arrayC = np.ones((ncentroids, ndims), dtype=dtype)
    arrayCsum = np.ones((ncentroids, ndims), dtype=dtype)
    arrayCnumpoint = np.ones(ncentroids, dtype=np.int64)

    return (arrayP, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint)
