# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nopt, niters, seed, ndims=2, ncentroids=10):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.float64
    XL = 1.0
    XH = 5.0

    default_rng.seed(seed)

    arrayP = default_rng.uniform(XL, XH, (nopt, ndims)).astype(dtype)
    arrayPclusters = np.ones(nopt, dtype=np.int32)
    arrayC = np.ones((ncentroids, 2), dtype=dtype)
    arrayCsum = np.ones((ncentroids, 2), dtype=dtype)
    arrayCnumpoint = np.ones(ncentroids, dtype=np.int32)

    return (
        arrayP,
        arrayPclusters,
        arrayC,
        arrayCsum,
        arrayCnumpoint,
        niters,
        nopt,
        ncentroids,
    )
