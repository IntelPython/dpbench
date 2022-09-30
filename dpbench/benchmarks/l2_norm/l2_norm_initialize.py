# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0


def initialize(ndims, seed):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.float64

    default_rng.seed(seed)

    return (
        default_rng.random(ndims).astype(dtype),
        np.empty(1),
    )
