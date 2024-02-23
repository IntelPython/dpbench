# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(npoints, dims, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype = types_dict["float"]

    default_rng.seed(seed)

    return (
        default_rng.random((npoints, dims)).astype(dtype),
        default_rng.random((npoints, dims)).astype(dtype),
        np.empty((npoints, npoints), dtype),
    )
