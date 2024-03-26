# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(npoints, niters, seed, ndims, ncentroids, types_dict):
    import numpy as np
    import numpy.random as default_rng

    f_dtype: np.dtype = types_dict["float"]
    i_dtype: np.dtype = types_dict["int"]
    XL = f_dtype.type(1.0)
    XH = f_dtype.type(5.0)

    default_rng.seed(seed)

    arrayP = default_rng.uniform(XL, XH, (npoints, ndims)).astype(f_dtype)
    arrayPclusters = np.ones(npoints, dtype=i_dtype)
    arrayC = np.empty((ncentroids, ndims), dtype=f_dtype)
    arrayCnumpoint = np.ones(ncentroids, dtype=np.int64)

    arrayC[:] = arrayP[:ncentroids]

    return (arrayP, arrayPclusters, arrayC, arrayCnumpoint)
