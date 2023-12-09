# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil, sqrt

from numba import cuda


@cuda.jit
def _pairwise_distance_kernel(X1, X2, D):
    i, j = cuda.grid(2)

    X2_cols = X2.shape[1]

    d = X1.dtype.type(0.0)
    for k in range(X2_cols):
        tmp = X1[i, k] - X2[j, k]
        d += tmp * tmp
    D[i, j] = sqrt(d)


def pairwise_distance(X1, X2, D):
    threadsperblock = (16, 16)
    blockspergrid_x = ceil(X1.shape[0] / threadsperblock[0])
    blockspergrid_y = ceil(X2.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _pairwise_distance_kernel[blockspergrid, threadsperblock](X1, X2, D)
