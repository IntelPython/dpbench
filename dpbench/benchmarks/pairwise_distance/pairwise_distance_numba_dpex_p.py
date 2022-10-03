# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def pairwise_distance(X1, X2, D):
    """Naïve pairwise distance impl - take an array representing M points in N
    dimensions, and return the M x M matrix of Euclidean distances

    Args:
        X1 : Set of points
        X2 : Set of points
        D  : Outputted distance matrix
    """
    # Size of inputs
    M = X1.shape[0]
    N = X2.shape[0]
    O = X1.shape[1]

    # Outermost parallel loop over the matrix X1
    for i in nb.prange(M):
        # Loop over the matrix X2
        for j in range(N):
            d = 0.0
            # Compute exclidean distance
            for k in range(O):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            # Write computed distance to distance matrix
            D[i, j] = np.sqrt(d)
