# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def pairwise_distance(X1, X2, D):
    """Pairwise Numpy implementation using the equation:
    (a-b)^2 = a^2 + b^2 - 2*a*b
    """
    M = X1.shape[0]
    N = X2.shape[0]
    O = X1.shape[1]
    for i in range(M):
        for j in range(N):
            d = 0.0
            for k in range(O):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
