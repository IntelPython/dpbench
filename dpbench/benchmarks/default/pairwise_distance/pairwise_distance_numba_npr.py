# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def pairwise_distance(X1, X2, D):
    X1_rows = X1.shape[0]
    X2_rows = X2.shape[0]
    X1_cols = X1.shape[1]
    for i in nb.prange(X1_rows):
        for j in range(X2_rows):
            d = 0.0
            for k in range(X1_cols):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
