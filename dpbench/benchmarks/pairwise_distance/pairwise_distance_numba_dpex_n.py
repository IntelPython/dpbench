# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def _pairwise_distance(X1, X2):
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)
    D = -2 * np.dot(X1, X2.T)
    x3 = x1.reshape(x1.size, 1)
    D = D + x3
    D = D + x2
    return np.sqrt(D)


def pairwise_distance(X1, X2, D):
    np.copyto(D, _pairwise_distance(X1, X2))
