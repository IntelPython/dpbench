# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba_dpex
import numpy as np


@numba_dpex.kernel
def _pairwise_kernel(X1, X2, D):
    i = numba_dpex.get_global_id(0)

    N = X2.shape[0]
    O = X1.shape[1]
    for j in range(N):
        d = 0.0
        for k in range(O):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)


def pairwise_distance(X1, X2, D):
    _pairwise_kernel[X1.shape[0], numba_dpex.DEFAULT_LOCAL_SIZE](X1, X2, D)
