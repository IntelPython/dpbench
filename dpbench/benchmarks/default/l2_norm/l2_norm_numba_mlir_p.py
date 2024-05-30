# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba
import numba_mlir as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def _l2_norm(a, d):
    for i in numba.prange(a.shape[0]):
        d[i] = 0.0
        for k in range(a.shape[1]):
            d[i] += np.square(a[i, k])
        d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    _l2_norm(a, d)
