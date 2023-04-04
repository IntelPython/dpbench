# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def l2_norm(a, d):
    for i in nb.prange(a.shape[0]):
        for k in range(a.shape[1]):
            d[i] += np.square(a[i, k])
        d[i] = np.sqrt(d[i])
