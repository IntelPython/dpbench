# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=False, fastmath=True)
def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)
