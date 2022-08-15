# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=False, fastmath=True)
def l2_distance(a, b):
    sub = a - b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d
