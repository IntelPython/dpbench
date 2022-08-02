# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import numba as nb

__njit = nb.njit(parallel=False, fastmath=True)


@__njit
def l2_distance_kernel(a, b):
    sub = a - b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d


def l2_distance(a, b):
    return l2_distance_kernel(a, b)
