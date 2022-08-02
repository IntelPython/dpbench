# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex
import numpy as np
from numba_dpex import DEFAULT_LOCAL_SIZE, kernel

atomic_add = numba_dpex.atomic.add


@kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
def l2_distance_kernel(a, b, c):
    i = numba_dpex.get_global_id(0)
    j = numba_dpex.get_global_id(1)
    sub = a[i, j] - b[i, j]
    sq = sub**2
    atomic_add(c, 0, sq)


def l2_distance(a, b):
    distance_np = np.asarray([0.0]).astype(np.float64)
    l2_distance_kernel[(a.shape[0], a.shape[1]), DEFAULT_LOCAL_SIZE](
        a, b, distance_np
    )
    return math.sqrt(distance_np)
