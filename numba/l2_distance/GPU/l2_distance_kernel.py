# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
import sys

import base_l2_distance
import dpctl
import numba_dpex
from numba_dpex import DEFAULT_LOCAL_SIZE, atomic, kernel

atomic_add = atomic.add


@kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
def l2_distance_kernel(a, b, c):
    i = numba_dpex.get_global_id(0)
    j = numba_dpex.get_global_id(1)
    sub = a[i, j] - b[i, j]
    sq = sub**2
    atomic_add(c, 0, sq)


def l2_distance(a, b, distance):
    with dpctl.device_context(
        base_l2_distance.get_device_selector(is_gpu=True)
    ):
        l2_distance_kernel[(a.shape[0], a.shape[1]), DEFAULT_LOCAL_SIZE](
            a, b, distance
        )
    return math.sqrt(distance)


base_l2_distance.run("l2 distance kernel", l2_distance)
