# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_l2_distance
import numpy as np
# import numba_dppy
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id, atomic, DEFAULT_LOCAL_SIZE
atomic_add = atomic.add

import math


# @kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
@kernel
def l2_distance_kernel(a, b, c):
    i = get_global_id(0)
    j = get_global_id(1)
    sub = a[i, j] - b[i, j]
    sq = sub ** 2
    atomic_add(c, 0, sq)


def l2_distance(a, b, distance):
    with dpctl.device_context(base_l2_distance.get_device_selector()):
        l2_distance_kernel[(a.shape[0], a.shape[1]), DEFAULT_LOCAL_SIZE](
            a, b, distance
        )

    return math.sqrt(distance)


base_l2_distance.run("l2 distance kernel", l2_distance)
