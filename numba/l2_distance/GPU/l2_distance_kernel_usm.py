# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_l2_distance
import numpy as np
import numba_dppy
import math


@numba_dppy.kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
def l2_distance_kernel(a, b, c):
    i = numba_dppy.get_global_id(0)
    j = numba_dppy.get_global_id(1)
    sub = a[i, j] - b[i, j]
    sq = sub ** 2
    numba_dppy.atomic.add(c, 0, sq)


def l2_distance(*args):
    a, b, distance = args
    with dpctl.device_context(base_l2_distance.get_device_selector()):
        l2_distance_kernel[(a.shape[0], a.shape[1]), numba_dppy.DEFAULT_LOCAL_SIZE](a, b, distance)

    distance_np = np.asarray([0.0])
    distance.usm_data.copy_to_host(distance_np.view("u1"))

    return math.sqrt(distance_np)


base_l2_distance.run("l2 distance", l2_distance)
