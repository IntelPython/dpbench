# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_l2_distance
import numpy as np
import numba
from device_selector import get_device_selector


@numba.njit(parallel=True, fastmath=True)
def l2_distance_kernel(a, b):
    sub = a - b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d


def l2_distance(a, b, distance):
    with dpctl.device_context(get_device_selector()):
        l2_distance_kernel(a, b)


base_l2_distance.run("l2 distance", l2_distance)
