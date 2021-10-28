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
    with dpctl.device_context(base_l2_distance.get_device_selector()):
        l2_distance_kernel[
            (args[0].shape[0], args[0].shape[1]), numba_dppy.DEFAULT_LOCAL_SIZE
        ](args[0], args[1], args[2])


base_l2_distance.run("l2 distance", l2_distance)
