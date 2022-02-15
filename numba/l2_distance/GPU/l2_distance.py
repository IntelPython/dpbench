# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_l2_distance
import numpy as np
import os
from device_selector import get_device_selector

backend = os.getenv("NUMBA_BACKEND", "legacy")
if backend == "legacy":
    import numba as nb
    @nb.njit(parallel=True, fastmath=True)
    def l2_distance_kernel(a, b):
        sub = a - b # this line is offloaded
        sq = np.square(sub) # this line is offloaded
        sum = np.sum(sq)
        d = np.sqrt(sum)
        return d
else:
    import numba_dpcomp as nb
    @nb.njit(parallel=True, fastmath=True, enable_gpu_pipeline=True)
    def l2_distance_kernel(a, b):
        sub = a - b
        sq = np.square(sub)
        sum = np.sum(sq)
        d = np.sqrt(sum)
        return d


def l2_distance(a, b, distance):
    with dpctl.device_context(get_device_selector()):
        return l2_distance_kernel(a, b)


base_l2_distance.run("l2 distance", l2_distance)
