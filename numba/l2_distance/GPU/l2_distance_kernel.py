# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import math
import os
import sys

from device_selector import get_device_selector
import base_l2_distance


backend = os.getenv("NUMBA_BACKEND", "legacy")

if backend == "legacy":
    from numba_dppy import kernel, atomic, DEFAULT_LOCAL_SIZE
    import numba_dppy

    atomic_add = atomic.add
else:
    from numba_dpcomp.mlir.kernel_impl import kernel, atomic, DEFAULT_LOCAL_SIZE
    import numba_dpcomp.mlir.kernel_impl as numba_dppy  # this doesn't work for dppy if no explicit numba_dppy before get_global_id(0)

    atomic_add = atomic.add


@kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
def l2_distance_kernel(a, b, c):
    i = numba_dppy.get_global_id(0)
    j = numba_dppy.get_global_id(1)
    sub = a[i, j] - b[i, j]
    sq = sub ** 2
    atomic_add(c, 0, sq)


def l2_distance(a, b, distance):
    with dpctl.device_context(get_device_selector(is_gpu=True)):
        l2_distance_kernel[(a.shape[0], a.shape[1]), DEFAULT_LOCAL_SIZE](a, b, distance)
    return math.sqrt(distance)


base_l2_distance.run("l2 distance kernel", l2_distance)
