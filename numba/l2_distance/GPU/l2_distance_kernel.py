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
    import numba_dppy
    from numba_dppy import kernel, get_global_id, atomic, DEFAULT_LOCAL_SIZE

    atomic_add = atomic.add

    @kernel(access_types={"read_only": ["a", "b"], "write_only": ["c"]})
    def l2_distance_kernel(a, b, c):
        i = numba_dppy.get_global_id(0)
        j = numba_dppy.get_global_id(1)
        sub = a[i, j] - b[i, j]
        sq = sub ** 2
        atomic_add(c, 0, sq)


else:
    from numba_dpcomp.mlir.kernel_impl import (
        kernel,
        get_global_id,
        atomic,
        DEFAULT_LOCAL_SIZE,
    )

    atomic_add = atomic.add

    @kernel
    def l2_distance_kernel(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        sub = a[i, j] - b[i, j]
        sq = sub ** 2
        atomic_add(c, 0, sq)


def l2_distance(a, b, distance):
    with dpctl.device_context(get_device_selector(is_gpu=True)):
        print("before", flush=True, file=sys.stderr)
        l2_distance_kernel[(a.shape[0], a.shape[1]), DEFAULT_LOCAL_SIZE](a, b, distance)
        print("after", flush=True, file=sys.stderr)

    if int(distance) >= 0:
        return math.sqrt(distance)
    else:
        return 0.0


base_l2_distance.run("l2 distance kernel", l2_distance)
