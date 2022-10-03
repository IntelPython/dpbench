# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex
import numpy as np
from numba_dpex import DEFAULT_LOCAL_SIZE, kernel

atomic_add = numba_dpex.atomic.add


@kernel(access_types={"read_only": ["a"], "write_only": ["d"]})
def l2_norm_kernel(a, d):
    i = numba_dpex.get_global_id(0)
    O = a.shape[1]
    d[i] = 0.0
    for k in range(O):
        d[i] += a[i, k] * a[i, k]
    d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[a.shape[0], DEFAULT_LOCAL_SIZE](a, d)
