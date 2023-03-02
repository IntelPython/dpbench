# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba_dpex as nbdx
import numpy as np


@nbdx.kernel
def l2_norm_kernel(a, d):
    i = nbdx.get_global_id(0)
    O = a.shape[1]
    d[i] = 0.0
    for k in range(O):
        d[i] += a[i, k] * a[i, k]
    d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[a.shape[0],](a, d)
