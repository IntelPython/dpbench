# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba_mlir.kernel as nb
import numpy as np


@nb.kernel
def l2_norm_kernel(a, d):
    i = nb.get_global_id(0)
    a_rows = a.shape[1]
    d[i] = 0.0
    for k in range(a_rows):
        d[i] += a[i, k] * a[i, k]
    d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[a.shape[0], nb.DEFAULT_LOCAL_SIZE](a, d)
