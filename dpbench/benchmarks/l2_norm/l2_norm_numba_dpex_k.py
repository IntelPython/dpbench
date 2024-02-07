# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex as dpex


@dpex.kernel
def l2_norm_kernel(a, d):
    i = dpex.get_global_id(0)
    a_rows = a.shape[1]
    d[i] = 0.0
    for k in range(a_rows):
        d[i] += a[i, k] * a[i, k]
    d[i] = math.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[dpex.Range(a.shape[0])](a, d)
