# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi


@dpex.kernel
def l2_norm_kernel(item: kapi.Item, a, d):
    i = item.get_id(0)
    a_rows = a.shape[1]
    d[i] = 0.0
    for k in range(a_rows):
        d[i] += a[i, k] * a[i, k]
    d[i] = math.sqrt(d[i])


def l2_norm(a, d):
    dpex.call_kernel(l2_norm_kernel, kapi.Range(a.shape[0]), a, d)
