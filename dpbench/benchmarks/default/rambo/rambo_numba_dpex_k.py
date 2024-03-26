# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import cos, log, pi, sin, sqrt

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi


@dpex.kernel
def _rambo(item: kapi.Item, C1, F1, Q1, nout, output):
    dtype = C1.dtype
    i = item.get_id(0)
    for j in range(nout):
        C = dtype.type(2.0) * C1[i, j] - dtype.type(1.0)
        S = sqrt(dtype.type(1) - C * C)
        F = dtype.type(2.0 * pi) * F1[i, j]
        Q = -log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q * S * sin(F)
        output[i, j, 2] = Q * S * cos(F)
        output[i, j, 3] = Q * C


def rambo(nevts, nout, C1, F1, Q1, output):
    dpex.call_kernel(_rambo, kapi.Range(nevts), C1, F1, Q1, nout, output)
