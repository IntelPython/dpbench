# Copyright 2022 Intel Corp.
# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import cos, log, pi, sin, sqrt

import numba_dpex as nbdx


@nbdx.kernel
def _rambo(C1, F1, Q1, nout, output):
    i = nbdx.get_global_id(0)
    for j in range(nout):
        C = 2.0 * C1[i, j] - 1.0
        S = sqrt(1 - C * C)
        F = 2.0 * pi * F1[i, j]
        Q = -log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q * S * sin(F)
        output[i, j, 2] = Q * S * cos(F)
        output[i, j, 3] = Q * C


def rambo(nevts, nout, C1, F1, Q1, output):
    _rambo[nevts,](
        C1,
        F1,
        Q1,
        nout,
        output,
    )
