# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba
import numpy


@numba.jit(nopython=True, parallel=False, fastmath=True)
def rambo(nevts, nout, C1, F1, Q1, output):

    for i in numba.prange(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = numpy.sqrt(1 - numpy.square(C))
            F = 2.0 * numpy.pi * F1[i, j]
            Q = -numpy.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * numpy.sin(F)
            output[i, j, 2] = Q * S * numpy.cos(F)
            output[i, j, 3] = Q * C
