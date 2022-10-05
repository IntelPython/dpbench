# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numba_dpex
import numpy
from numba_dpex import DEFAULT_LOCAL_SIZE, kernel


def gen_rand_data(nevts, nout):
    C1 = numpy.empty((nevts, nout))
    F1 = numpy.empty((nevts, nout))
    Q1 = numpy.empty((nevts, nout))

    numpy.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = numpy.random.rand()
            F1[i, j] = numpy.random.rand()
            Q1[i, j] = numpy.random.rand() * numpy.random.rand()

    return C1, F1, Q1


@kernel
def _rambo(C1, F1, Q1, nout, output):
    i = numba_dpex.get_global_id(0)
    for j in range(nout):
        C = 2.0 * C1[i, j] - 1.0
        S = math.sqrt(1 - C * C)
        F = 2.0 * math.pi * F1[i, j]
        Q = -math.log(Q1[i, j])

        output[i, j, 0] = Q
        output[i, j, 1] = Q * S * math.sin(F)
        output[i, j, 2] = Q * S * math.cos(F)
        output[i, j, 3] = Q * C


def rambo(nevts, nout, output):
    C1, F1, Q1 = gen_rand_data(nevts, nout)
    _rambo[nevts, numba_dpex.DEFAULT_LOCAL_SIZE](
        C1,
        F1,
        Q1,
        nout,
        output,
    )
