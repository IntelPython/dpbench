# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy
import numba


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


@numba.jit(nopython=True, parallel=False, fastmath=True)
def get_output_mom2(C1, F1, Q1, nevts, nout):
    output = numpy.empty((nevts, nout, 4))

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

    return output


def generate_points(nevts, nout):
    C1, F1, Q1 = gen_rand_data(nevts, nout)

    output_particles = get_output_mom2(C1, F1, Q1, nevts, nout)

    return output_particles


def rambo(evt_per_calc):
    ng = 4
    outint = 1

    nruns = int(outint / evt_per_calc) + 1
    for i in range(nruns):
        e = generate_points(evt_per_calc, ng)

    return e
