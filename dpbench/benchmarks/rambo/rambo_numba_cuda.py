# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil, cos, log, pi, sin, sqrt

from numba import cuda


@cuda.jit
def _rambo(C1, F1, Q1, nout, output):
    dtype = C1.dtype
    i = cuda.grid(1)
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
    nthreads = 256
    nblocks = ceil(nevts // nthreads)

    _rambo[nblocks, nthreads](
        C1,
        F1,
        Q1,
        nout,
        output,
    )

    # _rambo[nevts,](
    #     C1,
    #     F1,
    #     Q1,
    #     nout,
    #     output,
    # )
