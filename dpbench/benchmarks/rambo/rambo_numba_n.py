# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
import numpy as np


@nb.njit(parallel=False)
def gen_rand_data(nevts, nout):
    C1 = np.empty((nevts, nout))
    F1 = np.empty((nevts, nout))
    Q1 = np.empty((nevts, nout))

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return C1, F1, Q1


@nb.njit(parallel=False, fastmath=True)
def _rambo(C1, F1, Q1, nevts, nout, output):

    for i in nb.prange(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = np.sqrt(1 - np.square(C))
            F = 2.0 * np.pi * F1[i, j]
            Q = -np.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * np.sin(F)
            output[i, j, 2] = Q * S * np.cos(F)
            output[i, j, 3] = Q * C


def rambo(nevts, nout, output):
    C1, F1, Q1 = gen_rand_data(nevts, nout)
    _rambo(C1, F1, Q1, nevts, nout, output)
