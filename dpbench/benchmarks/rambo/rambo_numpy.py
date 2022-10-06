# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def rambo(nevts, nout, C1, F1, Q1, output):
    for i in range(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = np.sqrt(1 - np.square(C))
            F = 2.0 * np.pi * F1[i, j]
            Q = -np.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * np.sin(F)
            output[i, j, 2] = Q * S * np.cos(F)
            output[i, j, 3] = Q * C
