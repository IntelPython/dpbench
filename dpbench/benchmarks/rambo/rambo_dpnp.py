# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpnp


def rambo(nevts, nout, C1, F1, Q1, output):
    for i in range(nevts):
        for j in range(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = dpnp.sqrt(1 - dpnp.square(C))
            F = 2.0 * dpnp.pi * F1[i, j]
            Q = -dpnp.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * dpnp.sin(F)
            output[i, j, 2] = Q * S * dpnp.cos(F)
            output[i, j, 3] = Q * C
