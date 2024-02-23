# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
import numba as nb
from numba_dpex import dpjit


@dpjit
def rambo(nevts, nout, C1, F1, Q1, output):
    # TODO: get rid of it once prange supports dtype
    # https://github.com/IntelPython/numba-dpex/issues/1063
    float1 = C1.dtype.type(1.0)
    float2 = C1.dtype.type(2.0)
    floatPi = C1.dtype.type(np.pi)

    for i in nb.prange(nevts):
        for j in range(nout):
            C = float2 * C1[i, j] - float1
            S = np.sqrt(float1 - np.square(C))
            F = float2 * floatPi * F1[i, j]
            Q = -np.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * np.sin(F)
            output[i, j, 2] = Q * S * np.cos(F)
            output[i, j, 3] = Q * C
