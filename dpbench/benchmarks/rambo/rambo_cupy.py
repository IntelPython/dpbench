# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def rambo(nevts, nout, C1, F1, Q1, output):
    C = 2.0 * C1 - 1.0
    S = cp.sqrt(1 - cp.square(C))
    F = 2.0 * cp.pi * F1
    Q = -cp.log(Q1)

    output[:, :, 0] = Q
    output[:, :, 1] = Q * S * cp.sin(F)
    output[:, :, 2] = Q * S * cp.cos(F)
    output[:, :, 3] = Q * C

    cp.cuda.stream.get_current_stream().synchronize()
