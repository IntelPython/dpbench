# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(TSTEPS: dc.int64, A: dc.float64[N, N, N], B: dc.float64[N, N, N]):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1, 1:-1] = (
            0.125
            * (
                A[2:, 1:-1, 1:-1]
                - 2.0 * A[1:-1, 1:-1, 1:-1]
                + A[:-2, 1:-1, 1:-1]
            )
            + 0.125
            * (
                A[1:-1, 2:, 1:-1]
                - 2.0 * A[1:-1, 1:-1, 1:-1]
                + A[1:-1, :-2, 1:-1]
            )
            + 0.125
            * (
                A[1:-1, 1:-1, 2:]
                - 2.0 * A[1:-1, 1:-1, 1:-1]
                + A[1:-1, 1:-1, 0:-2]
            )
            + A[1:-1, 1:-1, 1:-1]
        )
        A[1:-1, 1:-1, 1:-1] = (
            0.125
            * (
                B[2:, 1:-1, 1:-1]
                - 2.0 * B[1:-1, 1:-1, 1:-1]
                + B[:-2, 1:-1, 1:-1]
            )
            + 0.125
            * (
                B[1:-1, 2:, 1:-1]
                - 2.0 * B[1:-1, 1:-1, 1:-1]
                + B[1:-1, :-2, 1:-1]
            )
            + 0.125
            * (
                B[1:-1, 1:-1, 2:]
                - 2.0 * B[1:-1, 1:-1, 1:-1]
                + B[1:-1, 1:-1, 0:-2]
            )
            + B[1:-1, 1:-1, 1:-1]
        )
