# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(TSTEPS, N, A):
    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2]
                + A[i - 1, 1:-1]
                + A[i - 1, 2:]
                + A[i, 2:]
                + A[i + 1, :-2]
                + A[i + 1, 1:-1]
                + A[i + 1, 2:]
            )
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0
