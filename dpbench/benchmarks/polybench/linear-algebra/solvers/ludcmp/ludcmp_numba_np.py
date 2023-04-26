# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A, b):
    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(A.shape[0]):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, A.shape[0]):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(A.shape[0]):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]

    return x, y
