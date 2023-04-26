# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


@dc.program
def flip(A: dc.float64[M]):
    B = np.ndarray((M,), dtype=np.float64)
    for i in dc.map[0:M]:
        B[i] = A[M - 1 - i]
    return B


@dc.program
def kernel(r: dc.float64[N]):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * flip(y[:k])
        y[k] = alpha

    return y
