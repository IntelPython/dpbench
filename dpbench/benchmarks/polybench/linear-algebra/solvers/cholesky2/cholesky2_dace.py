# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)
k = dc.symbol("k", dtype=dc.int64)


@dc.program
def triu(A: dc.float64[N, N]):
    B = np.zeros_like(A)
    for i in dc.map[0:N]:
        for j in dc.map[i + k : N]:
            B[i, j] = A[i, j]
    return B


@dc.program
def kernel(A: dc.float64[N, N]):
    A[:] = np.linalg.cholesky(A) + triu(A, k=1)
