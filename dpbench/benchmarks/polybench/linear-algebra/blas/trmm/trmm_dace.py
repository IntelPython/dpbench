# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N", "S"))

# @dc.program
# def dot(l: dc.float64[S], r: dc.float64[S]):
#     return np.add.reduce(np.multiply(l, r))


@dc.program
def kernel(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N]):
    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
    B *= alpha
