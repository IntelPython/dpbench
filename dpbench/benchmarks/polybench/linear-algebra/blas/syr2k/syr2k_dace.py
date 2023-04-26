# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


@dc.program
def kernel(
    alpha: dc.float64,
    beta: dc.float64,
    C: dc.float64[N, N],
    A: dc.float64[N, M],
    B: dc.float64[N, M],
):
    for i in range(N):
        C[i, : i + 1] *= beta
        for k in range(M):
            C[i, : i + 1] += (
                A[: i + 1, k] * alpha * B[i, k]
                + B[: i + 1, k] * alpha * A[i, k]
            )
