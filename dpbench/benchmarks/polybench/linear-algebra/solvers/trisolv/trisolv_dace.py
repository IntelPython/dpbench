# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N]):
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
