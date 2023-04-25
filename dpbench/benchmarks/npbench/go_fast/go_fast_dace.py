# SPDX-FileCopyrightText: 2012-2020 Anaconda, Inc. and others
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://numba.readthedocs.io/en/stable/user/5minguide.html

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def go_fast(a: dc.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace
