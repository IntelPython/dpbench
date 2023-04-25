# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(
    x1: dc.float64[N],
    x2: dc.float64[N],
    y_1: dc.float64[N],
    y_2: dc.float64[N],
    A: dc.float64[N, N],
):
    x1 += A @ y_1
    x2 += y_2 @ A
