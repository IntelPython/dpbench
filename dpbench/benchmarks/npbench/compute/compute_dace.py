# SPDX-FileCopyrightText: 2023 Stefan Behnel, Robert Bradshaw,
#   Dag Sverre Seljebotn, Greg Ewing, William Stein, Gabriel Gellner, et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import dace as dc
import numpy as np

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


@dc.program
def compute(
    array_1: dc.int64[M, N],
    array_2: dc.int64[M, N],
    a: dc.int64,
    b: dc.int64,
    c: dc.int64,
):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
