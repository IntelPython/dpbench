# SPDX-FileCopyrightText: 2012-2020 Anaconda, Inc. and others
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True, fastmath=True)
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace
