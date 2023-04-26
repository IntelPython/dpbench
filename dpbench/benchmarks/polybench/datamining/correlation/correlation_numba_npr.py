# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(M, float_n, data):
    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in nb.prange(M - 1):
        corr[i + 1 : M, i] = corr[i, i + 1 : M] = (
            data[:, i] @ data[:, i + 1 : M]
        )

    return corr
