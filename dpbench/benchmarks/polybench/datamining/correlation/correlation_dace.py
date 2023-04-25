# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dace as dc
import numpy as np

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


@dc.program
def kernel(float_n: dc.float64, data: dc.float64[N, M]):
    mean = np.mean(data, axis=0)
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.mean(np.subtract(data, mean) ** 2, axis=0))
    stddev[stddev <= 0.1] = 1.0
    # data -= mean
    np.subtract(data, mean, out=data)
    # data /= np.sqrt(float_n) * stddev
    np.divide(data, np.sqrt(float_n) * stddev, out=data)
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        # corr[i, i+1:M] = np.transpose(data[:, i+1:M]) @ data[:, i]
        corr[i, i + 1 : M] = data[:, i] @ data[:, i + 1 : M]
        corr[i + 1 : M, i] = corr[i, i + 1 : M]

    return corr
