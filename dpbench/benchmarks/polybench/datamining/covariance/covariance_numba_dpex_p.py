# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dpnp as np
import numba as nb
from numba_dpex import dpjit


@dpjit
def kernel(M, float_n, data):
    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in nb.prange(M):
    #     for j in nb.prange(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in nb.prange(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov
