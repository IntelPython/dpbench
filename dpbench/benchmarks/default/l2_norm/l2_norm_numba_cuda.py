# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil, sqrt

from numba import cuda


@cuda.jit
def l2_norm_kernel(a, d):
    i = cuda.grid(1)

    a_rows = a.shape[1]
    d[i] = 0.0
    for k in range(a_rows):
        d[i] += a[i, k] * a[i, k]
    d[i] = sqrt(d[i])


def l2_norm(a, d):
    nthreads = 256
    nblocks = ceil(a.shape[0] // nthreads)

    l2_norm_kernel[nblocks, nthreads](a, d)
