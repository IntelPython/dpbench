# SPDX-FileCopyrightText: 2017 Nicolas P. Rougier
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# more information at https://github.com/rougier/numpy-book

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True, fastmath=True)
def linspace(start, stop, num, dtype):
    X = np.empty((num,), dtype=dtype)
    dist = (stop - start) / (num - 1)
    for i in nb.prange(num):
        X[i] = start + i * dist
    return X


@nb.jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    # X = np.linspace(xmin, xmax, xn, dtype=np.float64)
    # Y = np.linspace(ymin, ymax, yn, dtype=np.float64)
    X = linspace(xmin, xmax, xn, dtype=np.float64)
    Y = linspace(ymin, ymax, yn, dtype=np.float64)
    # C = X + Y[:,None]*1j
    C = X + np.reshape(Y, (yn, 1)) * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(maxiter):
        # I = np.less(abs(Z), horizon)
        I = np.less(np.absolute(Z), horizon)  # noqa: E741 math variable
        # N[I] = n
        for j in nb.prange(C.shape[0]):
            for k in nb.prange(C.shape[1]):
                if I[j, k]:
                    N[j, k] = n
                    # Z[I] = Z[I]**2 + C[I]
                    # for j in nb.prange(C.shape[0]):
                    #    for k in nb.prange(C.shape[1]):
                    #:        if I[j, k]:
                    Z[j, k] = Z[j, k] ** 2 + C[j, k]
    # N[N == maxiter-1] = 0
    for j in nb.prange(C.shape[0]):
        for k in nb.prange(C.shape[1]):
            if N[j, k] == maxiter - 1:
                N[j, k] = 0
    return Z, N
