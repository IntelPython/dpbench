# SPDX-FileCopyrightText: 2017 Nicolas P. Rougier
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# more information at https://github.com/rougier/numpy-book

import dask.array as da
import numpy as np


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = da.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = da.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:, None] * 1j
    N = da.zeros(C.shape, dtype=np.int64)
    Z = da.zeros(C.shape, dtype=np.complex64)
    # dC = da.from_array(C, chunks='auto')
    # dN = da.from_array(N, chunks='auto')
    # dZ = da.from_array(Z, chunks='auto')
    for n in range(maxiter):
        # I = da.less(abs(Z), horizon)
        I = abs(Z) < horizon
        N[I] = n
        # Z[I] = Z[I]**2 + C[I]
        # Z[I] **=2
        # Z[I] += C[I]
        Z[I] = Z[I] * Z[I] + C[I]
    N[N == maxiter - 1] = 0
    Z.compute()
    N.compute()
    return Z, N
