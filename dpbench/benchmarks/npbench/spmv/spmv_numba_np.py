# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# Sparse Matrix-Vector Multiplication (SpMV)
import numba as nb
import numpy as np


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@nb.jit(nopython=True, parallel=True, fastmath=True)
def spmv(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i] : A_row[i + 1]]
        vals = A_val[A_row[i] : A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y
