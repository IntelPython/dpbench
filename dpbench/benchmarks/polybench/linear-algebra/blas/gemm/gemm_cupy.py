# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as np


def kernel(alpha, beta, C, A, B):
    with np.cuda.Device(0):
        C[:] = alpha * A @ B + beta * C

    # with cp.cuda.Device(0):

    #     g_A = cp.asarray(A)
    #     g_B = cp.asarray(B)
    #     g_C = cp.asarray(C)

    #     g_C[:] = g_alpha * g_A @ g_B + g_beta * g_C
