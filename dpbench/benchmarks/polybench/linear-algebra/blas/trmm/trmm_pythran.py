# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


# pythran export kernel(float64, float64[:,:], float64[:,:])
def kernel(alpha, A, B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
    B *= alpha
