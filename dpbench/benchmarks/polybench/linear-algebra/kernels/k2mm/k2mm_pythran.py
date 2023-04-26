# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:])
def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D
