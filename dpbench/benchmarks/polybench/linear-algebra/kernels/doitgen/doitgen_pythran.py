# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


# pythran export kernel(int, int, int, float64[:,:,:], float64[:,:])
def kernel(NR, NQ, NP, A, C4):
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    A[:] = (A.reshape(NR, NQ, 1, NP) @ C4).reshape(NR, NQ, NP)
