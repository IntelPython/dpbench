# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import dpnp as np
from numba_dpex import dpjit


@dpjit
def kernel(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (
            A[1:-1, 1:-1]
            + A[1:-1, :-2]
            + A[1:-1, 2:]
            + A[2:, 1:-1]
            + A[:-2, 1:-1]
        )
        A[1:-1, 1:-1] = 0.2 * (
            B[1:-1, 1:-1]
            + B[1:-1, :-2]
            + B[1:-1, 2:]
            + B[2:, 1:-1]
            + B[:-2, 1:-1]
        )
