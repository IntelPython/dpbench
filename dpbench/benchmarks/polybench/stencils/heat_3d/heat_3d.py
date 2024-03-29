# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N), dtype=datatype
    )
    B = np.copy(A)

    return A, B
