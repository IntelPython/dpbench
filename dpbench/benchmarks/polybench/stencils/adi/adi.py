# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)

    return u