# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(TMAX, NX, NY, datatype=np.float64):
    ex = np.fromfunction(
        lambda i, j: (i * (j + 1)) / NX, (NX, NY), dtype=datatype
    )
    ey = np.fromfunction(
        lambda i, j: (i * (j + 2)) / NY, (NX, NY), dtype=datatype
    )
    hz = np.fromfunction(
        lambda i, j: (i * (j + 3)) / NX, (NX, NY), dtype=datatype
    )
    _fict_ = np.fromfunction(lambda i: i, (TMAX,), dtype=datatype)

    return ex, ey, hz, _fict_
