# SPDX-FileCopyrightText: 2017 Lorena A. Barba, Gilbert F. Forsyth.
# SPDX-FileCopyrightText: 2018 Barba, Lorena A., and Forsyth, Gilbert F.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.1 / ((nx - 1) * (ny - 1))
    return u, v, p, dx, dy, dt
