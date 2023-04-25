# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N,), dtype=np.uint8)
    return data
