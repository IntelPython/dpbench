# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(N, H, SM):
    from numpy.random import default_rng

    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x
