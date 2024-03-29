# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(C_in, C_out, H, K, N, W):
    from numpy.random import default_rng

    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out,), dtype=np.float32)
    return input, weights, bias
