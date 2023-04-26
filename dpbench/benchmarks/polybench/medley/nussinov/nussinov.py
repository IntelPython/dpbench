# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize(N, datatype=np.int32):
    seq = np.fromfunction(lambda i: (i + 1) % 4, (N,), dtype=datatype)

    return seq
