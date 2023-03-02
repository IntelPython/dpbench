# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)
