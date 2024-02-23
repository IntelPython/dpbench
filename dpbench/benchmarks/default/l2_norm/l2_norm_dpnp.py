# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np


def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1, dtype=sq.dtype)
    d[:] = np.sqrt(sum)
