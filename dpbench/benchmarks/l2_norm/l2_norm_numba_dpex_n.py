# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
from numba_dpex import dpjit


@dpjit
def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)
