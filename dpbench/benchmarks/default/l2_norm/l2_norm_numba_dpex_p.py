# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
import numba as nb
from numba_dpex import dpjit


@dpjit
def l2_norm(a, d):
    for i in nb.prange(a.shape[0]):
        for k in range(a.shape[1]):
            d[i] += np.square(a[i, k])
        d[i] = np.sqrt(d[i])
