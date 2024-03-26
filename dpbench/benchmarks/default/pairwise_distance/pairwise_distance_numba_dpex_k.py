# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
import numba_dpex as dpex
from numba_dpex import kernel_api as kapi


@dpex.kernel
def _pairwise_distance_kernel(item: kapi.Item, X1, X2, D):
    i = item.get_id(1)
    j = item.get_id(0)

    X1_cols = X1.shape[1]

    d = X1.dtype.type(0.0)
    for k in range(X1_cols):
        tmp = X1[i, k] - X2[j, k]
        d += tmp * tmp
    D[i, j] = np.sqrt(d)


def pairwise_distance(X1, X2, D):
    dpex.call_kernel(
        _pairwise_distance_kernel,
        kapi.Range(X2.shape[0], X1.shape[0]),
        X1,
        X2,
        D,
    )
