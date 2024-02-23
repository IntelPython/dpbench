# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba_mlir.kernel as nb
import numpy as np


@nb.kernel(gpu_fp64_truncate="auto")
def _pairwise_distance_kernel(X1, X2, D):
    i = nb.get_global_id(1)
    j = nb.get_global_id(0)

    X1_cols = X1.shape[1]

    d = 0.0
    for k in range(X1_cols):
        tmp = X1[i, k] - X2[j, k]
        d += tmp * tmp
    D[i, j] = np.sqrt(d)


def pairwise_distance(X1, X2, D):
    _pairwise_distance_kernel[
        (X2.shape[0], X1.shape[0]), nb.DEFAULT_LOCAL_SIZE
    ](X1, X2, D)
