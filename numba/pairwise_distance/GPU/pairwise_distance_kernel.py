# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pair_wise
import numpy as np
# import numba_dppy
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id


@kernel
def pairwise_python(X1, X2, D):
    i = get_global_id(0)

    N = X2.shape[0]
    O = X1.shape[1]
    for j in range(N):
        d = 0.0
        for k in range(O):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)


def pw_distance(X1, X2, D):
    with dpctl.device_context(base_pair_wise.get_device_selector()):
        # pairwise_python[X1.shape[0],numba_dppy.DEFAULT_LOCAL_SIZE](X1, X2, D)
        pairwise_python[X1.shape[0], 128](X1, X2, D)


base_pair_wise.run("Pairwise Distance Kernel", pw_distance)
