# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

import numba_dpex as nb
import numpy as np

from numba import jit

UNDEFINED = -2


@nb.kernel(
    access_types={
        "read_only": ["data"],
        "write_only": ["assignments", "ind_lst"],
        "read_write": ["sz_lst"],
    }
)
def get_neighborhood(
    n, dim, data, eps, ind_lst, sz_lst, assignments, block_size, nblocks
):
    i = nb.get_global_id(0)

    start = i * block_size
    stop = n if i + 1 == nblocks else start + block_size
    for j in range(start, stop):
        assignments[j] = UNDEFINED

    eps2 = eps * eps
    block_size1 = 256
    nblocks1 = n // block_size1 + int(n % block_size1 > 0)
    for ii in range(nblocks1):
        i1 = ii * block_size1
        i2 = n if ii + 1 == nblocks1 else i1 + block_size1
        for j in range(start, stop):
            for k in range(i1, i2):
                dist = 0.0
                for m in range(dim):
                    diff = data[k * dim + m] - data[j * dim + m]
                    dist += diff * diff
                if dist <= eps2:
                    size = sz_lst[j]
                    ind_lst[j * n + size] = k
                    # dist_lst[j * n + size] = dist
                    sz_lst[j] = size + 1


def dbscan(n, dim, data, eps, min_pts, assignments):
    indices_list = np.empty(n * n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    get_neighborhood[n, nb.DEFAULT_LOCAL_SIZE](
        n, dim, data, eps, indices_list, sizes, assignments, 1, n
    )
