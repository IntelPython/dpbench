# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import dpctl
import numpy as np
from numba import jit
# import numba_dppy
from numba_dpcomp.mlir.kernel_impl import kernel, get_global_id, DEFAULT_LOCAL_SIZE
import base_dbscan
import utils

NOISE = -1
UNDEFINED = -2
DEFAULT_QUEUE_CAPACITY = 10


# @numba_dppy.kernel(
#     access_types={
#         "read_only": ["data"],
#         "write_only": ["assignments", "ind_lst"],
#         "read_write": ["sz_lst"],
#     }
# )
@kernel
def get_neighborhood(
    n, dim, data, eps, ind_lst, sz_lst, assignments, block_size, nblocks
):
    i = get_global_id(0)

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


# def call_ocl(n, dim, data, eps, ind_lst, dist_lst, sz_lst, assignments):
#     device_env = numba_dppy.runtime.get_gpu_device()

#     ddata = device_env.copy_array_to_device(data)
#     dsz_lst = device_env.copy_array_to_device(sz_lst)

#     dind_lst = numba_dppy.DeviceArray(device_env.get_env_ptr(), ind_lst)
#     ddist_lst = numba_dppy.DeviceArray(device_env.get_env_ptr(), dist_lst)
#     dassignments = numba_dppy.DeviceArray(device_env.get_env_ptr(), assignments)

#     block_size = 1  # nBlocks to be equal to n on GPU
#     nblocks = n // block_size + int(n % block_size > 0)

#     get_neighborhood[nblocks,](n, dim, ddata, eps, dind_lst, ddist_lst, dsz_lst, dassignments,
#                                 block_size, nblocks)

#     device_env.copy_array_from_device(dind_lst)
#     device_env.copy_array_from_device(dsz_lst)
#     device_env.copy_array_from_device(dassignments)


@jit(nopython=True)
def compute_clusters(n, min_pts, assignments, sizes, indices_list):
    nclusters = 0
    nnoise = 0
    for i in range(n):
        if assignments[i] != UNDEFINED:
            continue
        size = sizes[i]
        if size < min_pts:
            nnoise += 1
            assignments[i] = NOISE
            continue
        nclusters += 1
        assignments[i] = nclusters - 1

        qu = utils.Queue(DEFAULT_QUEUE_CAPACITY)
        for j in range(size):
            next_point = indices_list[i * n + j]
            if assignments[next_point] == NOISE:
                nnoise -= 1
                assignments[next_point] = nclusters - 1
            elif assignments[next_point] == UNDEFINED:
                assignments[next_point] = nclusters - 1
                qu.push(next_point)

        while not qu.empty():
            cur_point = qu.pop()
            size = sizes[cur_point]
            assignments[cur_point] = nclusters - 1
            if size < min_pts:
                continue

            for j in range(size):
                next_point = indices_list[cur_point * n + j]
                if assignments[next_point] == NOISE:
                    nnoise -= 1
                    assignments[next_point] = nclusters - 1
                elif assignments[next_point] == UNDEFINED:
                    assignments[next_point] = nclusters - 1
                    qu.push(next_point)

    return nclusters


def dbscan(n, dim, data, eps, min_pts, assignments):
    indices_list = np.empty(n * n, dtype=np.int64)
    # distances_list = np.empty(n*n)
    sizes = np.zeros(n, dtype=np.int64)

    with dpctl.device_context(base_dbscan.get_device_selector()):
        get_neighborhood[n, DEFAULT_LOCAL_SIZE](
            n, dim, data, eps, indices_list, sizes, assignments, 1, n
        )

    # call_ocl(n, dim, data, eps, indices_list, distances_list, sizes, assignments)

    return compute_clusters(n, min_pts, assignments, sizes, indices_list)


base_dbscan.run("dbscan", dbscan)
