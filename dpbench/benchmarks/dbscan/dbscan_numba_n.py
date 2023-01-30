# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

import numba as nb
import numpy as np

NOISE = -1
UNDEFINED = -2
DEFAULT_QUEUE_CAPACITY = 10


@nb.njit()
def _queue_create(capacity):
    return (np.empty(capacity, dtype=np.int64), 0, 0)


@nb.njit()
def _queue_resize(qu, tail, new_capacity):
    tail = min(tail, new_capacity)

    new_qu = np.empty(new_capacity, dtype=np.int64)
    new_qu[:tail] = qu[:tail]
    return new_qu, tail


@nb.njit()
def _queue_push(qu, value, tail, capacity):
    if tail == capacity:
        capacity = 2 * capacity
        qu, tail = _queue_resize(qu, tail, capacity)

    qu[tail] = value
    tail += 1

    return qu, tail, capacity


@nb.njit()
def _queue_pop(qu, head, tail):
    head += 1
    return qu[head - 1], head


@nb.njit()
def _queue_empty(head, tail):
    return head == tail


@nb.njit(parallel=False, fastmath=True)
def get_neighborhood(n, dim, data, eps, ind_lst, sz_lst, assignments):
    block_size = 1
    nblocks = n // block_size + int(n % block_size > 0)
    for i in range(nblocks):
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
                        sz_lst[j] = size + 1


@nb.njit(parallel=False, fastmath=True)
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

        qu_capacity = DEFAULT_QUEUE_CAPACITY
        qu, head, tail = _queue_create(qu_capacity)
        for j in range(size):
            next_point = indices_list[i * n + j]
            if assignments[next_point] == NOISE:
                nnoise -= 1
                assignments[next_point] = nclusters - 1
            elif assignments[next_point] == UNDEFINED:
                assignments[next_point] = nclusters - 1
                qu, tail, qu_capacity = _queue_push(
                    qu, next_point, tail, qu_capacity
                )

        while not _queue_empty(head, tail):
            cur_point, head = _queue_pop(qu, head, tail)
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
                    qu, tail, qu_capacity = _queue_push(
                        qu, next_point, tail, qu_capacity
                    )

    return nclusters


def dbscan(n_samples, n_features, data, eps, min_pts, assignments):
    indices_list = np.empty(n_samples * n_samples, dtype=np.int64)
    sizes = np.zeros(n_samples, dtype=np.int64)

    get_neighborhood(
        n_samples, n_features, data, eps, indices_list, sizes, assignments
    )

    return compute_clusters(
        n_samples, min_pts, assignments, sizes, indices_list
    )
