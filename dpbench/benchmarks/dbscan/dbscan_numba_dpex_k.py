# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0


import numba_dpex as nb
import numpy as np
from numba import int64, jit
from numba.experimental import jitclass

queue_spec = [
    ("capacity", int64),
    ("head", int64),
    ("tail", int64),
    ("values", int64[:]),
]


@jitclass(queue_spec)
class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.head = self.tail = 0
        self.values = np.empty(capacity, dtype=np.int64)

    def resize(self, new_capacity):
        self.capacity = new_capacity
        self.tail = min(self.tail, new_capacity)

        new_values = np.empty(new_capacity, dtype=np.int64)
        new_values[: self.tail] = self.values[: self.tail]
        self.values = new_values

    def push(self, value):
        if self.tail == self.capacity:
            self.resize(2 * self.capacity)

        self.values[self.tail] = value
        self.tail += 1

    def pop(self):
        if self.head < self.tail:
            self.head += 1
            return self.values[self.head - 1]

        return -1

    def empty(self):
        return self.head == self.tail

    @property
    def size(self):
        return self.tail - self.head


NOISE = -1
UNDEFINED = -2
DEFAULT_QUEUE_CAPACITY = 10


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

        qu = Queue(DEFAULT_QUEUE_CAPACITY)
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


def dbscan(n_samples, n_features, data, eps, min_pts, assignments):
    indices_list = np.empty(n_samples * n_samples, dtype=np.int64)
    sizes = np.zeros(n_samples, dtype=np.int64)
    get_neighborhood[n_samples, nb.DEFAULT_LOCAL_SIZE](
        n_samples,
        n_features,
        data,
        eps,
        indices_list,
        sizes,
        assignments,
        1,
        n_samples,
    )
    return compute_clusters(
        n_samples, min_pts, assignments, sizes, indices_list
    )
