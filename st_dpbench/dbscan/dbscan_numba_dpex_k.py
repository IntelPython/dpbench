# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from timeit import default_timer

import dpnp as np
import numba as nb
import numba_dpex as dpex
import numpy
from numba_dpex import kernel_api as kapi

now = default_timer

NOISE = -1
UNDEFINED = -2
DEFAULT_QUEUE_CAPACITY = 10


@nb.njit()
def _queue_create(capacity):
    return (numpy.empty(capacity, dtype=numpy.int64), 0, 0)


@nb.njit()
def _queue_resize(qu, tail, new_capacity):
    tail = min(tail, new_capacity)

    new_qu = numpy.empty(new_capacity, dtype=numpy.int64)
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


@dpex.kernel
def get_neighborhood(item: kapi.Item, n, dim, data, eps, ind_lst, sz_lst):
    i = item.get_id(0)

    for j in range(n):
        dist = data.dtype.type(0.0)
        for m in range(dim):
            diff = data[i * dim + m] - data[j * dim + m]
            dist += diff * diff
        if dist <= eps:
            ind_lst[i * n + sz_lst[i]] = j
            sz_lst[i] = sz_lst[i] + 1


@nb.njit(parallel=False, fastmath=True)
def compute_clusters(  # noqa: C901: TODO: can we simplify logic?
    n, min_pts, assignments, sizes, indices_list
):
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


def dbscan(n_samples, n_features, data, eps, min_pts, measure_time):
    indices_list = np.empty_like(
        data, shape=n_samples * n_samples, dtype=np.int64
    )
    sizes = np.zeros_like(data, shape=n_samples, dtype=np.int64)

    if measure_time:
        t_i = now()

    dpex.call_kernel(
        get_neighborhood,
        kapi.Range(n_samples),
        n_samples,
        n_features,
        data,
        eps,
        indices_list,
        sizes,
    )

    if measure_time:
        t_j = now()
        print("Device Compute time = ", t_j - t_i)

    assignments = numpy.empty(n_samples, dtype=numpy.int64)
    for i in range(n_samples):
        assignments[i] = UNDEFINED

    if measure_time:
        t_i = now()

    sizes_np = np.asnumpy(sizes)
    indices_list_np = np.asnumpy(indices_list)

    if measure_time:
        t_j = now()
        print("Data transfer time = ", t_j - t_i)
        t_i = now()

    clusters = compute_clusters(
        n_samples,
        min_pts,
        assignments,
        sizes_np,
        indices_list_np,
    )

    if measure_time:
        t_j = now()
        print("Sequential host compute time = ", t_j - t_i)

    return clusters


def initialize(n_samples, n_features, centers, seed):
    from typing import NamedTuple

    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    DEFAULT_EPS = 0.6
    DEFAULT_MINPTS = 20

    class DataSize(NamedTuple):
        n_samples: int
        n_features: int

    class Params(NamedTuple):
        eps: float
        minpts: int

    OPTIMAL_PARAMS = {
        DataSize(n_samples=2**8, n_features=2): Params(eps=0.173, minpts=4),
        DataSize(n_samples=2**8, n_features=3): Params(eps=0.35, minpts=6),
        DataSize(n_samples=2**8, n_features=10): Params(eps=0.8, minpts=20),
        DataSize(n_samples=2**9, n_features=2): Params(eps=0.15, minpts=4),
        DataSize(n_samples=2**9, n_features=3): Params(eps=0.1545, minpts=6),
        DataSize(n_samples=2**9, n_features=10): Params(eps=0.7, minpts=20),
        DataSize(n_samples=2**10, n_features=2): Params(eps=0.1066, minpts=4),
        DataSize(n_samples=2**10, n_features=3): Params(eps=0.26, minpts=6),
        DataSize(n_samples=2**10, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**11, n_features=2): Params(eps=0.095, minpts=4),
        DataSize(n_samples=2**11, n_features=3): Params(eps=0.18, minpts=6),
        DataSize(n_samples=2**11, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**12, n_features=2): Params(eps=0.0715, minpts=4),
        DataSize(n_samples=2**12, n_features=3): Params(eps=0.17, minpts=6),
        DataSize(n_samples=2**12, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**13, n_features=2): Params(eps=0.073, minpts=4),
        DataSize(n_samples=2**13, n_features=3): Params(eps=0.149, minpts=6),
        DataSize(n_samples=2**13, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**14, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**14, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**14, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**15, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**15, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**15, n_features=10): Params(eps=0.6, minpts=20),
        DataSize(n_samples=2**16, n_features=2): Params(eps=0.0695, minpts=4),
        DataSize(n_samples=2**16, n_features=3): Params(eps=0.108, minpts=6),
        DataSize(n_samples=2**16, n_features=10): Params(eps=0.6, minpts=20),
    }

    X, *_ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)

    data_size = DataSize(n_samples=n_samples, n_features=n_features)
    params = OPTIMAL_PARAMS.get(
        data_size, Params(eps=DEFAULT_EPS, minpts=DEFAULT_MINPTS)
    )

    return (
        X.flatten().astype(np.float64),
        params.eps,
        params.minpts,
    )


def copy_to_func():
    """Returns the copy-method that should be used
    for copying the benchmark arguments."""

    def _copy_to_func_impl(ref_array):
        import dpnp

        if ref_array.flags["C_CONTIGUOUS"]:
            order = "C"
        elif ref_array.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "K"
        return dpnp.asarray(
            ref_array,
            dtype=ref_array.dtype,
            order=order,
            like=None,
            usm_type=None,
            sycl_queue=None,
        )

    return _copy_to_func_impl


n_samples = 16384
n_features = 10
centers = 10

X, eps, minpts = initialize(n_samples, n_features, centers, 7777777)

X_d = copy_to_func()(X)
nclusters = dbscan(n_samples, n_features, X_d, eps, minpts, measure_time=False)

nclusters = dbscan(n_samples, n_features, X_d, eps, minpts, measure_time=True)

# print(nclusters)
