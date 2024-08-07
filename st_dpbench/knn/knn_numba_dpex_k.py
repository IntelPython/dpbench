# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import sqrt
from timeit import default_timer

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

now = default_timer


@dpex.kernel
def _knn_kernel(  # noqa: C901: TODO: can we simplify logic?
    item: kapi.Item,
    train,
    train_labels,
    test,
    k,
    classes_num,
    train_size,
    predictions,
    votes_to_classes_lst,
    data_dim,
):
    dtype = train.dtype
    i = item.get_id(0)
    # here k has to be 5 in order to match with numpy
    queue_neighbors = kapi.PrivateArray(
        shape=(5, 2), dtype=dtype, fill_zeros=True
    )

    for j in range(k):
        x1 = train[j]
        x2 = test[i]

        distance = dtype.type(0.0)
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = sqrt(distance)

        queue_neighbors[j, 0] = dist
        queue_neighbors[j, 1] = train_labels[j]

    for j in range(k):
        new_distance = queue_neighbors[j, 0]
        new_neighbor_label = queue_neighbors[j, 1]
        index = j

        while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
            queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
            queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

            index = index - 1

            queue_neighbors[index, 0] = new_distance
            queue_neighbors[index, 1] = new_neighbor_label

    for j in range(k, train_size):
        x1 = train[j]
        x2 = test[i]

        distance = dtype.type(0.0)
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = sqrt(distance)

        if dist < queue_neighbors[k - 1, 0]:
            queue_neighbors[k - 1, 0] = dist
            queue_neighbors[k - 1, 1] = train_labels[j]
            new_distance = queue_neighbors[k - 1, 0]
            new_neighbor_label = queue_neighbors[k - 1, 1]
            index = k - 1

            while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
                queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                index = index - 1

                queue_neighbors[index, 0] = new_distance
                queue_neighbors[index, 1] = new_neighbor_label

    votes_to_classes = votes_to_classes_lst[i]

    for j in range(len(queue_neighbors)):
        votes_to_classes[int(queue_neighbors[j, 1])] += 1

    max_ind = 0
    max_value = dtype.type(0)

    for j in range(classes_num):
        if votes_to_classes[j] > max_value:
            max_value = votes_to_classes[j]
            max_ind = j

    predictions[i] = max_ind


def knn(
    x_train,
    y_train,
    x_test,
    k,
    classes_num,
    test_size,
    train_size,
    predictions,
    votes_to_classes,
    data_dim,
):
    dpex.call_kernel(
        _knn_kernel,
        kapi.Range(test_size),
        x_train,
        y_train,
        x_test,
        k,
        classes_num,
        train_size,
        predictions,
        votes_to_classes,
        data_dim,
    )


def initialize(
    test_size,
    train_size,
    data_dim,
    classes_num,
    seed_test,
    seed_train,
):
    import numpy as np
    import numpy.random as default_rng

    dtype = np.float64

    def _gen_data_x(ip_size, data_dim, seed, dtype):
        default_rng.seed(seed)
        data = default_rng.rand(ip_size, data_dim)
        return data.astype(dtype)

    def _gen_data_y(ip_size, classes_num, seed):
        default_rng.seed(seed)
        data = default_rng.randint(classes_num, size=ip_size, dtype=np.int64)
        return data

    def _gen_train_data(train_size, data_dim, classes_num, seed_train, dtype):
        return (
            _gen_data_x(train_size, data_dim, seed_train, dtype),
            _gen_data_y(train_size, classes_num, seed_train),
        )

    def _gen_test_data(test_size, data_dim, seed_test, dtype):
        return _gen_data_x(test_size, data_dim, seed_test, dtype)

    x_train, y_train = _gen_train_data(
        train_size, data_dim, classes_num, seed_train, dtype
    )
    x_test = _gen_test_data(test_size, data_dim, seed_test, dtype)
    predictions = np.empty(test_size, np.int64)
    votes_to_classes = np.zeros((test_size, classes_num), dtype)

    return (x_train, y_train, x_test, predictions, votes_to_classes)


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


(x_train, y_train, x_test, predictions, votes_to_classes) = initialize(
    16777216, 1024, 16, 3, 777777, 0
)

d_x_train = copy_to_func()(x_train)
d_y_train = copy_to_func()(y_train)
d_x_test = copy_to_func()(x_test)
d_predictions = copy_to_func()(predictions)
d_votes_to_classes = copy_to_func()(votes_to_classes)

knn(
    d_x_train,
    d_y_train,
    d_x_test,
    5,
    3,
    16777216,
    1024,
    d_predictions,
    d_votes_to_classes,
    16,
)

t0 = now()
knn(
    d_x_train,
    d_y_train,
    d_x_test,
    5,
    3,
    16777216,
    1024,
    d_predictions,
    d_votes_to_classes,
    16,
)
t1 = now()

print("TIME: {:10.6f}".format((t1 - t0)), flush=True)
