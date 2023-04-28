# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(
    test_size,
    train_size,
    data_dim,
    classes_num,
    seed_test,
    seed_train,
    types_dict,
):
    import numpy as np
    import numpy.random as default_rng

    dtype = types_dict["float"]

    def _gen_data_x(ip_size, data_dim, seed, dtype):
        default_rng.seed(seed)
        data = default_rng.rand(ip_size, data_dim)
        return data.astype(dtype)

    def _gen_data_y(ip_size, classes_num, seed):
        default_rng.seed(seed)
        data = default_rng.randint(classes_num, size=ip_size)
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
    predictions = np.empty(test_size, types_dict["int"])
    votes_to_classes = np.zeros((test_size, classes_num))

    return (x_train, y_train, x_test, predictions, votes_to_classes)
