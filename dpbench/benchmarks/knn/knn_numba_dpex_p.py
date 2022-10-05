# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

from .knn_numba_npr import knn as knn_npr


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
    knn_npr(
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
    )
