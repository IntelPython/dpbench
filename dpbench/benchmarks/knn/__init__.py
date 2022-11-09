# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

"""
Documentation for knn function

The knn program calculates the k nearest neighbors of a training set.

Input
-----
train: double
    training set data
train_lables: double
    training set results as labels
test: double
    test set
train_nrows, test_size: int
    training and test set sizes
votes_to_classes: double
    vector representing vote labels

Output
-------
predictions: double
    vectors representing output predictions from test set
"""

from .knn_dpnp import knn as knn_dpnp
from .knn_initialize import initialize
from .knn_numba_dpex_k import knn as knn_numba_dpex_k
from .knn_numba_dpex_p import knn as knn_numba_dpex_p
from .knn_numba_npr import knn as knn_numba_npr
from .knn_python import knn as knn_python
from .knn_sycl_native_ext import knn_sycl

__all__ = [
    "initialize",
    "knn_dpnp",
    "knn_numba_dpex_k",
    "knn_numba_dpex_p",
    "knn_numba_npr",
    "knn_python",
    "knn_sycl",
]
