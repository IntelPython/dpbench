# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .knn_sycl._knn_sycl import knn as knn_sycl

__all__ = ["knn_sycl"]
