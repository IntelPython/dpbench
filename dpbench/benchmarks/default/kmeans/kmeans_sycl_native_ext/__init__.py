# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .kmeans_sycl._kmeans_sycl import kmeans as kmeans_sycl

__all__ = ["kmeans_sycl"]
