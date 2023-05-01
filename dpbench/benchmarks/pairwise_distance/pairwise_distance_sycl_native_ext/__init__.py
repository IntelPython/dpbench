# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .pairwise_distance_sycl._pairwise_distance_sycl import (
    pairwise_distance as pairwise_distance_sycl,
)

__all__ = ["pairwise_distance_sycl"]
