# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .dbscan_sycl._dbscan_sycl import (
    dbscan as dbscan_sycl,
)

__all__ = ["dbscan_sycl"]
