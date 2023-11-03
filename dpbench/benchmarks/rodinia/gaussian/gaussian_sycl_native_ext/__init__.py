# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Sycl implementation for gaussian elimination."""

from .gaussian_sycl._gaussian_sycl import gaussian as gaussian_sycl

__all__ = ["gaussian_sycl"]
