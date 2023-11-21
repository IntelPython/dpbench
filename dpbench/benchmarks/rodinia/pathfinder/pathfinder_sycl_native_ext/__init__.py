# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Sycl implementation for Pathfinder."""

from .pathfinder_sycl._pathfinder_sycl import pathfinder as pathfinder_sycl

__all__ = ["pathfinder_sycl"]
