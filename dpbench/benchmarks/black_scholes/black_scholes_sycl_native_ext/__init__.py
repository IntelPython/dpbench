# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .black_scholes_sycl._black_scholes_sycl import (
    black_scholes as black_scholes_sycl,
)

__all__ = ["black_scholes_sycl"]
