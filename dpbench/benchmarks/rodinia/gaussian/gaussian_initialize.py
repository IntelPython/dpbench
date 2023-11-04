# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Initialization function for matrices for gaussian elimination."""


def initialize(size, Lambda, types_dict=None):
    """Initialize the matrices based on size and type.

    Args:
        size: size for matrices(sizexsize).
        Lambda: lambda value.
        types_dict: data type of operand.

    Returns: a: actual matrix.
             b: base matrix (column matrix).
             m: multiplier matrix.
             result: result of operation.
    """
    import math

    import numpy as np

    dtype = types_dict["float"]

    coe = np.empty((2 * size - 1), dtype=dtype)
    a = np.empty((size * size), dtype=dtype)

    for i in range(size):
        coe_i = 10 * math.exp(Lambda * i)
        j = size - 1 + i
        coe[j] = coe_i
        j = size - 1 - i
        coe[j] = coe_i

    for i in range(size):
        for j in range(size):
            a[i * size + j] = coe[size - 1 - i + j]

    return (
        a,
        np.ones(size, dtype=dtype),
        np.zeros((size * size), dtype=dtype),
        np.zeros(size, dtype=dtype),
    )
