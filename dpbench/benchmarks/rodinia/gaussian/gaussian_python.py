# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""Gaussian elimination python serial implementation."""


def gaussian(a, b, m, size, block_sizeXY, result):
    """Python serial implementation for gaussian elimination.

    Args:
         a: actual matrix.
         b: base matrix (column matrix).
         m: multiplier matrix.
         size: size for matrices(sizexsize).
         block_sizeXY: block size for parallel 2d-kernel.
         result: result of operation.
    """
    # Forward Elimination
    for t in range(size - 1):
        for i in range(t + 1, size):
            m = a[i * size + t] / a[t * size + t]
            for j in range(t, size):
                a[i * size + j] = a[i * size + j] - m * a[t * size + j]
            b[i] = b[i] - m * b[t]

    # Back Substitution
    for i in range(size):
        result[size - i - 1] = b[size - i - 1]
        for j in range(i):
            result[size - i - 1] -= (
                a[size * (size - i - 1) + (size - j - 1)] * result[size - j - 1]
            )
        result[size - i - 1] = (
            result[size - i - 1] / a[size * (size - i - 1) + (size - i - 1)]
        )
