# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Gaussian elimination implementation."""

"""This is sycl and numba-dpex implementation for gaussian elimination
Input
---------
size<int_64> : Forms an input matrix of dimensions (size x size)
Output
--------
result<array<float>> : Result of the given set of linear equations using
                        gaussian elimination.
Method:
The gaussian transformations are applied to the input matrix to form the
diagonal matrix in forward elimination, and then the equations are solved
to find the result in back substitution.
"""
