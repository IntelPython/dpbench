# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

"""
Pairwise distance computation of 2 n-dim arrays

Input
---------
X1: double
    first n-dim array
X2: double
    second n-dim array
D : double
    distance matrix

Output
-------
d: array
    pairwise distance

Method
------
    D[i,j]+=sqrt((X1[i, k] - X2[j, k])^2)
    here i,j are 0->number of points, k is 0->dims
"""
