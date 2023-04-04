# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The gpairs benchmark calculates a weighted histogram of the distances between
points in a 3D vector space.

Input
-----
nopt: int
    number of points
nbins: int
    number of bins used for histogram computation
x1, x2, y1, y2, z1, z2: double
    vectors representing 2 sets of 3-dim points (x1, y1, z1), (x2, y2, z2)
w1, w2: double
    vectors representing weights
rbins: double
    threshold value for each bin

Output
-------
result: double
    vector representing histogram of pair counts
"""
