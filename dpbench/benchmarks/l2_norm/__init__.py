# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
l2-norm calculation of n vectors

Input
---------
npoints: int
    number of vectors
dims: int
    dimension of single vector
seed: int
    random seed to generate random number

Output
-------
d: array
    l2 norm of each vector

Method
------
    ||Vj||2=sqrt(sum(Xj[i]*Xj[i]))
    here i is 0->dims, j is 0->npoints
"""
