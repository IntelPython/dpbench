# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .deformable_convolution_initialize import initialize
from .deformable_convolution_numba_npr import (
    deformable_convolution as deformable_convolution_numba_npr,
)
from .deformable_convolution_numpy import (
    deformable_convolution as deformable_convolution_numpy,
)
from .deformable_convolution_sycl_native_ext import deformable_convolution_sycl

__all__ = [
    "initialize",
    "deformable_convolution_numba_npr",
    "deformable_convolution_numba_numpy",
    "deformable_convolution_sycl",
]

"""l2-norm calculation of n vectors

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
