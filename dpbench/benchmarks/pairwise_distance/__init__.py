# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .pairwise_distance_initialize import initialize
from .pairwise_distance_numba_dpex_k import (
    pairwise_distance as pairwise_distance_numba_dpex_k,
)
from .pairwise_distance_numba_dpex_p import (
    pairwise_distance as pairwise_distance_numba_dpex_p,
)
from .pairwise_distance_numba_dpex_n import (
    pairwise_distance as pairwise_distance_numba_dpex_n,
)
from .pairwise_distance_numba_npr import (
    pairwise_distance as pairwise_distance_numba_npr,
)
from .pairwise_distance_numba_np import (
    pairwise_distance as pairwise_distance_numba_np,
)
from .pairwise_distance_numpy import (
    pairwise_distance as pairwise_distance_numpy,
)
from .pairwise_distance_sycl_native_ext import pairwise_distance_sycl

__all__ = [
    "initialize",
    "pairwise_distance_numba_dpex_k",
    "pairwise_distance_numba_dpex_p",
    "pairwise_distance_numba_dpex_n",
    "pairwise_distance_numba_npr",
    "pairwise_distance_numba_np",
    "pairwise_distance_numpy",
    "pairwise_distance_sycl",
]

"""Pairwise distance computation of 2 n-dim arrays

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
