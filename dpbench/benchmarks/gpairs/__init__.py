# SPDX-FileCopyrightText: 2023 Intel Corporation
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

from .gpairs_dpnp import gpairs as gpairs_dpnp
from .gpairs_initialize import initialize
from .gpairs_numba_dpex_k import gpairs as gpairs_numba_dpex_k
from .gpairs_numba_dpex_p import gpairs as gpairs_numba_dpex_p
from .gpairs_numba_n import gpairs as gpairs_numba_n
from .gpairs_numba_npr import gpairs as gpairs_numba_npr
from .gpairs_numpy import gpairs as gpairs_numpy
from .gpairs_sycl_native_ext import gpairs_sycl

__all__ = [
    "initialize",
    "gpairs_dpnp",
    "gpairs_numba_dpex_k",
    "gpairs_numba_dpex_p",
    "gpairs_numba_n",
    "gpairs_numba_npr",
    "gpairs_numpy",
    "gpairs_sycl",
]
