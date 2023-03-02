# SPDX-FileCopyrightText: 2023 Intel Corporation
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

from .l2_norm_dpnp import l2_norm as l2_norm_dpnp
from .l2_norm_initialize import initialize
from .l2_norm_numba_dpex_k import l2_norm as l2_norm_numba_dpex_k
from .l2_norm_numba_dpex_n import l2_norm as l2_norm_numba_dpex_n
from .l2_norm_numba_dpex_p import l2_norm as l2_norm_numba_dpex_p
from .l2_norm_numba_n import l2_norm as l2_norm_numba_n
from .l2_norm_numba_np import l2_norm as l2_norm_numba_np
from .l2_norm_numba_npr import l2_norm as l2_norm_numba_npr
from .l2_norm_numpy import l2_norm as l2_norm_numpy
from .l2_norm_sycl_native_ext import l2_norm_sycl

__all__ = [
    "initialize",
    "l2_norm_dpnp",
    "l2_norm_numba_dpex_k",
    "l2_norm_numba_dpex_n",
    "l2_norm_numba_dpex_p",
    "l2_norm_numba_n",
    "l2_norm_numba_np",
    "l2_norm_numba_npr",
    "l2_norm_numpy",
    "l2_norm_sycl",
]
