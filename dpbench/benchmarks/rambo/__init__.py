# Copyright 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
# SPDX-License-Identifier: Apache-2.0

"""
The Rambo algorithm is a high energy physics application
that performs phase integration to produce a flat
phase-space with constant volume for massless particles.

Input
---------
nevts: int
    number of events
nout: int
    number of out particles
C1, F1, Q1: float
    input vectors
output: float
    output of particles

Output
-------
output: float
    output vector
"""

from .rambo_initialize import initialize
from .rambo_numba_dpex_k import rambo as rambo_numba_dpex_k
from .rambo_numba_dpex_p import rambo as rambo_numba_dpex_p
from .rambo_numba_n import rambo as rambo_numba_n
from .rambo_numba_npr import rambo as rambo_numba_npr
from .rambo_numpy import rambo as rambo_numpy
from .rambo_sycl_native_ext import rambo_sycl

__all__ = [
    "initialize",
    "rambo_numba_dpex_k",
    "rambo_numba_dpex_p",
    "rambo_numba_n",
    "rambo_numba_npr",
    "rambo_numpy",
    "rambo_sycl",
]
