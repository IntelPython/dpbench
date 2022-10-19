# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .gpairs_dpnp import gpairs as gpairs_dpnp
from .gpairs_initialize import initialize
from .gpairs_numba_dpex_k import gpairs as gpairs_numba_dpex_k
from .gpairs_numba_dpex_p import gpairs as gpairs_numba_dpex_p
from .gpairs_numba_n import gpairs as gpairs_numba_n
from .gpairs_numba_npr import gpairs as gpairs_numba_npr
from .gpairs_numpy import gpairs as gpairs_numpy

__all__ = [
    "initialize",
    "gpairs_dpnp",
    "gpairs_numba_dpex_k",
    "gpairs_numba_dpex_p",
    "gpairs_numba_n",
    "gpairs_numba_npr",
    "gpairs_numpy",
]
