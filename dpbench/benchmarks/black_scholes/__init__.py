# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
"""
The Black-Scholes program computes the price of a portfolio of
options using partial differential equations.
The entire computation performed by Black-Scholes is data-parallel
where each option can be priced independent of other options.

Input
---------
nopt: int
    number of options
price, strike, t: double
    vectors representing different components of portfolio
rate, volatility: int
    scalars used for price computation

Output
-------
call, put: double
    output vectors
"""

from .black_scholes_dpnp import black_scholes as black_scholes_dpnp
from .black_scholes_initialize import initialize
from .black_scholes_numba_dpex_k import (
    black_scholes as black_scholes_numba_dpex_k,
)
from .black_scholes_numba_dpex_n import (
    black_scholes as black_scholes_numba_dpex_n,
)
from .black_scholes_numba_dpex_p import (
    black_scholes as black_scholes_numba_dpex_p,
)
from .black_scholes_numba_n import black_scholes as black_scholes_numba_n
from .black_scholes_numba_np import black_scholes as black_scholes_numba_np
from .black_scholes_numba_npr import black_scholes as black_scholes_numba_npr
from .black_scholes_numpy import black_scholes as black_scholes_numpy
from .black_scholes_python import black_scholes as black_scholes_python
from .black_scholes_sycl_native_ext import black_scholes_sycl

__all__ = [
    "initialize",
    "black_scholes_dpnp",
    "black_scholes_numba_dpex_k",
    "black_scholes_numba_dpex_n",
    "black_scholes_numba_dpex_p",
    "black_scholes_numba_n",
    "black_scholes_numba_np",
    "black_scholes_numba_npr",
    "black_scholes_numpy",
    "black_scholes_python",
    "black_scholes_sycl",
]
