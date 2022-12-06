# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .black_scholes_initialize import initialize

# List of all available benchmarks, each element must either a postfix string
# (in this case specific benchmark module name will be deduced from postfix and
# benchmark name) or 2-element tuple where first element is potfix and second is
# module name.
all_benchmarks = [
    "dpnp",
    "numba_dpex_k",
    "numba_dpex_p",
    "numba_dpex_n",
    "numba_n",
    "numba_np",
    "numba_npr",
    "numpy",
    "python",
    ("sycl", "black_scholes_sycl_native_ext.black_scholes_sycl"),
]

__all__ = [
    "initialize",
    "all_benchmarks",
]

"""
Documentation for black_scholes function

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
