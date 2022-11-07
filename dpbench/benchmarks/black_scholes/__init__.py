# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

from .black_scholes_initialize import initialize
from .black_scholes_numpy import black_scholes as black_scholes_numpy
from .black_scholes_python import black_scholes as black_scholes_python

__all__ = [
    "initialize",
    "black_scholes_numpy",
    "black_scholes_python",
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
